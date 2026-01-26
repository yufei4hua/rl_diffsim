"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ["SCIPY_ARRAY_API"] = "1"
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params

# from rl_diffsim.bptt.bptt_agent_deterministic import Agent
from rl_diffsim.control.controller import Controller
from rl_diffsim.ppo.ppo_agent import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray

from scripts.utils import EvalRecorder


class AttitudeRL(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(
        self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, sim: object = None
    ):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            sim: For visualization purposes.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        drone_params = load_params(config.env.physics, config.env.drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass
        self.thrust_min = drone_params["thrust_min"] * 4  # min total thrust
        self.thrust_max = drone_params["thrust_max"] * 4  # max total thrust

        # # Set trajectory parameters
        self.n_samples = 10
        self.samples_dt = 0.1
        self.trajectory_time = 10.0
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )
        self._tick = 0

        # Figure-8 trajectory
        # Create the figure eight trajectory
        num_loops = 3
        self.trajectory_time = 6.0 * num_loops
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi * num_loops, n_steps)
        radius = 1  # Radius for the circles
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        y = np.zeros_like(t)  # x is 0 everywhere
        z = radius / 2 * np.sin(2 * t) + 1.25  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T

        # Load RL policy
        self.algo_name = "ppo"
        self.exp_name = "f8"
        model_path = (
            Path(__file__).parents[2] / f"saves/{self.algo_name}_{self.exp_name}_model.ckpt"
        )
        with open(model_path, "rb") as f:
            import pickle

            params = pickle.load(f)
            hidden_size = params["actor"]["params"]["Dense_0"]["kernel"].shape[1]
            num_layers = len(params["actor"]["params"].keys()) - 1
            if "actor_logstd" in params["actor"]["params"].keys():
                num_layers -= 1
            obs_dim = params["actor"]["params"]["Dense_0"]["kernel"].shape[0]
            act_dim = params["actor"]["params"][f"Dense_{num_layers}"]["kernel"].shape[1]
        agent = Agent.create(
            key=jax.random.PRNGKey(0), obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size
        )
        self.agent = agent.replace(actor_states=agent.actor_states.replace(params=params["actor"]))
        self.last_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # warm up jax policy
        obs_rl = self._obs_rl(obs)
        obs_rl = jp.array([obs_rl])
        for _ in range(4):
            self.agent.get_action_mean(self.agent.actor_states.params, obs_rl)

        self._finished = False

        # Initialize evaluation recorder
        self.eval_recorder = EvalRecorder()

        self.sim = sim  # For visualization

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [r_des, p_des, y_des, t_des] as a numpy array.
        """
        i = min(self._tick, self.trajectory.shape[0] - 1)
        if i == self.trajectory.shape[0] - 1:  # Maximum duration reached
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_rl = jp.array([obs_rl])
        act = self.agent.get_action_mean(self.agent.actor_states.params, obs_rl)
        act = np.array(act)
        act[..., 2] = 0.0  # Enforce yaw to zero
        self.last_action = act.squeeze(0)

        act = self._scale_actions(act.squeeze(0)).astype(np.float32)

        # self._render()

        return act

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Extract the relevant parts of the observation for the RL policy."""
        obs_rl_key = ["pos", "quat", "vel", "ang_vel"]
        obs_rl = {k: obs[k] for k in obs_rl_key}
        # idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        idx = self._tick + self.sample_offsets
        idx = jp.where(idx >= self.trajectory.shape[0], idx - self.trajectory.shape[0], idx)
        dpos = self.trajectory[idx] - obs["pos"]  # (n_samples, 3)
        obs_rl["local_samples"] = dpos.reshape(-1)  # (n_samples*3,)
        obs_rl["last_action"] = self.last_action  # (4,)
        # alphabetical key order: important for flax policy
        ordered_keys = sorted(obs_rl.keys())
        return np.concatenate([obs_rl[k] for k in ordered_keys], axis=-1).astype(np.float32)

    def _scale_actions(self, actions: NDArray) -> NDArray:
        """Rescale and clip actions from [-1, 1] to [action_sim_low, action_sim_high]."""
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        mean = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0], dtype=np.float32
        )
        return np.clip(actions, -1.0, 1.0) * scale + mean

    def _render(self):
        """Render the trajectory and the next waypoints."""
        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        next_trajectory = self.trajectory[idx]
        draw_line(
            self.sim,
            self.trajectory[0:-1:2, :],
            rgba=np.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            self.sim, next_trajectory, rgba=np.array([1, 0, 0, 1]), start_size=3.0, end_size=3.0
        )
        draw_points(self.sim, next_trajectory, rgba=np.array([1.0, 0, 0, 1]), size=0.01)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter and record data.

        Returns:
            True if the controller is finished, False otherwise.
        """
        # Record data with batch dimension (1, dim)
        idx = min(self._tick, self.trajectory.shape[0] - 1)
        goal = self.trajectory[idx].copy()
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        self.eval_recorder.record_step(
            action=action.copy()[None, :],
            position=obs["pos"].copy()[None, :],
            goal=goal[None, :],
            rpy=rpy[None, :],
        )

        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
        self.eval_recorder.plot_eval(save_path=f"{self.algo_name}_{self.exp_name}_deploy_plot.png")
