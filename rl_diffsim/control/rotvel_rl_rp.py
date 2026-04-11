"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
from jax import Array
from scipy.spatial.transform import Rotation as R

os.environ["SCIPY_ARRAY_API"] = "1"
from crazyflow.sim.visualize import draw_line, draw_points
from drone_controllers.mellinger.params import ForceTorqueParams
from drone_models.core import load_params

from rl_diffsim.control.controller import Controller

# from rl_diffsim.bptt.bptt_agent import Agent
from rl_diffsim.td3.td3_agent import TD3Agent as Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray

from scripts.utils import EvalRecorder


class RotvelRL(Controller):
    """Example of a controller using the collective thrust and rotor_vel interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, sim: object = None):
        """Initialize the rotor_vel controller.

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
        params = ForceTorqueParams.load(config.env.drone_model)
        self.rotor_vel_min, self.rotor_vel_max = (
            np.sqrt(params.thrust_min / params.rpm2thrust[2]),
            np.sqrt(params.thrust_max / params.rpm2thrust[2]),
        )
        self.rotor_vel_neutral = 18967.0

        # # Set trajectory parameters
        self.n_samples = 1
        self.samples_dt = 0.1
        self.trajectory_time = 5.5
        self.sample_offsets = np.array(np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int)
        self._tick = 0

        # Figure-8 trajectory
        # Create the figure eight trajectory
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        radius = 1  # Radius for the circles
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        z = np.zeros_like(t) + 1.25  # y is 0 everywhere
        y = radius / 2 * np.sin(2 * t)  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T  # (n_steps, 3)
        d_x = radius * np.cos(t) * (2 * np.pi / self.trajectory_time)
        d_z = np.zeros_like(t)
        d_y = radius * np.cos(2 * t) * (2 * np.pi / self.trajectory_time)
        self.trajectory_vel = np.array([d_x, d_y, d_z]).T  # (n_steps, 3)

        # Load RL policy
        self.algo_name = "td3"
        self.exp_name = "rprv"
        model_path = Path(__file__).parents[2] / f"saves/{self.algo_name}_{self.exp_name}_model.ckpt"
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
            key=jax.random.PRNGKey(0),
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.agent = agent.replace(actor_states=agent.actor_states.replace(params=params["actor"]))
        self.num_last_actions = 4
        init_action = np.full(act_dim, 0.0, dtype=np.float32)
        self.last_actions = np.broadcast_to(init_action[None, :], (self.num_last_actions, act_dim)).copy()

        # warm up jax policy
        obs_rl = self._obs_rl(self.trajectory[0], self.trajectory_vel[0], obs)
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

        # obs["vel"] = info["obs"]["vel"]  # override with onboard sensor data
        # obs["ang_vel"] = info["obs"]["ang_vel"]  # override with onboard sensor data

        # i = 100  # Debug: ignore trajectory and just test policy output
        goal_pos = self.trajectory[i]
        goal_vel = self.trajectory_vel[i]
        # goal_vel = np.zeros_like(self.trajectory_vel[i])
        obs_rl = self._obs_rl(goal_pos, goal_vel, obs)
        obs_rl = jp.array([obs_rl])
        act = self.agent.get_action_mean(self.agent.actor_states.params, obs_rl)
        act = jp.squeeze(act, axis=0)
        latest_action = np.asarray(act, dtype=np.float32)
        self.last_actions = np.roll(self.last_actions, shift=1, axis=0)
        self.last_actions[0] = latest_action

        act = self._scale_actions(act)

        # self._render()

        return np.asarray(act, dtype=np.float32)

    def _obs_rl(
        self, goal_pos: NDArray, goal_vel: NDArray, obs: dict[str, NDArray[np.floating]]
    ) -> NDArray[np.floating]:
        """Extract the relevant parts of the observation for the RL policy."""
        obs_rl_key = ["pos", "quat", "vel", "ang_vel"]
        obs_rl = {k: obs[k] for k in obs_rl_key}
        obs_rl["pos"] = obs["pos"] - goal_pos  # (3,)
        obs_rl["vel"] = obs_rl["vel"] - goal_vel  # (3,)
        obs_rl["rot_mat"] = R.from_quat(obs_rl.pop("quat")).as_matrix().flatten()  # (9,)
        obs_rl["last_actions"] = self.last_actions.reshape(-1)  # (num_last_actions * act_dim,)
        # alphabetical key order: important for flax policy
        ordered_keys = sorted(obs_rl.keys())
        return np.concatenate([obs_rl[k] for k in ordered_keys], axis=-1).astype(np.float32)

    @partial(jax.jit, static_argnums=0)
    def _remap_actions(self, actions: Array, low: Array, high: Array, neutral: Array | None = None) -> Array:
        """Map actions in [-1, 1] to [low, high] using wrapper-matching quadratic form."""
        if neutral is None:
            k0 = (high + low) / 2.0
            k1 = (high - low) / 2.0
            k2 = jp.zeros_like(k0)
        else:
            # Same coefficient construction used in NormalizeActions.
            k0 = neutral
            k1 = (high - low) / 2.0
            k2 = (high + low - 2.0 * neutral) / 2.0
            # Ensure monotonicity by adjusting coefficients when needed.
            is_monotonic = jp.abs(k1) >= 2.0 * jp.abs(k2)
            k2_fix_high = (neutral - low) / 2.0
            k1_fix_high = 2.0 * k2_fix_high
            k2_fix_low = (neutral - high) / 2.0
            k1_fix_low = -2.0 * k2_fix_low
            k1_fixed = jp.where(k2 > 0, k1_fix_high, k1_fix_low)
            k2_fixed = jp.where(k2 > 0, k2_fix_high, k2_fix_low)
            k1 = jp.where(is_monotonic, k1, k1_fixed)
            k2 = jp.where(is_monotonic, k2, k2_fixed)

        return k0 + k1 * actions + k2 * (actions**2)

    def _scale_actions(self, actions: Array) -> Array:
        """Rescale actions from [-1, 1] to [low, high] using optional neutral midpoint."""
        return self._remap_actions(actions, self.rotor_vel_min, self.rotor_vel_max, self.rotor_vel_neutral)

    def _render(self):
        """Render the trajectory and the next waypoints."""
        idx = np.clip(self._tick + self.sample_offsets, 0, self.trajectory.shape[0] - 1)
        next_trajectory = self.trajectory[idx]
        draw_line(
            self.sim, self.trajectory[0:-1:2, :], rgba=np.array([1, 1, 1, 0.4]), start_size=2.0, end_size=2.0
        )
        draw_line(self.sim, next_trajectory, rgba=np.array([1, 0, 0, 1]), start_size=3.0, end_size=3.0)
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
        self.eval_recorder.plot_eval(save_path=f"{self.algo_name}_{self.exp_name}_deploy_plot.png", traj_plane=[0, 1])
