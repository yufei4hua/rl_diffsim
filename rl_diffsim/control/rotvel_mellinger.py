"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jp
import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ["SCIPY_ARRAY_API"] = "1"
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.sim import build_control_fns, increment_steps
from drone_controllers.mellinger.params import ForceTorqueParams
from drone_models.core import load_params

from rl_diffsim.control.controller import Controller

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from numpy.typing import NDArray

from scripts.utils import EvalRecorder


class MellingerController(Controller):
    """A copy of the onboard Mellinger controller for testing."""

    def __init__(
        self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, sim: object = None
    ):
        """Initialize the Mellinger controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            sim: For visualization purposes.
        """
        super().__init__(obs, info, config)
        self.freq = config.sim.freq
        self.algo_name = "mellinger"
        self.exp_name = "f8"

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]  # alternatively from sim.drone_mass
        self.thrust_min = drone_params["thrust_min"] * 4  # min total thrust
        self.thrust_max = drone_params["thrust_max"] * 4  # max total thrust
        params = ForceTorqueParams.load(config.sim.drone_model)
        self.rotor_vel_min, self.rotor_vel_max = (
            np.sqrt(params.thrust_min / params.rpm2thrust[2]),
            np.sqrt(params.thrust_max / params.rpm2thrust[2]),
        )

        # # Set trajectory parameters
        self.n_samples = 1
        self.samples_dt = 0.1
        self.trajectory_time = 5.0
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )
        self._tick = 0

        # Figure-8 trajectory
        # Create the figure eight trajectory
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        radius = 1  # Radius for the circles
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        y = np.zeros_like(t)  # x is 0 everywhere
        z = radius / 2 * np.sin(2 * t) + 1.5  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T

        # build controller functions
        self.state_freq = 50
        self.attitude_freq = 500
        self.force_torque_freq = 500
        foo_sim = Sim(
            n_worlds=1,
            n_drones=1,
            drone_model=config.sim.drone_model,
            physics=Physics.first_principles,
            control=Control.state,
            device="cpu",
            freq=self.freq,
            state_freq=self.state_freq,
            attitude_freq=self.attitude_freq,
            force_torque_freq=self.force_torque_freq,
        )  # Initialize sim data for controllers
        self.data = foo_sim.data
        self.control_pipeline = build_control_fns(Control.state, Physics.first_principles)
        self.control_pipeline += (increment_steps,)

        @jax.jit
        def ctrl_step(data: SimData) -> SimData:
            for fn in self.control_pipeline:
                data = fn(data)
            return data

        self._ctrl_step = ctrl_step

        # warm up jax controllers
        for _ in range(2):
            self._ctrl_step(self.data)

        self._finished = False

        # Initialize evaluation recorder
        self.eval_recorder = EvalRecorder()

        self.sim = sim  # for visualization only

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

        des_pos = self.trajectory[0]

        # 1. Write in obs data
        states = self.data.states
        controls = self.data.controls
        pos = jp.array(obs["pos"][None, None, :], dtype=jp.float32)
        quat = jp.array(obs["quat"][None, None, :], dtype=jp.float32)
        vel = jp.array(obs["vel"][None, None, :], dtype=jp.float32)
        ang_vel = jp.array(obs["ang_vel"][None, None, :], dtype=jp.float32)
        states = states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
        # 2. Set desired position command
        cmd = jp.broadcast_to(
            jp.concatenate([jp.array(des_pos, dtype=jp.float32), jp.zeros(10, dtype=jp.float32)]),
            self.data.controls.state.cmd.shape,
        )
        controls = controls.replace(state=controls.state.replace(staged_cmd=cmd))
        # # test cmd on attitude
        # att_cmd = jp.broadcast_to(
        #     jp.array(
        #         [0.0, 0.0, 0.0, self.drone_mass * 9.86], dtype=jp.float32
        #     ),
        #     self.data.controls.attitude.cmd.shape,
        # )
        # controls = controls.replace(attitude=controls.attitude.replace(staged_cmd=att_cmd))
        self.data = self.data.replace(states=states, controls=controls)
        # 3. Step controllers
        self.data = self._ctrl_step(self.data)
        # 4. Extract command outputs
        # rpyt_cmd = self.data.controls.attitude.cmd[0, 0, :] # TODO: publish this action to estimator, or use legacy
        rotor_vel = self.data.controls.rotor_vel[0, 0, :]
        # print(f"Rotor Vel: {rotor_vel}")
        # print(f"Onboard: {self.sim.data.controls.rotor_vel}")
        # print(f"Diff : {rotor_vel - self.sim.data.controls.rotor_vel}")
        actions = rotor_vel
        return np.array(actions, dtype=np.float32)

    @staticmethod
    def _ctrl_step(data: SimData) -> SimData:
        """Dummy ctrl step function to be replaced by build_control_fns."""
        rotor_vel = jp.zeros_like(data.controls.rotor_vel)
        return data.replace(controls=data.controls.replace(rotor_vel=rotor_vel))

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
