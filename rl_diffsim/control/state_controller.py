"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline  # noqa: F401
from scipy.spatial.transform import Rotation as R

from rl_diffsim.control import Controller
from scripts.utils import EvalRecorder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, sim: object = None):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
            sim: The simulator object, if applicable.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        self.algo_name = "td3"
        self.exp_name = "f8state"

        # Figure-8 trajectory
        # Create the figure eight trajectory
        num_loops = 3
        self.trajectory_time = 5.5 * num_loops
        n_steps = int(np.ceil(self.trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi * num_loops, n_steps)
        radius = 1.0  # Radius for the circles
        t_dot = 2 * np.pi * num_loops / self.trajectory_time
        x = radius * np.sin(t)  # Scale amplitude for 1-meter diameter
        z = np.zeros_like(t) + 1.25  # y is 0 everywhere
        y = radius / 2 * np.sin(2 * t)  # Scale amplitude for 1-meter diameter
        self.trajectory = np.array([x, y, z]).T
        d_x = radius * np.cos(t) * t_dot
        d_z = np.zeros_like(t)
        d_y = radius * np.cos(2 * t) * t_dot
        self.trajectory_vel = np.array([d_x, d_y, d_z]).T
        dd_x = -radius * np.sin(t) * t_dot**2
        dd_y = np.zeros_like(t)
        dd_z = -2 * radius * np.sin(2 * t) * t_dot**2
        self.trajectory_acc = np.array([dd_x, dd_y, dd_z]).T

        # # Racing trajectory
        # waypoints = np.array(
        #     [
        #         [-1.5, 0.75, 0.05],
        #         [-1.0, 0.55, 0.4],
        #         [0.2, 0.20, 0.7],
        #         [1.3, -0.15, 0.9],
        #         [0.85, 0.85, 1.2],
        #         [-0.5, -0.05, 0.7],
        #         [-1.2, -0.2, 0.8],
        #         [-1.2, -0.2, 1.2],
        #         [-0.0, -0.7, 1.2],
        #         [0.5, -0.75, 1.2],
        #     ]
        # )
        # self._t_total = 10  # s
        # t = np.linspace(0, self._t_total, len(waypoints))
        # self._des_pos_spline = CubicSpline(t, waypoints)
        # # Sample racing trajectory and velocity at controller frequency
        # t_samples = np.linspace(0, self._t_total, int(np.ceil(self._t_total * self.freq)))
        # self.trajectory = self._des_pos_spline(t_samples)
        # self.trajectory_vel = self._des_pos_spline(t_samples, 1)

        self._tick = 0
        self._finished = False

        # Initialize evaluation recorder
        self.eval_recorder = EvalRecorder()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        i = min(self._tick, len(self.trajectory) - 1)
        if i == len(self.trajectory) - 1:  # Maximum duration reached
            self._finished = True

        des_pos = self.trajectory[i]
        des_vel = self.trajectory_vel[i]
        des_acc = np.zeros_like(self.trajectory_acc[i])
        action = np.concatenate((des_pos, des_vel, des_acc, np.zeros(4)), dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        # Record data with batch dimension (1, dim)
        idx = min(self._tick, self.trajectory.shape[0] - 1)
        position = obs["pos"].copy()
        goal = self.trajectory[idx].copy()
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")

        action = info.get("actions", np.zeros((4,)))
        self.eval_recorder.record_step(
            action=action[None, :], position=position[None, :], goal=goal[None, :], rpy=rpy[None, :]
        )

        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self.eval_recorder.plot_eval(
            save_path=f"{self.algo_name}_{self.exp_name}_deploy_plot.png", traj_plane=[0, 1]
        )
