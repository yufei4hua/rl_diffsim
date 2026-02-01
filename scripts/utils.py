"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import mujoco
import numpy as np
import toml
from ml_collections import ConfigDict

os.environ["SCIPY_ARRAY_API"] = "1"
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R

from rl_diffsim.control.controller import Controller
from rl_diffsim.envs.drone_env import DroneEnv

matplotlib.use("Agg")  # render to raster images
from pathlib import Path

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from rl_diffsim.envs.drone_race_env import DroneRaceEnv


logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_environment(path: Path) -> "DroneEnv":
    """Load the environment module from the given path and return the DroneRaceEnv class."""
    assert path.exists(), f"Environment file not found: {path}"
    assert path.is_file(), f"Environment path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("environment", path)
    environment_module = importlib.util.module_from_spec(spec)
    sys.modules["environment"] = environment_module
    spec.loader.exec_module(environment_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid environment classes.

        Args:
            mod: Any attribute of the environment module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, DroneEnv)
        return subcls and mod.__module__ == environment_module.__name__

    environments = inspect.getmembers(environment_module, filter)
    environments = [c for _, c in environments if issubclass(c, DroneEnv)]
    assert len(environments) > 0, f"No environment found in {path}. Have you subclassed DroneEnv?"
    assert len(environments) == 1, f"Multiple environments found in {path}. Only one is allowed."
    environment_module.DroneEnv = environments[0]
    assert issubclass(environment_module.DroneEnv, DroneEnv)
    return environment_module.DroneEnv


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def draw_line(
    env: DroneRaceEnv,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        env: The drone racing environment.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    sim = env.unwrapped.sim
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))


@dataclass
class EvalRecorder:
    """Class to record evaluation data and plot them."""

    _record_act: list[NDArray]
    _record_pos: list[NDArray]
    _record_goal: list[NDArray]
    _record_rpy: list[NDArray]

    def __init__(self, control: str = "attitude"):
        """Initialize the recorder."""
        self._record_act = []
        self._record_pos = []
        self._record_goal = []
        self._record_rpy = []
        self.control = control

    def record_step(self, action: NDArray, position: NDArray, goal: NDArray, rpy: NDArray):
        """Record a single step's data.

        Args:
            action: The action taken at this step.
            position: The drone's position at this step.
            goal: The current goal position at this step.
            rpy: The roll, pitch, yaw angles at this step.
        """
        self._record_act.append(action)
        self._record_pos.append(position)
        self._record_goal.append(goal)
        self._record_rpy.append(rpy)

    def plot_eval(self, save_path: str = "eval_plot.png", traj_plane: list = [0, 2]) -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        # Actions
        if self.control == "attitude":
            action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        elif self.control == "force_torque":
            action_labels = ["Thrust", "Tx", "Ty", "Tz"]
        elif self.control == "rotor_vel":
            action_labels = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]
        else:
            raise ValueError(f"Unsupported control type: {self.control}")
        for i in range(4):
            axes[i].plot(actions[:, 0, i])
            axes[i].set_title(f"{action_labels[i]} Command")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Action Value")
            axes[i].grid(True)

        # Position plots and goals
        for i, label in enumerate(["X Position", "Y Position", "Z Position"]):
            axes[4 + i].plot(pos[:, 0, i])
            axes[4 + i].plot(goal[:, 0, i], linestyle="--")
            axes[4 + i].set_title(label)
            axes[4 + i].set_xlabel("Time Step")
            axes[4 + i].set_ylabel("Position (m)")
            axes[4 + i].grid(True)
            axes[4 + i].legend(["Position", "Goal"])

        # Position error
        pos_err = np.linalg.norm(pos[:, 0] - goal[:, 0], axis=1)
        axes[7].plot(pos_err)
        axes[7].set_title("Position Error")
        axes[7].set_xlabel("Time Step")
        axes[7].set_ylabel("Error (m)")
        axes[7].grid(True)

        # Angles
        rpy_labels = ["Roll", "Pitch", "Yaw"]
        for i in range(3):
            axes[8 + i].plot(rpy[:, 0, i])
            axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
            axes[8 + i].set_xlabel("Time Step")
            axes[8 + i].set_ylabel("Angle (rad)")
            axes[8 + i].grid(True)

        # compute RMSE for position
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        # trajectory plot
        axes[11].plot(pos[:, 0, traj_plane[0]], pos[:, 0, traj_plane[1]], label="Actual")
        axes[11].plot(
            goal[:, 0, traj_plane[0]],
            goal[:, 0, traj_plane[1]],
            linestyle="--",
            linewidth=0.5,
            label="Goal",
        )
        axes[11].set_title(f"Trajectory Plane (RMSE: {rmse_pos * 1000:.3f} mm)")
        axes[11].set_xlabel(f"{['X', 'Y', 'Z'][traj_plane[0]]} Position (m)")
        axes[11].set_ylabel(f"{['X', 'Y', 'Z'][traj_plane[1]]} Position (m)")
        axes[11].grid(True)
        axes[11].legend()
        axes[11].axis("equal")

        plt.tight_layout()
        plt.savefig(
            Path(__file__).parents[1] / "saves" / save_path
        )  # TODO: nicer way to get root path

        return fig


class RaceRecorder:
    """Class to record race deployment data and plot them."""

    _record_act: list[NDArray]
    _record_pos: list[NDArray]
    _record_vel: list[NDArray]
    _record_rpy: list[NDArray]
    action_scale: NDArray
    action_mean: NDArray

    def __init__(self, control: str = "attitude", action_scale: NDArray = np.ones(4), action_mean: NDArray = np.zeros(4)):
        """Initialize the recorder."""
        self._record_act = []
        self._record_pos = []
        self._record_vel = []
        self._record_rpy = []
        self.control = control
        self.action_scale = action_scale
        self.action_mean = action_mean
        self.gates = np.zeros((4, 3))
        self.obstacles = np.zeros((4, 3))

    def record_step(self, action: NDArray, position: NDArray, velocity: NDArray, rpy: NDArray):
        """Record a single step's data.

        Args:
            action: The action taken at this step.
            position: The drone's position at this step.
            velocity: The drone's velocity at this step.
            rpy: The roll, pitch, yaw angles at this step.
        """
        self._record_act.append(action)
        self._record_pos.append(position)
        self._record_vel.append(velocity)
        self._record_rpy.append(rpy)

    def update_track(self, gates: NDArray, obstacles: NDArray):
        """Update the track gates and obstacles positions.

        Args:
            gates: The positions of the gates.
            obstacles: The positions of the obstacles.
        """
        self.gates = gates
        self.obstacles = obstacles

    def plot_eval(self, save_path: str = "race_eval_plot.png") -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        vel = np.array(self._record_vel)
        rpy = np.array(self._record_rpy)
        lap_time = pos.shape[0] * 0.02 

        fig = plt.figure(figsize=(18, 12), constrained_layout=True)
        gs = GridSpec(nrows=3, ncols=4, figure=fig, hspace=0.05, wspace=0.05)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1:3, 0:2]),
            fig.add_subplot(gs[1:3, 2:4]),
        ]

        # Actions
        if self.control == "attitude":
            action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        else:
            raise ValueError(f"Unsupported control type: {self.control}")
        actions = actions * self.action_scale + self.action_mean  # rescale to sim action range
        for i in range(4):
            if i < 3:
                axes[i].plot(rpy[:, 0, i], label="Actual")
            axes[i].plot(actions[:, 0, i], linestyle="--", color="orange", label="Command")
            axes[i].set_title(f"{action_labels[i]}")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Angle (rad)")
            axes[i].legend()
            axes[i].grid(True)

        # Race trajectory plot
        gates = self.gates
        obstacles = self.obstacles

        # Calculate velocity norm for color mapping
        vel_norm = np.linalg.norm(vel[:, 0, :], axis=-1)  # shape: (T)
        # XY Plane trajectory with velocity color mapping
        # Create line segments for LineCollection
        points_xy = np.array([pos[:, 0, 0], pos[:, 0, 1]]).T.reshape(-1, 1, 2)
        segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
        lc_xy = LineCollection(segments_xy, cmap="turbo", linewidth=2)
        lc_xy.set_array(vel_norm[:-1])  # Use velocity at start of each segment
        line_xy = axes[4].add_collection(lc_xy)
        axes[4].scatter(
            gates[:, 0], gates[:, 1], c="green", s=80, marker="o", label="Gates", zorder=5
        )
        axes[4].scatter(
            obstacles[:, 0], obstacles[:, 1], c="red", s=80, marker="x", label="Obstacles", zorder=5
        )
        axes[4].set_title(f"Race Trajectory XY Plane (Lap Time: {lap_time:.2f}s)")
        axes[4].set_xlabel("X Position (m)")
        axes[4].set_ylabel("Y Position (m)")
        axes[4].grid(True)
        axes[4].axis("equal")
        axes[4].autoscale()
        # Add colorbar for XY plane
        cbar_xy = fig.colorbar(line_xy, ax=axes[4], fraction=0.046, pad=0.04)
        cbar_xy.set_label("Velocity (m/s)", rotation=270, labelpad=15)

        # XZ Plane trajectory with velocity color mapping
        # Create line segments for LineCollection
        points_xz = np.array([pos[:, 0, 0], pos[:, 0, 2]]).T.reshape(-1, 1, 2)
        segments_xz = np.concatenate([points_xz[:-1], points_xz[1:]], axis=1)
        lc_xz = LineCollection(segments_xz, cmap="turbo", linewidth=2)
        lc_xz.set_array(vel_norm[:-1])  # Use velocity at start of each segment
        line_xz = axes[5].add_collection(lc_xz)
        axes[5].scatter(
            gates[:, 0], gates[:, 2], c="green", s=80, marker="o", label="Gates", zorder=5
        )
        axes[5].set_title("Race Trajectory XZ Plane")
        axes[5].set_xlabel("X Position (m)")
        axes[5].set_ylabel("Z Position (m)")
        axes[5].grid(True)
        axes[5].axis("equal")
        axes[5].autoscale()
        # Add colorbar for XZ plane
        cbar_xz = fig.colorbar(line_xz, ax=axes[5], fraction=0.046, pad=0.04)
        cbar_xz.set_label("Velocity (m/s)", rotation=270, labelpad=15)

        fig.savefig(Path(__file__).parents[1] / "saves" / save_path, bbox_inches="tight")

        return fig