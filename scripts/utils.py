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
from scipy.spatial.transform import Rotation as R

from rl_diffsim.control.controller import Controller
from rl_diffsim.envs.drone_env import DroneEnv

matplotlib.use("Agg")  # render to raster images
from pathlib import Path

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from rl_diffsim.gym_envs.race_core import RaceCoreEnv


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
    """Load the environment module from the given path and return the RaceCoreEnv class."""
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
    env: RaceCoreEnv,
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

    def __init__(self):
        """Initialize the recorder."""
        self._record_act = []
        self._record_pos = []
        self._record_goal = []
        self._record_rpy = []

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

    def plot_eval(self, save_path: str = "eval_plot.png") -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        # Actions
        action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
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
        axes[11].text(0.1, 0.5, f"Position RMSE: {rmse_pos * 1000:.3f} mm", fontsize=14)
        axes[11].axis("off")

        plt.tight_layout()
        plt.savefig(
            Path(__file__).parents[1] / "saves" / save_path
        )  # TODO: nicer way to get root path

        return fig
