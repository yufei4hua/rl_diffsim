"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

from crazyflow.utils import enable_cache

import rl_diffsim.gym_envs  # noqa: F401, register environments with gymnasium

enable_cache()  # Enable persistent caching of jax functions
