"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

from pathlib import Path

from crazyflow.utils import enable_cache

enable_cache(Path("./jax_cache"))  # Enable persistent caching of jax functions
