"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fire
import rclpy
from utils import load_config, load_controller

from rl_diffsim.envs.real_drone_env import RealDroneEnv

logger = logging.getLogger(__name__)


def main(config: str = "config.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    rclpy.init()
    config = load_config(Path(__file__).parents[1] / "scripts" / config)
    env: RealDroneEnv = RealDroneEnv(
        drones=config.deploy.drones,
        freq=config.env.freq,
        control_mode=config.env.control_mode,
    )
    try:
        obs, info = env.reset()
        env._move_to_start()  # Try to move to start position
        next_obs = env.obs()  # Set next_obs to avoid errors when the loop never enters
        print("Starting control loop...")

        control_path = Path(__file__).parents[1] / "rl_diffsim/control"
        controller_path = control_path / (controller or config.controller.file)
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config, None)
        start_time = time.perf_counter()
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}
            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = True
        controller.episode_callback()
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
    finally:
        env.close()



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("rl_diffsim").setLevel(logging.INFO)
    fire.Fire(main)
