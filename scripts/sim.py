"""Simulate drone control tasks using the plain drone environment.

Run as:

    $ python scripts/sim.py --config config.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import jax.numpy as jp
import numpy as np
from utils import load_config, load_controller, load_environment

if TYPE_CHECKING:
    from rl_diffsim.control.controller import Controller


logger = logging.getLogger(__name__)


def simulate(
    config: str = "config.toml", controller: str | None = None, n_runs: int = 1, render: bool | None = True
) -> tuple[list[float], list[float]]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `rl_diffsim/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.

    Returns:
        A tuple containing (episode_times, episode_rewards).
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parent / config)
    # Load the controller module
    control_path = Path(__file__).parents[1] / "rl_diffsim/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the drone environment
    env_cls = load_environment(Path(__file__).parents[1] / "rl_diffsim/envs" / config.exp.file)
    env = env_cls.create(**config.env)

    ep_times = []
    ep_rewards = []
    for ep in range(n_runs):  # Run n_runs episodes with the controller
        env, (obs, info) = env.reset(env, seed=config.exp.seed + ep)
        obs = {k: np.asarray(v[0]) for k, v in obs.items()}
        info = {k: v[0] for k, v in info.items()}
        controller: Controller = controller_cls(obs, info, config, sim=env.unwrapped.sim)
        i = 0
        fps = 60
        total_reward = 0.0

        while True:
            curr_time = i / config.env.freq
            action = controller.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)
            env, (obs, reward, terminated, truncated, info) = env.step(env, action[None, None, :])
            obs = {k: np.asarray(v[0]) for k, v in obs.items()}
            info = {k: v[0] for k, v in info.items()}
            reward = float(reward[0])
            terminated = bool(terminated[0])
            truncated = bool(truncated[0])
            total_reward += reward
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        print(f"Episode time (s): {curr_time:.3f}\nTotal reward: {total_reward:.3f}\n")
        controller.episode_reset()
        ep_times.append(curr_time)
        ep_rewards.append(total_reward)

    # Close the environment
    env.close()
    return ep_times, ep_rewards


if __name__ == "__main__":
    fire.Fire(simulate, serialize=lambda _: None)
