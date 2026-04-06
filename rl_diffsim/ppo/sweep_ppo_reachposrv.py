"""Sweep script for PPO reach position with rotor velocity control."""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.ppo.train_ppo_reachposrv import Args, evaluate_ppo, train_ppo


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl-ppo-rprv-sweep") as run:
        cfg = dict(run.config)
        # Rebuild coefficient tuples from flattened sweep parameters if present
        act_coefs = tuple(
            cfg.pop(f"act_coefs_{i}") if f"act_coefs_{i}" in cfg else Args.act_coefs[i]
            for i in range(len(Args.act_coefs))
        )
        d_act_coefs = tuple(
            cfg.pop(f"d_act_coefs_{i}") if f"d_act_coefs_{i}" in cfg else Args.d_act_coefs[i]
            for i in range(len(Args.d_act_coefs))
        )
        cfg["act_coefs"] = act_coefs
        cfg["d_act_coefs"] = d_act_coefs

        args = Args.create(**cfg)
        model_path = Path(__file__).parents[2] / "saves/ppo_rprv_model_sweep.ckpt"
        jax_device = args.jax_device

        reward_history, training_time = train_ppo(
            args=args, model_path=model_path, jax_device=jax_device, wandb_enabled=True
        )

        # Average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(reward_history).mean() if len(reward_history) > 0 else 0.0

        # Score based on final performance
        _, rmse_pos, episode_rewards, episode_lengths = evaluate_ppo(
            args=args, n_eval=5, model_path=model_path, render=False, plot=False
        )

        # Score: higher reward and lower position error is better
        final_reward = np.mean(episode_rewards)
        score = mean_rewards - training_time * 5.0

        run.log({"score": score})
        run.log({"mean_rewards": mean_rewards})
        run.log({"training_time": training_time})
        run.log({"final_reward": final_reward})
        run.log({"rmse_pos_mm": rmse_pos * 1000})
        run.log({"mean_episode_length": np.mean(episode_lengths)})


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",  # "random", "bayes", "grid"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        # PPO algorithm parameters
        "num_envs": {"values": [512, 1024, 2048]},
        "num_steps": {"values": [16, 32, 48, 64]},
        "num_minibatches": {"values": [8, 16, 32]},
        "update_epochs": {"values": [8, 12]},
        "actor_lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
        "critic_lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 1e-2},
        "gamma": {"distribution": "uniform", "min": 0.96, "max": 0.995},
        "gae_lambda": {"distribution": "uniform", "min": 0.9, "max": 0.99},
        "clip_coef": {"distribution": "uniform", "min": 0.2, "max": 0.5},
        "ent_coef": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.02},
        "vf_coef": {"distribution": "uniform", "min": 0.5, "max": 0.9},
        # Network architecture
        # "hidden_size": {"values": [32, 64, 128]},
        # Wrapper settings (reward shaping)
        "rpy_coef": {"distribution": "uniform", "min": 0.15, "max": 0.30},
        "act_coefs_0": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        "d_act_coefs_0": {"distribution": "uniform", "min": 0.05, "max": 0.15},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=f"{Args().wandb_project_name}-sweep", entity=Args().wandb_entity
)

wandb.agent(sweep_id, function=train, count=100)
