"""Sweep script for TD3 reach position with rotor velocity control."""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.td3.train_td3_reachposrv import Args, evaluate_td3, evaluate_td3_tracking, train_td3


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl-td3-rprv-sweep") as run:
        cfg = dict(run.config)

        # Rebuild env bounds from compact sweep knobs.
        pos_range = float(cfg.pop("pos_range", 0.4))
        vel_range = float(cfg.pop("vel_range", 0.5))
        ang_vel_range = float(cfg.pop("ang_vel_range", 0.5))
        z_center = 1.5
        cfg["pos_min"] = (-pos_range, -pos_range, z_center - pos_range)
        cfg["pos_max"] = (pos_range, pos_range, z_center + pos_range)
        cfg["vel_min"] = -vel_range
        cfg["vel_max"] = vel_range
        cfg["ang_vel_min"] = (-ang_vel_range,) * 3
        cfg["ang_vel_max"] = (ang_vel_range,) * 3

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
        model_path = Path(__file__).parents[2] / "saves/td3_rprv_model_sweep.ckpt"
        jax_device = args.jax_device

        reward_history, training_time = train_td3(
            args=args, model_path=model_path, jax_device=jax_device, wandb_enabled=True
        )

        # Average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(reward_history).mean() if len(reward_history) > 0 else 0.0

        # Score based on final performance
        _, _, episode_rewards, episode_lengths = evaluate_td3(
            args=args, n_eval=5, model_path=model_path, render=False, plot=False
        )
        # Score based on tracking evaluation performance
        _, rmse_pos, _, _ = evaluate_td3_tracking(
            args=args, n_eval=1, model_path=model_path, render=False, plot=False
        )

        # Score: maximize reward while minimizing RMSE.
        final_reward = float(np.mean(episode_rewards))
        rmse_pos_mm = float(rmse_pos * 1000.0)
        score = mean_rewards - rmse_pos_mm

        run.log({"score": score})
        run.log({"mean_rewards": mean_rewards})
        run.log({"training_time": training_time})
        run.log({"final_reward": final_reward})
        run.log({"rmse_pos_mm": rmse_pos_mm})
        run.log({"mean_episode_length": np.mean(episode_lengths)})

# 2: Define the search space
sweep_configuration = {
    "method": "bayes",  # "random", "bayes", "grid"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        # TD3 algorithm parameters
        "num_envs": {"values": [48, 64, 96, 128]},
        "num_steps": {"values": [4, 8, 16, 24]},
        "updates_epochs": {"values": [32, 48, 64, 96]},
        "batch_size": {"values": [256, 512, 768]},
        # "buffer_size": {"values": [131_072, 262_144, 393_216, 524_288]},
        # "learning_starts": {"values": [16384, 32768, 65536, 98304]},
        "actor_lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 3e-3},
        "critic_lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 4e-3},
        # "gamma": {"distribution": "uniform", "min": 0.98, "max": 0.99},
        # "tau": {"distribution": "uniform", "min": 0.04, "max": 0.60},
        "policy_delay": {"values": [4, 8]},
        "exploration_noise": {"distribution": "uniform", "min": 0.1, "max": 0.25},
        # "policy_noise": {"distribution": "uniform", "min": 0.1, "max": 0.2},
        # "noise_clip": {"distribution": "uniform", "min": 0.10, "max": 0.16},
        # Network architecture
        # "hidden_size": {"values": [32, 48]},
        # "num_layers": {"values": [2, 3]},
        # Envs settings
        "pos_range": {"distribution": "uniform", "min": 0.2, "max": 1.0},
        "vel_range": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "ang_vel_range": {"distribution": "uniform", "min": 0.3, "max": 2.0},
        # Wrapper settings
        # "num_last_actions": {"values": [1, 2, 4, 8]},
        "rpy_coef": {"distribution": "uniform", "min": 0.15, "max": 0.8},
        "yaw_coef": {"distribution": "uniform", "min": 0.3, "max": 1.2},
        "act_coefs_0": {"distribution": "uniform", "min": 0.05, "max": 0.2},
        # "act_coefs_1": {"distribution": "uniform", "min": 0.05, "max": 0.4},
        # "act_coefs_2": {"distribution": "uniform", "min": 0.05, "max": 0.4},
        # "act_coefs_3": {"distribution": "uniform", "min": 0.05, "max": 0.4},
        "d_act_coefs_0": {"distribution": "uniform", "min": 0.1, "max": 0.3},
        # "d_act_coefs_1": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        # "d_act_coefs_2": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        # "d_act_coefs_3": {"distribution": "uniform", "min": 0.01, "max": 0.1},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=f"{Args().wandb_project_name}-sweep", entity=Args().wandb_entity
)

wandb.agent(sweep_id, function=train, count=150)
