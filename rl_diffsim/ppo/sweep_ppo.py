"""Sweep it!"""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.ppo.train_ppo_figure8 import Args, evaluate_ppo, train_ppo


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl_diffsim-PPO-sweep") as run:
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
        # gate_size = tuple(
        #     cfg.pop(f"gate_size_{i}") if f"gate_size_{i}" in cfg else Args.gate_size[i]
        #     for i in range(len(Args.gate_size))
        # )
        # gate_vel_coef = tuple(
        #     cfg.pop(f"gate_vel_coef_{i}") if f"gate_vel_coef_{i}" in cfg else Args.gate_vel_coef[i]
        #     for i in range(len(Args.gate_vel_coef))
        # )
        # contact_coef = tuple(
        #     cfg.pop(f"contact_coef_{i}") if f"contact_coef_{i}" in cfg else Args.contact_coef[i]
        #     for i in range(len(Args.contact_coef))
        # )
        cfg["act_coefs"] = act_coefs
        cfg["d_act_coefs"] = d_act_coefs
        # cfg["gate_size"] = gate_size
        # cfg["gate_vel_coef"] = gate_vel_coef
        # cfg["contact_coef"] = contact_coef
        args = Args.create(**cfg)
        model_path = Path(__file__).parents[2] / "saves/ppo_model_flax_sweep.ckpt"
        jax_device = args.jax_device
        sum_rewards_hist, training_time = train_ppo(
            args=args, model_path=model_path, jax_device=jax_device, wandb_enabled=True
        )
        # average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(sum_rewards_hist).mean()
        # # score based on final performance
        # _, success_count, episode_rewards, avg_lap_time = evaluate_ppo(
        #     args=args, n_eval=20, model_path=model_path, render=False, plot=False
        # )
        # score = success_count - avg_lap_time * 5.0
        # run.log({"score": score})
        # run.log({"mean_rewards": mean_rewards})
        # run.log({"final_reward": np.mean(episode_rewards)})
        # run.log({"success_count": success_count})
        # run.log({"avg_lap_time": avg_lap_time})

        _, rmse_pos, episode_rewards, _ = evaluate_ppo(
            args=args, n_eval=1, model_path=model_path, render=False
        )
        score = mean_rewards - 200 * rmse_pos
        run.log({"score": score})
        run.log({"mean_rewards": mean_rewards})
        run.log({"training_time": training_time})
        run.log({"final_reward": np.mean(episode_rewards)})
        run.log({"rmse_pos": rmse_pos})


# 2: Define the search space
sweep_configuration = {
    "method": "bayes",  # "random", "bayes", "grid"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        # "num_envs": {"values": [1024]},
        # "num_steps": {"values": [32, 48, 64]},
        # "num_minibatches": {"values": [32, 48, 64]},
        "actor_lr": {"distribution": "log_uniform_values", "min": 4e-4, "max": 3e-3},
        # "critic_lr": {"distribution": "log_uniform_values", "min": 1e-3, "max": 1e-2},
        # "gamma": {"min": 0.85, "max": 0.99},
        # "gae_lambda": {"min": 0.85, "max": 0.99},
        "hidden_size": {"values": [16, 24, 32]},
        # "seed": {"values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
        # "num_envs": {"values": [1024, 2048]},
        # "num_steps": {"values": [48, 64, 96, 128]},
        # # "num_minibatches": {"values": [4, 8, 16]},
        # # "actor_lr": {"distribution": "log_uniform_values", "min": 5e-4, "max": 2e-3},
        # # "critic_lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 5e-3},
        # "gamma": {"min": 0.85, "max": 0.99},
        # "gae_lambda": {"min": 0.8, "max": 0.95},
        # # "update_epochs": {"values": [7, 10, 15]},
        # # "clip_coef": {"min": 0.1, "max": 0.5},
        # # "ent_coef": {"min": 0.0, "max": 1e-2},
        # # "vf_coef": {"min": 0.5, "max": 1.0},
        # "hidden_size": {"values": [48, 64, 96]},
        # wrapper settings (race-specific)
        # "min_vel": {"distribution": "uniform", "min": 0.3, "max": 0.6},
        # "max_vel": {"distribution": "uniform", "min": 1.0, "max": 2.5},
        # "cont_gate_safe_dist": {"distribution": "uniform", "min": 0.05, "max": 0.2},
        # "cont_obst_safe_dist": {"distribution": "uniform", "min": 0.05, "max": 0.2},
        # "gate_size_1": {"distribution": "uniform", "min": 0.2, "max": 0.5},
        # "gate_vel_coef_0": {"distribution": "uniform", "min": 1.5, "max": 3.0},
        # "gate_vel_coef_1": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        # "contact_coef_1": {"distribution": "uniform", "min": 40.0, "max": 60.0},
        "act_coefs_0": {"distribution": "uniform", "min": 0.01, "max": 0.2},
        # "act_coefs_1": {"distribution": "uniform", "min": 0.05, "max": 0.25},
        # "act_coefs_2": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        "act_coefs_3": {"distribution": "uniform", "min": 0.01, "max": 0.2},
        "d_act_coefs_0": {"distribution": "uniform", "min": 0.1, "max": 1.5},
        # "d_act_coefs_1": {"distribution": "uniform", "min": 0.5, "max": 1.5},
        # "d_act_coefs_2": {"distribution": "uniform", "min": 0.0, "max": 0.2},
        "d_act_coefs_3": {"distribution": "uniform", "min": 0.1, "max": 0.9},
    },
}


# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=f"{Args().wandb_project_name}-sweep", entity="fresssack"
)

wandb.agent(sweep_id, function=train, count=50)
