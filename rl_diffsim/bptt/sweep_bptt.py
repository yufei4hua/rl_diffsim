"""Sweep it!"""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.bptt.train_bptt_figure8 import Args, evaluate_bptt, train_bptt


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl_diffsim-SHAC-sweep") as run:
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
        model_path = Path(__file__).parents[2] / "saves/bptt_model_flax_sweep.ckpt"
        jax_device = args.jax_device
        sum_rewards_hist, training_time = train_bptt(
            args=args, model_path=model_path, jax_device=jax_device, wandb_enabled=True
        )
        # average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(sum_rewards_hist).mean()
        # score based on final performance
        _, rmse_pos, episode_rewards, _ = evaluate_bptt(
            args=args, n_eval=1, model_path=model_path, render=False
        )
        score = 0.0
        score += mean_rewards
        score += -rmse_pos * 1000
        # score += -10 * training_time
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
        # "num_envs": {"distribution": "int_uniform", "min": 4, "max": 32},
        "num_steps": {"distribution": "int_uniform", "min": 32, "max": 48},
        "actor_lr": {"distribution": "log_uniform_values", "min": 3e-2, "max": 1e-1},
        # "gamma": {"min": 0.90, "max": 1.0},
        # "hidden_size": {"distribution": "int_uniform", "min": 8, "max": 64},
        # "hidden_size": {"values": [8, 16]},
        # "rpy_coef": {"distribution": "uniform", "min": 0.01, "max": 1.0},
        "act_coefs_0": {"distribution": "uniform", "min": 0.01, "max": 1.0},
        # "act_coefs_1": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        # "act_coefs_2": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        "act_coefs_3": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        "d_act_coefs_0": {"distribution": "uniform", "min": 0.01, "max": 1.0},
        # "d_act_coefs_1": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        # "d_act_coefs_2": {"distribution": "uniform", "min": 0.1, "max": 1.0},
        "d_act_coefs_3": {"distribution": "uniform", "min": 0.1, "max": 1.0},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project=Args().wandb_project_name, entity="fresssack"
)

wandb.agent(sweep_id, function=train, count=50)
