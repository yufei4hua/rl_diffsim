"""Sweep it!"""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.shac.train_shac import Args, evaluate_shac, train_shac


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl_diffsim-SHAC-sweep") as run:
        args = Args.create(**dict(run.config))
        model_path = Path(__file__).parents[2] / "saves/shac_model_flax_sweep.ckpt"
        jax_device = args.jax_device
        sum_rewards_hist, training_time = train_shac(
            args=args, model_path=model_path, jax_device=jax_device, wandb_enabled=True
        )
        # average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(sum_rewards_hist).mean()
        # score based on final performance
        _, rmse_pos, episode_rewards, _ = evaluate_shac(
            args=args, n_eval=1, model_path=model_path, render=False
        )
        score = mean_rewards - 2 * training_time
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
        "num_envs": {"values": [16, 32, 64]},
        "num_steps": {"values": [8, 16, 32]},
        "num_minibatches": {"values": [1, 2, 4]},
        "actor_lr": {
            "distribution": "log_uniform_values",
            "min": 2e-2,
            "max": 1e-1,
        },
        "critic_lr": {
            "distribution": "log_uniform_values",
            "min": 1e-3,
            "max": 5e-3,
        },
        "gamma": {"min": 0.9, "max": 0.999},
        "gae_lambda": {"min": 0.9, "max": 0.99},
        "update_epochs": {"values": [10, 15]},
        "clip_coef": {"min": 0.2, "max": 0.6},
        "hidden_size": {"values": [8, 16]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="rl_diffsim-SHAC-sweep", entity="lsy-tum"
)

wandb.agent(sweep_id, function=train, count=100)
