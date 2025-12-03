"""Sweep it!"""

from pathlib import Path

import numpy as np
import torch

import wandb
from rl_diffsim.ppo.train_ppo_torch import Args, evaluate_ppo, train_ppo  # noqa: F401


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl_diffsim-PPO-sweep") as run:
        args = Args.create(**dict(run.config))
        model_path = Path(__file__).parent / "saves/ppo_model_flax_sweep.ckpt"
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        jax_device = args.jax_device
        sum_rewards_hist, training_time = train_ppo(
            args=args,
            model_path=model_path,
            device=device,
            jax_device=jax_device,
            wandb_enabled=True,
        )
        # average over rewards curve: aiming at faster convergence
        mean_rewards = np.asarray(sum_rewards_hist).mean()
        # score based on final performance
        _, rmse_pos, episode_rewards, _ = evaluate_ppo(
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
        "num_envs": {"values": [256, 512, 1024, 2048]},
        "num_steps": {"values": [4, 8, 16, 32]},
        "num_minibatches": {"values": [4, 8, 16]},
        "actor_lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-3},
        "critic_lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-3},
        "gamma": {"min": 0.85, "max": 0.95},
        "gae_lambda": {"min": 0.9, "max": 0.99},
        "clip_coef": {"min": 0.1, "max": 0.8},
        "ent_coef": {"min": 0.0, "max": 1e-2},
        "vf_coef": {"min": 0.5, "max": 1.0},
        "max_grad_norm": {"min": 1.5, "max": 4.0},
        "hidden_size": {"values": [8, 16, 32, 64]},
    },
}


# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="rl_diffsim-PPO-sweep", entity="fresssack"
)

wandb.agent(sweep_id, function=train, count=50)
