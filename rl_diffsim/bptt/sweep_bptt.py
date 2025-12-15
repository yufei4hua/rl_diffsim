"""Sweep it!"""

from pathlib import Path

import numpy as np

import wandb
from rl_diffsim.bptt.train_bptt_figure8 import Args, evaluate_bptt, train_bptt


# 1: Define objective/training function
def train():
    """Train."""
    with wandb.init(project="rl_diffsim-SHAC-sweep") as run:
        args = Args.create(**dict(run.config))
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
        score = mean_rewards - rmse_pos - training_time * 10
        run.log({"score": score})
        run.log({"mean_rewards": mean_rewards})
        run.log({"training_time": training_time})
        run.log({"final_reward": np.mean(episode_rewards)})
        run.log({"rmse_pos": rmse_pos})


# 2: Define the search space
sweep_configuration = {
    "method": "random",  # "random", "bayes", "grid"
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "num_envs": {"distribution": "int_uniform", "min": 4, "max": 16},
        "num_steps": {"distribution": "int_uniform", "min": 32, "max": 96},
        "actor_lr": {"distribution": "log_uniform_values", "min": 5e-4, "max": 4e-1},
        # "gamma": {"min": 0.90, "max": 1.0},
        "hidden_size": {"distribution": "int_uniform", "min": 4, "max": 16},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="rl_diffsim-BPTT-sweep-deploy", entity="fresssack"
)

wandb.agent(sweep_id, function=train, count=200)
