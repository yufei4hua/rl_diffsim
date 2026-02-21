"""Integration tests for training loops and reward improvements."""

import importlib
from pathlib import Path

import numpy as np
import pytest

TRAIN_CONFIGS = [
    {
        "id": "bptt_f8",
        "module": "rl_diffsim.bptt.train_bptt_figure8",
        "train_fn": "train_bptt",
        "eval_fn": "evaluate_bptt",
        "min_reward": 900.0,
    },
    {
        "id": "bptt_f8rv",
        "module": "rl_diffsim.bptt.train_bptt_figure8rv",
        "train_fn": "train_bptt",
        "eval_fn": "evaluate_bptt",
        "min_reward": 900.0,
    },
    {
        "id": "bptt_rp",
        "module": "rl_diffsim.bptt.train_bptt_reachpos",
        "train_fn": "train_bptt",
        "eval_fn": "evaluate_bptt",
        "min_reward": 170.0,
    },
    {
        "id": "bptt_rprv",
        "module": "rl_diffsim.bptt.train_bptt_reachposrv",
        "train_fn": "train_bptt",
        "eval_fn": "evaluate_bptt",
        "min_reward": 400.0,
    },
    {
        "id": "bptt_rt",
        "module": "rl_diffsim.bptt.train_bptt_randtraj",
        "train_fn": "train_bptt",
        "eval_fn": "evaluate_bptt",
        "min_reward": 440.0,
    },
    {
        "id": "ppo_f8",
        "module": "rl_diffsim.ppo.train_ppo_figure8",
        "train_fn": "train_ppo",
        "eval_fn": "evaluate_ppo",
        "min_reward": 900.0,
    },
]


# @pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.parametrize("cfg", TRAIN_CONFIGS, ids=lambda c: c["id"])
def test_training_and_rewards(cfg: dict, tmp_path: Path):
    """Integration test that runs a short training loop and checks that rewards improve."""
    mod = importlib.import_module(cfg["module"])
    Args = getattr(mod, "Args")
    train_fn = getattr(mod, cfg["train_fn"])
    eval_fn = getattr(mod, cfg["eval_fn"])

    args = Args.create(jax_device="cpu")
    model_path = tmp_path / f"{cfg['id']}_model_flax.ckpt"

    _ = train_fn(args=args, model_path=model_path, jax_device=args.jax_device, wandb_enabled=False)

    assert model_path.exists(), "Model file was not saved."

    fig, rmse_pos, episode_rewards, episode_lengths = eval_fn(
        args=args, n_eval=1, model_path=model_path, render=False
    )

    last_reward = float(np.mean(episode_rewards))
    assert not np.isnan(last_reward), "Final reward is NaN."
    assert last_reward > cfg["min_reward"], (
        f"Expected final eval reward > {cfg['min_reward']}, but got {last_reward:.2f}"
    )

    assert episode_lengths[-1] > 0
    if rmse_pos is not None:
        rmse_pos_val = float(rmse_pos)
        assert rmse_pos_val < 100.0, f"Expected RMSE position < 100mm, but got {rmse_pos_val:.2f}mm"
