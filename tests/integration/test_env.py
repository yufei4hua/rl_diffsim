from typing import Generator

import flax.struct as struct
import jax
import jax.numpy as jp
import pytest
from crazyflow.sim.physics import Physics

from rl_diffsim.envs.drone_env_jittable import DroneJittableEnv
from rl_diffsim.envs.figure_8_env_jittable import FigureEightJittableEnv
from rl_diffsim.envs.rand_traj_env_jittable import RandTrajJittableEnv
from rl_diffsim.envs.reach_pos_env_jittable import ReachPosJittableEnv

available_envs = [
    DroneJittableEnv,
    FigureEightJittableEnv,
    ReachPosJittableEnv,
    RandTrajJittableEnv,
]


@pytest.fixture(params=available_envs)
def env(request: pytest.FixtureRequest) -> Generator[struct.PyTreeNode, None, None]:
    """Small env fixture for structural tests."""
    env_class = request.param
    env = env_class.create(
        num_envs=2,
        max_episode_time=2.0,
        physics=Physics.so_rpy_rotor_drag,
        drone_model="cf21B_500",
        freq=500,
        device="cpu",
    )
    yield env


@pytest.mark.parametrize("EnvClass", available_envs)
def test_pytree_metadata_flags(EnvClass: struct.PyTreeNode):
    """Check that fields have the expected pytree_node=True/False flags."""
    fields = EnvClass.__dataclass_fields__  # type: ignore[attr-defined]

    # Non-pytree (static / metadata) fields
    non_pytree_fields = [
        "sim",
        "num_envs",
        "max_episode_time",
        "physics",
        "drone_model",
        "freq",
        "device",
        "single_action_space",
        "action_space",
        "single_observation_space",
        "observation_space",
        "n_substeps",
        "reset",
        "step",
    ]

    for name in non_pytree_fields:
        assert name in fields, f"Field {name} not found on DroneJittableEnv"
        meta = fields[name].metadata
        # flax.struct.field stores this as metadata["pytree_node"]
        assert meta.get("pytree_node", True) is False, f"{name} should have pytree_node=False"

    # PyTree (dynamic state) fields
    pytree_fields = ["data", "steps", "_marked_for_reset"]

    for name in pytree_fields:
        assert name in fields, f"Field {name} not found on DroneJittableEnv"
        meta = fields[name].metadata
        assert meta.get("pytree_node", True) is True, f"{name} should have pytree_node=True"


def test_variable_data_exist(env: DroneJittableEnv):
    """Sanity check that dynamic variable data actually exist on instances."""
    assert hasattr(env, "data")
    assert hasattr(env, "steps")
    assert hasattr(env, "_marked_for_reset")


def test_callables_exist(env: DroneJittableEnv):
    """Ensure reset and step functions exist and are callable."""
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert callable(env.reset)
    assert callable(env.step)


def test_rollout_scan_and_jit_shape(env: struct.PyTreeNode) -> None:
    """Replicate the example script: scan over step() and check trajectory shapes."""
    # Start from a deterministic reset state
    env, _ = env.reset(env, seed=42)

    def step_once(
        env: struct.PyTreeNode, _
    ) -> tuple[struct.PyTreeNode, tuple[jax.Array, jax.Array]]:
        """Single env step body for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(
        env: struct.PyTreeNode, num_steps: int
    ) -> tuple[struct.PyTreeNode, tuple[jax.Array, jax.Array]]:
        """Rollout for multiple steps using lax.scan."""
        env_out, (pos_traj, vel_traj) = jax.lax.scan(step_once, env, xs=None, length=num_steps)
        return env_out, (pos_traj, vel_traj)

    rollout_jit = jax.jit(rollout, static_argnames=("num_steps",))

    num_steps = 8
    env_out, (pos_traj, vel_traj) = rollout_jit(env, num_steps)

    num_envs = env.num_envs

    # Trajectories should have shape (T, num_envs, 3)
    assert pos_traj.shape == (num_steps, num_envs, 3)
    assert vel_traj.shape == (num_steps, num_envs, 3)

    # Sanity: env_out remains a struct.PyTreeNode and step counter advanced
    assert isinstance(env_out, struct.PyTreeNode)
    assert jp.all(env_out.steps >= 0)
