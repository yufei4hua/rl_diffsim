"""Jittable wrappers for struct.PyTreeNode-style environments."""
from __future__ import annotations

from typing import Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R


@struct.dataclass
class AngleRewardJittable(struct.PyTreeNode):
    """Jittable wrapper to penalize orientation in the reward."""
    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, rpy_coef: float = 0.08) -> "AngleRewardJittable":
        def _reset(env: "AngleRewardJittable", *, seed: int | None = None, options: dict | None = None):
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _rewards(rewards: Array, observations: dict[str, Array]) -> Array:
            """Additional angular rewards."""
            # apply rpy penalty
            rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
            rewards -= rpy_coef * rpy_norm
            return rewards

        def _step(env: "AngleRewardJittable", actions: Array):
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, rewards, terminations, truncations, infos) = env.base.step(env.base, actions)
            rewards = _rewards(rewards, obs)

            env = env.replace(base=base_env)
            return env, (obs, rewards, terminations, truncations, infos)

        return cls(
            base=base,
            rpy_coef=rpy_coef,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

    @property
    def single_observation_space(self):
        return self.base.single_observation_space

    @property
    def observation_space(self):
        return self.base.observation_space

    @property
    def action_space(self):
        return self.base.action_space

    @property
    def num_envs(self):
        return self.base.num_envs

    @property
    def unwrapped(self):
        return getattr(self.base, "unwrapped", self.base)

@struct.dataclass
class ActionPenaltyJittable(struct.PyTreeNode):
    """Jittable wrapper to apply action penalty and augment observations with last_action."""

    base: struct.PyTreeNode | AngleRewardJittable = struct.field(pytree_node=True)

    single_observation_space: spaces.Dict = struct.field(pytree_node=False)
    observation_space: spaces.Dict = struct.field(pytree_node=False)

    last_action: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode | AngleRewardJittable,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ) -> "ActionPenaltyJittable":
        num_envs = base.num_envs
        act_dim = base.action_space.shape[-1]

        spec = {k: v for k, v in base.single_observation_space.items()}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(act_dim,), dtype=np.float32)
        single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, num_envs)

        last_action = jp.zeros((num_envs, act_dim), dtype=jp.float32)

        def obs_with_last_action(env: "ActionPenaltyJittable", observations: dict[str, Array]) -> dict[str, Array]:
            observations = dict(observations)
            observations["last_action"] = env.last_action
            return observations

        def wrapped_reset(env: "ActionPenaltyJittable", *, seed: int | None = None, options: dict | None = None):
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, last_action=jp.zeros_like(env.last_action))
            obs = obs_with_last_action(env, obs)
            return env, (obs, info)

        def wrapped_step(env: "ActionPenaltyJittable", action: Array):
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            action_diff = action - env.last_action

            reward = reward - env.act_coef * action[..., -1] ** 2
            reward = reward - env.d_act_th_coef * action_diff[..., -1] ** 2
            reward = reward - env.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)

            new_last_action = action
            env = env.replace(base=base_env, last_action=new_last_action)

            obs = obs_with_last_action(env, obs)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            act_coef=act_coef,
            d_act_th_coef=d_act_th_coef,
            d_act_xy_coef=d_act_xy_coef,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            last_action=last_action,
            step=jax.jit(wrapped_step),
            reset=jax.jit(wrapped_reset),
        )

    @property
    def action_space(self):
        return self.base.action_space

    @property
    def num_envs(self):
        return self.base.num_envs

    @property
    def unwrapped(self):
        return getattr(self.base, "unwrapped", self.base)

@struct.dataclass
class FlattenJaxObservationJittable(struct.PyTreeNode):
    """Jittable wrapper to flatten dict observations into a single array."""
    base: Any = struct.field(pytree_node=True)

    single_observation_space: spaces.Box = struct.field(pytree_node=False)
    observation_space: spaces.Box = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: Any) -> "FlattenJaxObservationJittable":
        single_observation_space = flatten_space(base.single_observation_space)
        observation_space = flatten_space(base.observation_space)

        def flatten_obs(observations: dict[str, Array]) -> Array:
            return jp.concatenate([jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1)

        def wrapped_reset(env: "FlattenJaxObservationJittable", *, seed: int | None = None, options: dict | None = None):
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, info)

        def wrapped_step(env: "FlattenJaxObservationJittable", action: Array):
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            step=jax.jit(wrapped_step),
            reset=jax.jit(wrapped_reset),
        )

    @property
    def single_action_space(self):
        return self.base.single_action_space
    
    @property
    def action_space(self):
        return self.base.action_space

    @property
    def num_envs(self):
        return self.base.num_envs

    @property
    def unwrapped(self):
        return getattr(self.base, "unwrapped", self.base)

if __name__ == "__main__":
    import time  # noqa: I001
    from rl_diffsim.envs.figure_8_env_jittable import FigureEightJittableEnv
    """Test the jittable drone environment implementation."""
    # Create the jittable environment
    env = FigureEightJittableEnv.create(
        num_envs=1024,
        max_episode_time=10.0,
        physics="so_rpy_rotor_drag",
        drone_model="cf21B_500",
        freq=500,
        device="gpu",
        n_samples=10,
        trajectory_time=10.0,
        samples_dt=0.1,
        reset_rotor=True,
    )
    env = FlattenJaxObservationJittable.create(env)

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Trajectories:", env.unwrapped.trajectories.shape)
    print("Initial Obs:", obs.shape)

    def step_once(env: FigureEightJittableEnv, _) -> tuple[FigureEightJittableEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.unwrapped.data.states.pos[:, 0, :]   # (num_envs, 3)
        vel = env.unwrapped.data.states.vel[:, 0, :]   # (num_envs, 3)

        return env, (pos, vel)

    def rollout(env: FigureEightJittableEnv, num_steps: int) -> tuple[FigureEightJittableEnv, tuple[Array, Array]]:
        """Rollout for multiple steps using lax.scan."""
        env, (pos_traj, vel_traj) = jax.lax.scan(
            step_once, env, xs=None, length=num_steps
        )
        return env, (pos_traj, vel_traj)

    rollout_jit = jax.jit(rollout, static_argnames=("num_steps",))

    # Warm-up rollout
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Warm-up rollout took {end_time - start_time:.4f} seconds")

    # After jitting
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")

    print("\nPos trajectory shape:", pos_traj.shape)
    print("Vel trajectory shape:", vel_traj.shape)

    print("obs.trajectories[0, :5, :]:", env.unwrapped.trajectories[0, :5, :])