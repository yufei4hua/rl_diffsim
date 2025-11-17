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
class JittableWrapper(struct.PyTreeNode):
    """Base class for jittable wrappers that delegates common metadata to the wrapped base.

    Subclasses are expected to provide jitted `step` and `reset` callables and may override
    any of the spaces by providing instance attributes (e.g. `single_observation_space`).
    """
    base: struct.PyTreeNode = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the single-observation Space of the wrapped base."""
        return getattr(self.base, "single_observation_space")

    @property
    def observation_space(self) -> spaces.Space:
        """Return the (possibly batched) observation Space of the wrapped base."""
        return getattr(self.base, "observation_space")

    @property
    def single_action_space(self) -> spaces.Space:
        """Return the single-action Space of the wrapped base."""
        return getattr(self.base, "single_action_space")

    @property
    def action_space(self) -> spaces.Space:
        """Return the (possibly batched) action Space of the wrapped base."""
        return getattr(self.base, "action_space")

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments from the wrapped base."""
        return getattr(self.base, "num_envs")

    @property
    def unwrapped(self) -> struct.PyTreeNode:
        """Return the unwrapped (innermost) base environment."""
        return getattr(self.base, "unwrapped", self.base)


@struct.dataclass
class NormalizeActionsJittable(JittableWrapper):
    """Jittable wrapper that exposes actions in [-1, 1] and rescales them to the simulator range.

    The wrapper stores the precomputed `scale` and `mean` as JAX arrays.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_action_space(self) -> spaces.Space:
        """Expose a normalized single-action space in [-1, 1]."""
        base_space = self.base.single_action_space
        return spaces.Box(-1.0, 1.0, shape=base_space.shape, dtype=base_space.dtype)

    @property
    def action_space(self) -> spaces.Space:
        """Batched action space matching the wrapper's num_envs."""
        return batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> "NormalizeActionsJittable":
        """Create a NormalizeActions wrapper around `base`.

        Parameters:
            base: a jittable environment that exposes `single_action_space` and `num_envs`.

        Returns:
            NormalizeActionsJittable: wrapper with jitted step/reset.
        """
        # Read simulator action bounds from base (may be numpy arrays)
        action_sim_low = jp.array(base.single_action_space.low)
        action_sim_high = jp.array(base.single_action_space.high)

        # Precompute scale and mean for rescaling from [-1,1] -> [low, high]
        scale = (action_sim_high - action_sim_low) / 2.0
        mean = (action_sim_high + action_sim_low) / 2.0

        def _reset(env: "NormalizeActionsJittable", *, seed: int | None = None, options: dict | None = None) -> tuple["NormalizeActionsJittable", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(env: "NormalizeActionsJittable", actions: Array) -> tuple["NormalizeActionsJittable", tuple[Any, ...]]:
            # actions are expected in [-1, 1]; clip and rescale to simulator range
            action = jp.clip(actions, -1.0, 1.0) * scale + mean
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


@struct.dataclass
class AngleRewardJittable(JittableWrapper):
    """Jittable wrapper to penalize orientation in the reward."""
    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, rpy_coef: float = 0.08) -> "AngleRewardJittable":
        """Create an AngleRewardJittable around `base`.

        Parameters:
            base: The jittable base environment to wrap.
            rpy_coef: Coefficient used to penalize roll/pitch/yaw magnitudes in rewards.

        Returns:
            AngleRewardJittable: A configured wrapper with jitted step/reset.
        """

        def _reset(env: "AngleRewardJittable", *, seed: int | None = None, options: dict | None = None) -> tuple["AngleRewardJittable", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _rewards(rewards: Array, observations: dict[str, Array]) -> Array:
            """Additional angular rewards."""
            # apply rpy penalty
            rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
            rewards -= rpy_coef * rpy_norm
            return rewards

        def _step(env: "AngleRewardJittable", actions: Array) -> tuple["AngleRewardJittable", tuple[Any, ...]]:
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, rewards, terminations, truncations, infos) = env.base.step(env.base, actions)
            rewards = _rewards(rewards, obs)

            env = env.replace(base=base_env)
            return env, (obs, rewards, terminations, truncations, infos)

        return cls(
            base=base,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


@struct.dataclass
class ActionPenaltyJittable(JittableWrapper):
    """Jittable wrapper to apply action penalty and augment observations with last_action."""

    base: struct.PyTreeNode | AngleRewardJittable = struct.field(pytree_node=True)
    last_action: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the base single_observation_space augmented with last_action."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        act_dim = self.base.action_space.shape[-1]
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(act_dim,), dtype=np.float32)
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode | AngleRewardJittable,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ) -> "ActionPenaltyJittable":
        """Create an ActionPenaltyJittable that augments observations with `last_action` and applies action-based penalties to rewards.

        Parameters:
            base: The jittable environment to wrap.
            act_coef, d_act_th_coef, d_act_xy_coef: Coefficients controlling the penalty terms.

        Returns:
            ActionPenaltyJittable: Configured wrapper instance.
        """
        num_envs = base.num_envs
        act_dim = base.action_space.shape[-1]
        # last_action is part of the observation dict (computed in property)
        last_action = jp.zeros((num_envs, act_dim), dtype=jp.float32)
        def _reset(env: "ActionPenaltyJittable", *, seed: int | None = None, options: dict | None = None) -> tuple["ActionPenaltyJittable", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, last_action=jp.zeros_like(env.last_action))
            obs["last_action"] = env.last_action
            return env, (obs, info)

        def _step(env: "ActionPenaltyJittable", action: Array) -> tuple["ActionPenaltyJittable", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            # penalty on actions
            action_diff = action - env.last_action
            # energy
            reward = reward - act_coef * action[..., -1] ** 2
            # smoothness
            reward = reward - d_act_th_coef * action_diff[..., -1] ** 2
            reward = reward - d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)

            new_last_action = action
            env = env.replace(base=base_env, last_action=new_last_action)

            obs["last_action"] = env.last_action
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            last_action=last_action,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

@struct.dataclass
class FlattenJaxObservationJittable(JittableWrapper):
    """Jittable wrapper to flatten dict observations into a single array."""
    base: Any = struct.field(pytree_node=True)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Flattened single-observation space derived from the base."""
        return flatten_space(self.base.single_observation_space)

    @property
    def observation_space(self) -> spaces.Space:
        """Flattened (batched) observation space derived from the base."""
        return flatten_space(self.base.observation_space)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: Any) -> "FlattenJaxObservationJittable":
        """Create a FlattenJaxObservationJittable that concatenates dict observations.

        Parameters:
            base: The jittable environment to wrap.

        Returns:
            FlattenJaxObservationJittable: Configured wrapper instance.
        """

        def flatten_obs(observations: dict[str, Array]) -> Array:
            return jp.concatenate([jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1)
        def _reset(env: "FlattenJaxObservationJittable", *, seed: int | None = None, options: dict | None = None) -> tuple["FlattenJaxObservationJittable", tuple[Array, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, info)
        def _step(env: "FlattenJaxObservationJittable", action: Array) -> tuple["FlattenJaxObservationJittable", tuple[Array, Any]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

if __name__ == "__main__":
    import time  # noqa: I001
    from rl_diffsim.envs.figure_8_env_jittable import FigureEightJittableEnv
    """Test the jittable wrappers implementation."""
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
    # wrap the environment
    env = NormalizeActionsJittable.create(env)
    env = AngleRewardJittable.create(env)
    env = ActionPenaltyJittable.create(env)
    env = FlattenJaxObservationJittable.create(env)

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Trajectories:", env.unwrapped.trajectories.shape)
    print("Initial Obs:", obs.shape)

    def step_once(env: FigureEightJittableEnv, _) -> tuple[FigureEightJittableEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32) # fixed action
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