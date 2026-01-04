"""wrappers for struct.PyTreeNode-style environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import matplotlib
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")  # render to raster images
import matplotlib.pyplot as plt


# region Base
@struct.dataclass
class Wrapper(struct.PyTreeNode):
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

    @property
    def steps(self) -> Array:
        """Get the current step count for each environment."""
        return getattr(self.base, "steps")

    def render(self, **kwargs: dict) -> None:
        """Returns the render mode from the base vector environment."""
        return self.base.render(**kwargs)

    def close(self, **kwargs: Any) -> None:
        """Close all environments."""
        return self.base.close(**kwargs)


# region ActionTransform
@struct.dataclass
class ActionTransform(Wrapper):
    """Wrapper that interprets policy orientation output as normalized rot_vec."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_action_space(self) -> spaces.Space:
        """Expose normalized action space in [-1, 1]."""
        shape = self.base.single_action_space.shape
        dtype = getattr(self.base.single_action_space, "dtype", np.float32)
        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=dtype)

    @property
    def action_space(self) -> spaces.Space:
        """Batched normalized action space."""
        return batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode, action_scale: float = 1.0) -> "ActionTransform":
        """Create wrapper.

        Parameters:
            base: Underlying jittable env expecting [rpy(3), thrust].
            action_scale: Rotvec per-component bound in radians.
        """
        a_scale = jp.array(action_scale, dtype=jp.float32)

        base_low = jp.array(base.single_action_space.low, dtype=jp.float32)
        base_high = jp.array(base.single_action_space.high, dtype=jp.float32)
        scale = (base_high - base_low) / 2.0
        mean = (base_high + base_low) / 2.0
        scale = scale.at[:3].set(a_scale)
        mean = mean.at[:3].set(0.0)

        def _step(
            env: "ActionTransform", actions: Array
        ) -> tuple["ActionTransform", tuple[Any, ...]]:
            # Normalize -> physical rotvec + thrust.
            actions = actions * scale + mean

            # # delta rotvec (recommended for your use-case)
            # quat = env.unwrapped.data.states.quat[:, 0, :]  # (num_envs, 4)
            # rpy = (R.from_quat(quat) * R.from_rotvec(actions[..., :3])).as_euler("xyz")

            # abs rotvec (for ablation/debug)
            rpy = R.from_rotvec(actions[..., :3]).as_euler("xyz")

            act = jp.concatenate([rpy, actions[..., 3:]], axis=-1)
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, act)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        def _reset(
            env: "ActionTransform", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["ActionTransform", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region NormAct
@struct.dataclass
class NormalizeActions(Wrapper):
    """Wrapper that exposes actions in [-1, 1] and rescales them to the simulator range.

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
    def create(cls, base: struct.PyTreeNode) -> "NormalizeActions":
        """Create a NormalizeActions wrapper around `base`.

        Parameters:
            base: The jittable base environment to wrap.

        Returns:
            NormalizeActions: wrapper with jitted step/reset.
        """
        # Read simulator action bounds from base (may be numpy arrays)
        action_sim_low = jp.array(base.single_action_space.low)
        action_sim_high = jp.array(base.single_action_space.high)

        # Precompute scale and mean for rescaling from [-1,1] -> [low, high]
        scale = (action_sim_high - action_sim_low) / 2.0
        mean = (action_sim_high + action_sim_low) / 2.0

        def _reset(
            env: "NormalizeActions", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["NormalizeActions", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(
            env: "NormalizeActions", actions: Array
        ) -> tuple["NormalizeActions", tuple[Any, ...]]:
            # actions are expected in [-1, 1]; clip and rescale to simulator range
            action = jp.clip(actions, -1.0, 1.0) * scale + mean
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region ZeroYaw
@struct.dataclass
class ZeroYaw(Wrapper):
    """Wrapper to set yaw output to zero."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> "ZeroYaw":
        """Create an ZeroYaw around `base`.

        Parameters:
            base: The jittable base environment to wrap.

        Returns:
            ZeroYaw: A configured wrapper with jitted step/reset.
        """

        def _reset(
            env: "ZeroYaw", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["ZeroYaw", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(env: "ZeroYaw", actions: Array) -> tuple["ZeroYaw", tuple[Any, ...]]:
            actions = actions.at[..., 2].set(0.0)
            base_env, (obs, rewards, terminations, truncations, infos) = env.base.step(
                env.base, actions
            )

            env = env.replace(base=base_env)
            return env, (obs, rewards, terminations, truncations, infos)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region AngleReward
@struct.dataclass
class AngleReward(Wrapper):
    """Wrapper to penalize orientation in the reward."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode, rpy_coef: float = 0.08) -> "AngleReward":
        """Create an AngleReward around `base`.

        Parameters:
            base: The jittable base environment to wrap.
            rpy_coef: Coefficient used to penalize roll/pitch/yaw magnitudes in rewards.

        Returns:
            AngleReward: A configured wrapper with jitted step/reset.
        """

        def _reset(
            env: "AngleReward", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["AngleReward", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _rewards(rewards: Array, observations: dict[str, Array]) -> Array:
            """Additional angular rewards."""
            # apply rpy penalty
            rotvec_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_rotvec(), axis=-1)
            rewards -= rpy_coef * rotvec_norm
            return rewards

        def _step(env: "AngleReward", actions: Array) -> tuple["AngleReward", tuple[Any, ...]]:
            base_env, (obs, rewards, terminations, truncations, infos) = env.base.step(
                env.base, actions
            )
            rewards = _rewards(rewards, obs)

            env = env.replace(base=base_env)
            return env, (obs, rewards, terminations, truncations, infos)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region ActionNoise
@struct.dataclass
class ActionNoise(Wrapper):
    """Wrapper to apply random disturbances on action.

    Apply this wrapper after NormalizeActions and before ActionPenalty.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    rng_key: Array = struct.field(pytree_node=True)
    action_bias: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode,
        seed: int | None = None,
        bias_range: float = 0.1,
        noise_std: float = 0.01,
    ) -> "ActionNoise":
        """Create an ActionNoise around `base`.

        Parameters:
            base: The jittable base environment to wrap.
            seed: Random seed for noise generation.
            noise_std: Standard deviation of the Gaussian noise to apply to actions.

        Returns:
            ActionNoise: A configured wrapper with jitted step/reset.
        """
        rng_key = jax.random.PRNGKey(seed) if seed is not None else 0
        action_bias = jp.zeros(base.action_space.shape)

        def _reset(
            env: "ActionNoise", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["ActionNoise", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            rng_key = jax.random.PRNGKey(seed) if seed is not None else 0
            rng_key, subkey = jax.random.split(rng_key)
            action_bias, _ = _sample_noise(subkey, env.action_bias, bias_range, noise_std, None)
            env = env.replace(base=base_env, rng_key=rng_key, action_bias=action_bias)
            return env, (obs, info)

        def _step(env: "ActionNoise", actions: Array) -> tuple["ActionNoise", tuple[Any, ...]]:
            rng_key, subkey = jax.random.split(env.rng_key)
            # 1. sample noise (bias + additive gaussian)
            action_bias, additive_noise = _sample_noise(
                subkey, env.action_bias, bias_range, noise_std, env.unwrapped._marked_for_reset
            )
            # 2. apply noise and step env
            actions = actions + jax.lax.stop_gradient(action_bias + additive_noise)
            base_env, (obs, rewards, terminations, truncations, infos) = env.base.step(
                env.base, actions
            )
            env = env.replace(base=base_env, rng_key=rng_key, action_bias=action_bias)
            return env, (obs, rewards, terminations, truncations, infos)

        def _sample_noise(
            key: Array, action_bias: Array, bias_range: Array, noise_std: Array, mask: Array | None
        ) -> Array:
            key1, key2 = jax.random.split(key)
            new_bias = jax.random.uniform(
                key1, shape=action_bias.shape, minval=-bias_range, maxval=bias_range
            )
            new_additive = jax.random.normal(key2, shape=action_bias.shape) * noise_std
            if mask is not None:  # update action bias upon reset
                new_bias = jp.where(mask[..., None], new_bias, action_bias)
            return new_bias, new_additive

        return cls(
            base=base,
            rng_key=rng_key,
            action_bias=action_bias,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


# region ActionPenalty
@struct.dataclass
class ActionPenalty(Wrapper):
    """Wrapper to apply action penalty and augment observations with last_action."""

    base: struct.PyTreeNode | AngleReward = struct.field(pytree_node=True)
    last_actions: Array = struct.field(pytree_node=True)
    num_actions: int = struct.field(pytree_node=False)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Return the base single_observation_space augmented with action_history."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        act_dim = self.base.action_space.shape[-1]
        spec["last_actions"] = spaces.Box(
            -np.inf, np.inf, shape=(self.num_actions, act_dim), dtype=np.float32
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: struct.PyTreeNode | AngleReward,
        num_actions: int = 1,
        init_last_actions: Array | None = None,
        hover_action: Array = jp.zeros((4,)),
        act_coefs: tuple = (0.0,) * 4,
        d_act_coefs: tuple = (0.0,) * 4,
    ) -> "ActionPenalty":
        """Create an ActionPenalty that augments observations with `last_action` and applies action-based penalties to rewards.

        Parameters:
            base: The jittable environment to wrap.
            act_coef, d_act_th_coef, d_act_xy_coef: Coefficients controlling the penalty terms.

        Returns:
            ActionPenalty: Configured wrapper instance.
        """
        num_envs = base.num_envs
        act_dim = base.action_space.shape[-1]
        act_coefs = jp.array(act_coefs, dtype=jp.float32)
        d_act_coefs = jp.array(d_act_coefs, dtype=jp.float32)
        # last_actions is part of the observation dict (computed in property)
        last_actions = jp.zeros((num_envs, num_actions, act_dim), dtype=jp.float32)
        if init_last_actions is not None:
            last_actions = jp.broadcast_to(init_last_actions, last_actions.shape)

        def _reset(
            env: "ActionPenalty", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["ActionPenalty", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(base=base_env, last_actions=jp.zeros_like(env.last_actions))
            obs["last_actions"] = env.last_actions
            return env, (obs, info)

        def _step(env: "ActionPenalty", action: Array) -> tuple["ActionPenalty", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            # energy
            action_diviation = action - hover_action
            reward = reward - jp.sum(act_coefs * (action_diviation**2), axis=-1)
            # smoothness
            action_diff = action - env.last_actions[:, 0, :]
            reward = reward - jp.sum(d_act_coefs * (action_diff**2), axis=-1)
            # update action history
            new_last_actions = jp.roll(env.last_actions, shift=1, axis=1)
            new_last_actions = new_last_actions.at[:, 0, :].set(action)
            env = env.replace(base=base_env, last_actions=new_last_actions)

            # flatten action history for observation: (num_envs, num_actions * act_dim)
            obs["last_actions"] = env.last_actions
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            last_actions=last_actions,
            num_actions=num_actions,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


# region StackObs
@struct.dataclass
class StackObs(Wrapper):
    """Wrapper to stack history observations.

    This wrapper appends `prev_obs` to the observation dict. `prev_obs` contains the last
    `n_obs` basic observations concatenated in chronological order.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    n_obs: int = struct.field(pytree_node=False)
    prev_obs: Array = struct.field(pytree_node=True)  # (num_envs, n_obs, 13)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Base single_observation_space augmented with `prev_obs`."""
        spec = {k: v for k, v in self.base.single_observation_space.items()}
        spec["prev_obs"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13 * self.n_obs,), dtype=np.float32
        )
        return spaces.Dict(spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(cls, base: struct.PyTreeNode, n_obs: int = 1) -> "StackObs":
        """Create a StackObs around `base`.

        Parameters:
            base: The jittable base environment to wrap.
            n_obs: Number of historical observations to stack. Must be >= 1.

        Returns:
            StackObs: Configured wrapper instance.
        """
        if n_obs < 1:
            raise ValueError(f"StackObs requires n_obs >= 1, got {n_obs}.")

        prev_obs = jp.zeros((base.num_envs, n_obs, 13), dtype=jp.float32)

        def _basic_obs(obs: dict[str, Array]) -> Array:
            """Extract the 13D basic observation [pos, quat, vel, ang_vel]."""
            basic_keys = ("pos", "quat", "vel", "ang_vel")
            return jp.concatenate(
                [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_keys], axis=-1
            )

        @jax.jit
        def _update_prev_obs(prev_obs: Array, obs: dict[str, Array]) -> Array:
            """Roll history and write current basic obs into the last slot."""
            basic = _basic_obs(obs)  # (num_envs, 13)
            prev_obs = jp.roll(prev_obs, shift=1, axis=1)
            prev_obs = prev_obs.at[:, 0, :].set(basic)
            return prev_obs

        def _reset(
            env: "StackObs", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["StackObs", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            basic = _basic_obs(obs)  # (num_envs, 13)
            init_prev = jp.broadcast_to(basic[:, None, :], env.prev_obs.shape)
            env = env.replace(base=base_env, prev_obs=init_prev)
            obs["prev_obs"] = env.prev_obs.reshape(env.num_envs, -1)
            return env, (obs, info)

        def _step(env: "StackObs", action: Array) -> tuple["StackObs", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            obs["prev_obs"] = env.prev_obs.reshape(env.num_envs, -1)
            new_prev = _update_prev_obs(env.prev_obs, obs)

            env = env.replace(base=base_env, prev_obs=new_prev)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base, n_obs=n_obs, prev_obs=prev_obs, step=jax.jit(_step), reset=jax.jit(_reset)
        )


# region FlattenObs
@struct.dataclass
class FlattenJaxObservation(Wrapper):
    """Wrapper to flatten dict observations into a single array."""

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
    def create(cls, base: Any) -> "FlattenJaxObservation":
        """Create a FlattenJaxObservation that concatenates dict observations.

        Parameters:
            base: The jittable environment to wrap.

        Returns:
            FlattenJaxObservation: Configured wrapper instance.
        """

        def flatten_obs(observations: dict[str, Array]) -> Array:
            return jp.concatenate(
                [jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1
            )

        def _reset(
            env: "FlattenJaxObservation", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["FlattenJaxObservation", tuple[Array, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, info)

        def _step(
            env: "FlattenJaxObservation", action: Array
        ) -> tuple["FlattenJaxObservation", tuple[Array, Any]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            flat_obs = flatten_obs(obs)
            env = env.replace(base=base_env)
            return env, (flat_obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


# region ObsNoise
@struct.dataclass
class ObsNoise(Wrapper):
    """Wrapper to add Gaussian noise to observations.

    Apply this wrapper after FlattenJaxObservation.
    """

    base: struct.PyTreeNode = struct.field(pytree_node=True)
    noise_std: float = struct.field(pytree_node=False)
    rng_key: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, base: struct.PyTreeNode, noise_std: float = 0.01, seed: int | None = 0
    ) -> "ObsNoise":
        """Create an ObsNoise around `base`.

        Parameters:
            base: The jittable base environment to wrap.
            noise_std: Standard deviation of Gaussian noise.
            seed: Random seed for noise generation.

        Returns:
            ObsNoise: Configured wrapper instance.
        """
        rng_key = jax.random.PRNGKey(seed if seed is not None else 0)

        def _add_noise(key: Array, obs: Array, std: Array) -> Array:
            """Add noise to array obs."""
            noise = jax.random.normal(key, shape=obs.shape) * std
            return obs + noise

        def _reset(
            env: "ObsNoise", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["ObsNoise", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)

            rng_key = env.rng_key
            if seed is not None:
                rng_key = jax.random.PRNGKey(seed)
            rng_key, subkey = jax.random.split(rng_key)

            obs = _add_noise(subkey, obs, jp.array(env.noise_std, dtype=jp.float32))
            env = env.replace(base=base_env, rng_key=rng_key)
            return env, (obs, info)

        def _step(env: "ObsNoise", action: Array) -> tuple["ObsNoise", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            rng_key, subkey = jax.random.split(env.rng_key)
            obs = _add_noise(subkey, obs, jp.array(env.noise_std, dtype=jp.float32))

            env = env.replace(base=base_env, rng_key=rng_key)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            noise_std=noise_std,
            rng_key=rng_key,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


# region RecordData
@struct.dataclass
class RecordData(Wrapper):
    """Wrapper that records debugging data."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    _record_act: Array = struct.field(pytree_node=True)
    _record_pos: Array = struct.field(pytree_node=True)
    _record_goal: Array = struct.field(pytree_node=True)
    _record_rpy: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> "RecordData":
        """Create a RecordData wrapper around `base`.

        Parameters:
            base: The jittable environment to wrap.

        Returns:
            RecordData: Configured wrapper instance.
        """
        assert hasattr(base.unwrapped, "max_episode_time"), (
            "Base env must have max_episode_time attribute"
        )
        max_T = int(base.unwrapped.max_episode_time * base.unwrapped.freq)
        num_envs = int(base.num_envs)
        act_dim = int(base.action_space.shape[-1])
        pos_dim = 3

        # initialize buffers # TODO: this takes a lot of memory
        empty_act = jp.zeros((max_T, num_envs, act_dim), dtype=jp.float32)
        empty_pos = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_goal = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_rpy = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)

        def _reset(
            env: "RecordData", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["RecordData", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            env = env.replace(
                base=base_env,
                _record_act=empty_act,
                _record_pos=empty_pos,
                _record_goal=empty_goal,
                _record_rpy=empty_rpy,
            )
            return env, (obs, info)

        def _step(env: "RecordData", action: Array) -> tuple["RecordData", tuple[Any, ...]]:
            # step the wrapped environment
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            # extract host-visible sim arrays from the advanced base_env
            raw = base_env.unwrapped

            act = action  # shape: (num_envs, act_dim)
            pos = raw.data.states.pos[:, 0, :]
            if hasattr(raw, "trajectories"):
                goal = raw.trajectories[jp.arange(raw.steps.shape[0]), raw.steps.squeeze(1)]
            if hasattr(raw, "goal_pos"):
                goal = raw.goal_pos
            rpy = R.from_quat(raw.data.states.quat[:, 0, :]).as_euler("xyz")

            # record data
            new_act = env._record_act.at[raw.steps, ...].set(act)
            new_pos = env._record_pos.at[raw.steps, ...].set(pos)
            new_goal = env._record_goal.at[raw.steps, ...].set(goal)
            new_rpy = env._record_rpy.at[raw.steps, ...].set(rpy)

            env = env.replace(
                base=base_env,
                _record_act=new_act,
                _record_pos=new_pos,
                _record_goal=new_goal,
                _record_rpy=new_rpy,
            )
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            _record_act=empty_act,
            _record_pos=empty_pos,
            _record_goal=empty_goal,
            _record_rpy=empty_rpy,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

    def calc_rmse(self) -> float:
        """Compute RMSE between recorded position and goal (return in meters)."""
        pos = np.array(self._record_pos)  # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)  # shape: (T, num_envs, 3)
        pos_err = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        rmse = np.sqrt(np.mean(pos_err**2))
        return rmse

    def plot_eval(self, save_path: str = "eval_plot.png") -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        # Actions
        if self.base.unwrapped.control == "attitude":
            action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        elif self.base.unwrapped.control == "force_torque":
            action_labels = ["Thrust", "Tx", "Ty", "Tz"]
        elif self.base.unwrapped.control == "rotor_vel":
            action_labels = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]
        else:
            raise ValueError(f"Unsupported control type: {self.base.unwrapped.sim.control}")
        for i in range(4):
            axes[i].plot(actions[:, 0, i])
            axes[i].set_title(f"{action_labels[i]} Command")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Action Value")
            axes[i].grid(True)

        # Position plots and goals
        for i, label in enumerate(["X Position", "Y Position", "Z Position"]):
            axes[4 + i].plot(pos[:, 0, i])
            axes[4 + i].plot(goal[:, 0, i], linestyle="--")
            axes[4 + i].set_title(label)
            axes[4 + i].set_xlabel("Time Step")
            axes[4 + i].set_ylabel("Position (m)")
            axes[4 + i].grid(True)
            axes[4 + i].legend(["Position", "Goal"])

        # Position error
        pos_err = np.linalg.norm(pos[:, 0] - goal[:, 0], axis=1)
        axes[7].plot(pos_err)
        axes[7].set_title("Position Error")
        axes[7].set_xlabel("Time Step")
        axes[7].set_ylabel("Error (m)")
        axes[7].grid(True)

        # Angles
        rpy_labels = ["Roll", "Pitch", "Yaw"]
        for i in range(3):
            axes[8 + i].plot(rpy[:, 0, i])
            axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
            axes[8 + i].set_xlabel("Time Step")
            axes[8 + i].set_ylabel("Angle (rad)")
            axes[8 + i].grid(True)

        # compute RMSE for position
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        axes[11].text(0.1, 0.5, f"Position RMSE: {rmse_pos * 1000:.3f} mm", fontsize=14)
        axes[11].axis("off")

        plt.tight_layout()
        plt.savefig(
            Path(__file__).parents[2] / "saves" / save_path
        )  # TODO: nicer way to get root path

        return fig


# region Examples
if __name__ == "__main__":
    import time  # noqa: I001
    from rl_diffsim.envs.figure_8_env import FigureEightEnv

    """Test the jittable wrappers implementation."""
    # Create the jittable environment
    env = FigureEightEnv.create(
        num_envs=1024,
        max_episode_time=10.0,
        physics="so_rpy_rotor_drag",
        drone_model="cf21B_500",
        freq=50,
        device="gpu",
        n_samples=10,
        trajectory_time=10.0,
        samples_dt=0.1,
        reset_rotor=True,
    )
    # wrap the environment
    env = NormalizeActions.create(env)
    env = AngleReward.create(env)
    env = ActionPenalty.create(env)
    env = FlattenJaxObservation.create(env)
    env = RecordData.create(env)

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Trajectories:", env.unwrapped.trajectories.shape)
    print("Initial Obs:", obs.shape)

    def step_once(env: FigureEightEnv, _) -> tuple[FigureEightEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.1], dtype=jp.float32)  # fixed action
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.unwrapped.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.unwrapped.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(env: FigureEightEnv, num_steps: int) -> tuple[FigureEightEnv, tuple[Array, Array]]:
        """Rollout for multiple steps using lax.scan."""
        env, (pos_traj, vel_traj) = jax.lax.scan(step_once, env, xs=None, length=num_steps)
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

    # test rendering & data logging
    env, (obs, info) = env.reset(env)
    for step in range(100):
        base_action = jp.array([0.0, 0.2, 0.0, 0.0], dtype=jp.float32)  # fixed action
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)
        env, _ = env.step(env, action)
        env.render()
    # env.plot_eval(save_path="eval_plot_test.png")

    print("\nPos trajectory shape:", pos_traj.shape)
    print("Vel trajectory shape:", vel_traj.shape)

    print("obs.trajectories[0, :5, :]:", env.unwrapped.trajectories[0, :5, :])
