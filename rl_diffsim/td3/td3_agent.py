"""TD3 agent implementation using Flax.

Twin Delayed Deep Deterministic Policy Gradient (TD3) with:
- Deterministic actor with tanh output
- Twin critics (Q1, Q2) to reduce overestimation
- Target networks with soft updates
- Delayed policy updates
- Target policy smoothing
"""

from typing import Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


class ActorNet(nn.Module):
    """Deterministic actor network for TD3."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Forward pass returning deterministic action in [-1, 1]."""
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        action = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        action = nn.tanh(action)  # Bound to [-1, 1]
        return action


class CriticNet(nn.Module):
    """Q-network for TD3. Takes concatenated (obs, action) as input."""

    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array, action: Array) -> Array:
        """Forward pass returning Q-value."""
        x = jp.concatenate([obs, action], axis=-1)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        q_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(x)
        return q_value.squeeze(-1)


class TD3Agent(struct.PyTreeNode):
    """TD3 agent with asymmetric actor-critic observations."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    critic1_state: train_state.TrainState = struct.field(pytree_node=True)
    critic2_state: train_state.TrainState = struct.field(pytree_node=True)
    target_actor_params: dict = struct.field(pytree_node=True)
    target_critic1_params: dict = struct.field(pytree_node=True)
    target_critic2_params: dict = struct.field(pytree_node=True)

    # Inference functions (not part of pytree)
    get_action_mean: Callable = struct.field(pytree_node=False)
    get_action_sample: Callable = struct.field(pytree_node=False)
    get_random_action: Callable = struct.field(pytree_node=False)
    get_q: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        actor_obs_dim: int | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> "TD3Agent":
        """Initialize TD3 agent with actor, twin critics, and target networks."""
        # Create networks (single critic instance, params are separate)
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim, num_layers=num_layers)
        critic = CriticNet(hidden_size=hidden_size, num_layers=num_layers)

        # Initialize parameters
        actor_obs_dim = actor_obs_dim if actor_obs_dim is not None else obs_dim
        k1, k2, k3 = jax.random.split(key, 3)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        dummy_actor_obs = dummy_obs[..., :actor_obs_dim]  # Slice for actor input
        dummy_action = jp.zeros((1, act_dim), dtype=jp.float32)

        actor_params = actor.init(k1, dummy_actor_obs)
        critic1_params = critic.init(k2, dummy_obs, dummy_action)
        critic2_params = critic.init(k3, dummy_obs, dummy_action)

        # Create optimizers
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        critic_tx = optax.adamw(learning_rate=critic_lr, eps=1e-5)

        # Create train states (both critics share the same apply_fn)
        actor_states = train_state.TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
        critic1_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic1_params, tx=critic_tx
        )
        critic2_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic2_params, tx=critic_tx
        )

        # Initialize target networks with same parameters
        target_actor_params = actor_params
        target_critic1_params = critic1_params
        target_critic2_params = critic2_params

        # Build jittable inference functions (slice obs for actor)
        def _get_action_mean(params: dict, obs: Array) -> Array:
            """Get deterministic action (for evaluation). Slices first actor_obs_dim dims."""
            return actor.apply(params, obs[..., :actor_obs_dim])

        def _get_action_sample(
            params: dict, obs: Array, key: Array, std: float, noise_clip: float | None = None
        ) -> tuple[Array, Array]:
            """Get action with exploration noise (for training). Slices first actor_obs_dim dims."""
            mean = actor.apply(params, obs[..., :actor_obs_dim])
            new_key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, mean.shape, dtype=mean.dtype) * std
            if noise_clip is not None:
                noise = jp.clip(noise, -noise_clip, noise_clip)
            action = jp.clip(mean + noise, -1.0, 1.0)
            return action, new_key

        def _get_random_action(num_envs: int, key: Array) -> tuple[Array, Array]:
            """Get action with exploration noise (for training). Slices first actor_obs_dim dims."""
            new_key, sub = jax.random.split(key)
            action = jax.random.uniform(sub, (num_envs, act_dim), minval=-1.0, maxval=1.0, dtype=jp.float32)
            return action, new_key

        def _get_q(params: dict, obs: Array, action: Array) -> Array:
            """Get Q value (works with either critic's params)."""
            return critic.apply(params, obs, action)

        return cls(
            actor_states=actor_states,
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_actor_params=target_actor_params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            get_action_mean=jax.jit(_get_action_mean),
            get_action_sample=jax.jit(_get_action_sample, static_argnames=("std", "noise_clip")),
            get_random_action=jax.jit(_get_random_action, static_argnames=("num_envs",)),
            get_q=jax.jit(_get_q),
        )


if __name__ == "__main__":
    """Test the TD3 agent implementation."""
    actor_obs_dim, critic_obs_dim, act_dim = 17, 21, 4

    # Create agent
    key = jax.random.PRNGKey(0)
    agent = TD3Agent.create(
        key=key,
        actor_obs_dim=actor_obs_dim,
        obs_dim=critic_obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )
    print("Agent created successfully")

    # Test action inference
    obs = jp.ones((2, actor_obs_dim), dtype=jp.float32)
    action = agent.get_action_mean(agent.actor_states.params, obs)
    print(f"Deterministic action shape: {action.shape}")

    # Test noisy action
    key = jax.random.PRNGKey(1)
    noisy_action, key = agent.get_action_sample(agent.actor_states.params, obs, key, std=0.1)
    print(f"Noisy action shape: {noisy_action.shape}")

    # Test Q-value inference
    critic_obs = jp.ones((2, critic_obs_dim), dtype=jp.float32)
    q1 = agent.get_q(agent.critic1_state.params, critic_obs, action)
    q2 = agent.get_q(agent.critic2_state.params, critic_obs, action)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")

    print("\nAll tests passed!")
