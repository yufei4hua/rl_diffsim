"""PPO agent implementation using Flax."""

from typing import Callable

import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax import struct
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


class ActorNet(nn.Module):
    """Class defining the actor-critic model."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Simple MLP model for actor-critic."""
        # Actor mean
        x = obs
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros, name=f"fc_{i}")(x)
            x = nn.tanh(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)
        # Actor logstd
        actor_logstd = self.param(
            "actor_logstd",
            lambda rng, shape: jp.array([[-2.0, -2.0, -2.0, -1.5]], dtype=jp.float32),
            # lambda rng, shape: -1.0 * jp.array([[1.0, 1.0, 1.0, 1.0]], dtype=jp.float32),
            (1, self.act_dim),
        )
        logstd = jp.broadcast_to(actor_logstd, (mean.shape[0], self.act_dim))

        return mean, logstd


class Agent(struct.PyTreeNode):
    """SHAC agent class with actor and critic networks."""

    actor_states: train_state.TrainState = struct.field(pytree_node=True)
    get_action_mean: Callable = struct.field(pytree_node=False)
    get_action_sample: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        actor_lr: float | optax.Schedule = 3e-4,
    ) -> "Agent":
        """Initialize the SHAC agent's actor and critic networks."""
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim, num_layers=num_layers)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        actor_params = actor.init(key, dummy_obs)
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        actor_states = train_state.TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=actor_tx
        )

        # Build jittable actor and critic inference functions
        def _get_action_sample(params: dict, obs: Array, key: Array) -> tuple[Array, Array]:
            """Get stochastic action sample for collecting rollout."""
            mean, logstd = actor.apply(params, obs)
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps
            logp = -0.5 * (eps**2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return (action, logp, entropy), new_key

        def _get_action_mean(params: dict, obs: Array) -> Array:
            """Get deterministic action (mean)."""
            mean, logstd = actor.apply(params, obs)
            return mean

        return cls(
            actor_states=actor_states,
            get_action_mean=jax.jit(_get_action_mean),
            get_action_sample=jax.jit(_get_action_sample),
        )


if __name__ == "__main__":
    """Test the agent implementation."""
    # initialization
    obs_dim, act_dim = 13, 4
    agent = Agent.create(
        jax.random.PRNGKey(0), obs_dim=obs_dim, act_dim=act_dim, hidden_size=64, actor_lr=3e-4
    )

    obs = jp.ones((2, obs_dim), dtype=jp.float32)

    # rollout - sample actions
    key = jax.random.PRNGKey(1)
    (action, logp, entropy), key = agent.get_action_sample(agent.actor_states.params, obs, key)
    print("Rollout:")
    print(action, logp, entropy)

    # deployment - get deterministic action
    action = agent.get_action_mean(agent.actor_states.params, obs)
    print("Deployment:")
    print(action)
