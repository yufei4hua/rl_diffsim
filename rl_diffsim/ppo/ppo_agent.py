""""PPO agent implementation using Flax."""
import functools
from typing import Any, Callable

import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


class ActorNet(nn.Module):
    """Class defining the actor-critic model."""
    hidden_size: int = 64
    act_dim: int = 4

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Simple MLP model for actor-critic."""
        # Actor mean
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)
        # Actor logstd
        actor_logstd = self.param(
            "actor_logstd",
            lambda rng, shape: jp.zeros(shape, dtype=jp.float32),
            (1, self.act_dim),
        )
        logstd = jp.broadcast_to(actor_logstd, (mean.shape[0], self.act_dim))

        return mean, logstd

class CriticNet(nn.Module):
    """Class defining the critic model."""
    hidden_size: int = 64

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Simple MLP model for critic."""
        x = obs
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
        x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=zeros)(x)
        return value.squeeze(-1)

class Agent:
    """PPO agent class with actor and critic networks."""
    def __init__(
            self,
            key: Array, 
            obs_dim: int, 
            act_dim: int, 
            hidden_size: int = 64,
            actor_lr: float | optax.Schedule = 3e-4,
            critic_lr: float | optax.Schedule = 1e-3,
        ) -> dict:
        """Initialize the PPO agent's actor and critic networks."""
        self.actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim)
        self.critic = CriticNet(hidden_size=hidden_size)
        k1, k2 = jax.random.split(key)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        actor_params = self.actor.init(k1, dummy_obs)
        critic_params = self.critic.init(k2, dummy_obs)
        actor_tx = optax.adam(learning_rate=actor_lr)
        critic_tx = optax.adam(learning_rate=critic_lr)
        self.actor_states = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.critic_states = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_tx,
        )
        # Build jittable actor and critic inference functions
        def _get_action_sample(params: dict, obs: Array, key: Array) -> tuple[Array, Array]:
            """Get stochastic action sample for collecting rollout."""
            mean, logstd = self.actor.apply(params, obs)
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps
            logp = -0.5 * (eps ** 2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return (action, logp, entropy), new_key 

        def _get_action_logprob(params: dict, obs: Array, action: Array) -> tuple[Array, Array]:
            """Get log probability and entropy of given action for training."""
            mean, logstd = self.actor.apply(params, obs)
            std = jp.exp(logstd)
            logp = -0.5 * (((action - mean) / (std + 1e-8)) ** 2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            logp = jp.sum(logp, axis=-1)
            entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
            return logp, entropy

        def _get_action_mean(params: dict, obs: Array) -> Array:
            """Get deterministic action (mean)."""
            mean, logstd = self.actor.apply(params, obs)
            return mean

        def _get_value(params: dict, obs: Array) -> Array:
            return jp.squeeze(self.critic.apply(params, obs))

        self.get_action_mean = jax.jit(_get_action_mean)
        self.get_action_sample = jax.jit(_get_action_sample)
        self.get_action_logprob = jax.jit(_get_action_logprob)
        self.get_value = jax.jit(_get_value)


if __name__ == "__main__":
    """Test the agent implementation."""
    # initialization
    obs_dim, act_dim = 13, 4
    agent = Agent(
        jax.random.PRNGKey(0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        actor_lr=3e-4,
        critic_lr=1e-3,
    )

    obs = jp.ones((2, obs_dim), dtype=jp.float32)

    # rollout - sample actions
    key = jax.random.PRNGKey(1)
    (action, logp, entropy), key = agent.get_action_sample(
        agent.actor_states.params,
        obs,
        key,
    )
    value = agent.get_value(agent.critic_states.params, obs)
    print("Rollout:")
    print(action, logp, entropy, value)

    # optimization - get log probabilities
    chosen_action = action
    logp, entropy = agent.get_action_logprob(
        agent.actor_states.params,
        obs,
        chosen_action,
    )
    value = agent.get_value(agent.critic_states.params, obs)
    print("Optimization:")
    print(chosen_action, logp, entropy, value)

    # deployment - get deterministic action
    action = agent.get_action_mean(agent.actor_states.params, obs)
    print("Deployment:")
    print(action)
