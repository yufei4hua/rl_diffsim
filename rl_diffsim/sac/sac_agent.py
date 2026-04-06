"""SAC agent implementation using Flax.

Soft Actor-Critic (SAC) with:
- Stochastic actor with learned mean and logstd
- Twin critics (Q1, Q2) to reduce overestimation
- Target critic networks with soft updates
- Automatic temperature (alpha) tuning
- Entropy maximization for exploration
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
    """Stochastic actor network for SAC with learned logstd."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Forward pass returning action mean and logstd."""
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(), bias_init=zeros)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.act_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(x)
        mean = nn.tanh(mean)  # Bound to [-1, 1]

        # Learnable logstd parameter (state-independent, like PPO)
        actor_logstd = self.param(
            "actor_logstd", lambda rng, shape: jp.zeros(shape, dtype=jp.float32), (1, self.act_dim)
        )
        logstd = jp.broadcast_to(actor_logstd, (mean.shape[0], self.act_dim))
        logstd = jp.clip(logstd, self.log_std_min, self.log_std_max)

        return mean, logstd


class CriticNet(nn.Module):
    """Q-network for SAC. Takes concatenated (obs, action) as input."""

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


class SACAgent(struct.PyTreeNode):
    """SAC agent with stochastic policy and automatic temperature tuning."""

    actor_state: train_state.TrainState = struct.field(pytree_node=True)
    critic1_state: train_state.TrainState = struct.field(pytree_node=True)
    critic2_state: train_state.TrainState = struct.field(pytree_node=True)
    target_critic1_params: dict = struct.field(pytree_node=True)
    target_critic2_params: dict = struct.field(pytree_node=True)

    # Temperature (alpha) parameter
    log_alpha: Array = struct.field(pytree_node=True)
    alpha_optimizer_state: optax.OptState = struct.field(pytree_node=True)
    target_entropy: float = struct.field(pytree_node=False)
    alpha_lr: float = struct.field(pytree_node=False)

    # Inference functions (not part of pytree)
    get_action_mean: Callable = struct.field(pytree_node=False)
    get_action_sample: Callable = struct.field(pytree_node=False)
    get_action_and_logprob: Callable = struct.field(pytree_node=False)
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
        alpha_lr: float = 3e-4,
        init_alpha: float = 1.0,
        init_logstd: Array | None = None,
    ) -> "SACAgent":
        """Initialize SAC agent with actor, twin critics, and temperature parameter."""
        # Create networks
        actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim, num_layers=num_layers)
        critic = CriticNet(hidden_size=hidden_size, num_layers=num_layers)

        # Initialize parameters
        actor_obs_dim = actor_obs_dim if actor_obs_dim is not None else obs_dim
        k1, k2, k3 = jax.random.split(key, 3)
        dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
        dummy_actor_obs = dummy_obs[..., :actor_obs_dim]
        dummy_action = jp.zeros((1, act_dim), dtype=jp.float32)

        actor_params = actor.init(k1, dummy_actor_obs)
        critic1_params = critic.init(k2, dummy_obs, dummy_action)
        critic2_params = critic.init(k3, dummy_obs, dummy_action)

        # Initialize logstd if provided
        if init_logstd is not None:
            init_logstd = jp.broadcast_to(init_logstd[None, :], (1, act_dim))
            actor_params["params"]["actor_logstd"] = init_logstd

        # Create optimizers
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        critic_tx = optax.adamw(learning_rate=critic_lr, eps=1e-5)

        # Create train states
        actor_state = train_state.TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
        critic1_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic1_params, tx=critic_tx
        )
        critic2_state = train_state.TrainState.create(
            apply_fn=critic.apply, params=critic2_params, tx=critic_tx
        )

        # Initialize target critic networks (no target actor in SAC)
        target_critic1_params = critic1_params
        target_critic2_params = critic2_params

        # Temperature (alpha) parameter - learnable log_alpha
        log_alpha = jp.log(jp.array(init_alpha, dtype=jp.float32))
        alpha_optimizer = optax.adam(learning_rate=alpha_lr)
        alpha_optimizer_state = alpha_optimizer.init(log_alpha)

        # Target entropy: -dim(A) (heuristic from SAC paper)
        target_entropy = -float(act_dim)

        # Build jittable inference functions
        def _get_action_mean(params: dict, obs: Array) -> Array:
            """Get deterministic action (mean) for evaluation."""
            mean, _ = actor.apply(params, obs[..., :actor_obs_dim])
            return mean

        def _get_action_sample(params: dict, obs: Array, key: Array) -> tuple[Array, Array, Array]:
            """Get stochastic action sample for data collection.

            Returns:
                (action, log_prob, new_key)
            """
            mean, logstd = actor.apply(params, obs[..., :actor_obs_dim])
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps

            # Compute log probability (Gaussian)
            log_prob = -0.5 * (eps**2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            log_prob = jp.sum(log_prob, axis=-1)

            return action, log_prob, new_key

        def _get_action_and_logprob(
            params: dict, obs: Array, key: Array
        ) -> tuple[Array, Array, Array]:
            """Sample action and compute log probability for SAC training.

            This uses the reparameterization trick and handles tanh squashing.

            Returns:
                (action, log_prob, new_key) where action is in [-1, 1]
            """
            mean, logstd = actor.apply(params, obs[..., :actor_obs_dim])
            std = jp.exp(logstd)
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)

            # Pre-tanh action (for proper log_prob computation)
            pre_tanh_action = mean + std * eps
            action = jp.tanh(pre_tanh_action)

            # Gaussian log probability with tanh correction
            # log π(a|s) = log N(u|μ,σ) - sum(log(1 - tanh²(u)))
            log_prob = -0.5 * (eps**2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
            log_prob = jp.sum(log_prob, axis=-1)

            # Tanh squashing correction (numerically stable)
            log_prob = log_prob - jp.sum(
                jp.log(jp.clip(1.0 - action**2, 1e-6, 1.0) + 1e-6), axis=-1
            )

            return action, log_prob, new_key

        def _get_random_action(num_envs: int, key: Array) -> tuple[Array, Array]:
            """Get uniform random action for initial exploration."""
            new_key, sub = jax.random.split(key)
            action = jax.random.uniform(sub, (num_envs, act_dim), minval=-1.0, maxval=1.0, dtype=jp.float32)
            return action, new_key

        def _get_q(params: dict, obs: Array, action: Array) -> Array:
            """Get Q value (works with either critic's params)."""
            return critic.apply(params, obs, action)

        return cls(
            actor_state=actor_state,
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            log_alpha=log_alpha,
            alpha_optimizer_state=alpha_optimizer_state,
            target_entropy=target_entropy,
            alpha_lr=alpha_lr,
            get_action_mean=jax.jit(_get_action_mean),
            get_action_sample=jax.jit(_get_action_sample),
            get_action_and_logprob=jax.jit(_get_action_and_logprob),
            get_random_action=jax.jit(_get_random_action, static_argnames=("num_envs",)),
            get_q=jax.jit(_get_q),
        )

    @property
    def alpha(self) -> Array:
        """Current temperature value."""
        return jp.exp(self.log_alpha)


if __name__ == "__main__":
    """Test the SAC agent implementation."""
    actor_obs_dim, critic_obs_dim, act_dim = 17, 21, 4

    # Create agent
    key = jax.random.PRNGKey(0)
    agent = SACAgent.create(
        key=key,
        actor_obs_dim=actor_obs_dim,
        obs_dim=critic_obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        init_alpha=0.2,
    )
    print("Agent created successfully")
    print(f"Initial alpha: {agent.alpha}")
    print(f"Target entropy: {agent.target_entropy}")

    # Test action inference
    obs = jp.ones((2, actor_obs_dim), dtype=jp.float32)
    action = agent.get_action_mean(agent.actor_state.params, obs)
    print(f"Deterministic action shape: {action.shape}")

    # Test stochastic action sampling
    key = jax.random.PRNGKey(1)
    action, log_prob, key = agent.get_action_sample(agent.actor_state.params, obs, key)
    print(f"Sampled action shape: {action.shape}, log_prob shape: {log_prob.shape}")

    # Test action and logprob (for training)
    action, log_prob, key = agent.get_action_and_logprob(agent.actor_state.params, obs, key)
    print(f"Action (tanh squashed) shape: {action.shape}, log_prob shape: {log_prob.shape}")

    # Test Q-value inference
    critic_obs = jp.ones((2, critic_obs_dim), dtype=jp.float32)
    q1 = agent.get_q(agent.critic1_state.params, critic_obs, action)
    q2 = agent.get_q(agent.critic2_state.params, critic_obs, action)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")

    print("\nAll tests passed!")
