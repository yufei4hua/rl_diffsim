"""TD3 agent implementation using Flax.

Twin Delayed Deep Deterministic Policy Gradient (TD3) with:
- Deterministic actor with tanh output
- Twin critics (Q1, Q2) to reduce overestimation
- Target networks with soft updates
- Delayed policy updates
- Target policy smoothing
"""

import functools
from typing import Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import optax
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
from flax.training import train_state
from jax import Array


@struct.dataclass
class TD3RolloutData:
    """TD3 rollout data for post-scan logging."""

    rewards: Array  # [num_envs] per step
    dones: Array  # [num_envs] per step
    episode_rewards: Array  # [num_envs] cumulative at step
    episode_lengths: Array  # [num_envs] lengths at step
    critic_loss: Array  # scalar per step
    actor_loss: Array  # scalar per step
    did_update: Array  # bool, whether updates occurred


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

    actor_state: train_state.TrainState = struct.field(pytree_node=True)
    critic1_state: train_state.TrainState = struct.field(pytree_node=True)
    critic2_state: train_state.TrainState = struct.field(pytree_node=True)
    target_actor_params: dict = struct.field(pytree_node=True)
    target_critic1_params: dict = struct.field(pytree_node=True)
    target_critic2_params: dict = struct.field(pytree_node=True)

    # Inference functions (not part of pytree)
    get_action_mean: Callable = struct.field(pytree_node=False)
    get_action_sample: Callable = struct.field(pytree_node=False)
    get_q: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        key: jax.random.PRNGKey,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
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
        k1, k2, k3 = jax.random.split(key, 3)
        dummy_actor_obs = jp.zeros((1, actor_obs_dim), dtype=jp.float32)
        dummy_critic_obs = jp.zeros((1, critic_obs_dim), dtype=jp.float32)
        dummy_action = jp.zeros((1, act_dim), dtype=jp.float32)

        actor_params = actor.init(k1, dummy_actor_obs)
        critic1_params = critic.init(k2, dummy_critic_obs, dummy_action)
        critic2_params = critic.init(k3, dummy_critic_obs, dummy_action)

        # Create optimizers
        actor_tx = optax.adamw(learning_rate=actor_lr, eps=1e-5)
        critic_tx = optax.adamw(learning_rate=critic_lr, eps=1e-5)

        # Create train states (both critics share the same apply_fn)
        actor_state = train_state.TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
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

        # Build jittable inference functions
        def _get_action_mean(params: dict, obs: Array) -> Array:
            """Get deterministic action (for evaluation)."""
            return actor.apply(params, obs)

        def _get_action_sample(
            params: dict, obs: Array, key: Array, std: float, noise_clip: float | None = None
        ) -> tuple[Array, Array]:
            """Get action with exploration noise (for training)."""
            mean = actor.apply(params, obs)
            new_key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, mean.shape, dtype=mean.dtype) * std
            if noise_clip is not None:
                noise = jp.clip(noise, -noise_clip, noise_clip)
            action = jp.clip(mean + noise, -1.0, 1.0)
            return action, new_key

        def _get_q(params: dict, obs: Array, action: Array) -> Array:
            """Get Q value (works with either critic's params)."""
            return critic.apply(params, obs, action)

        return cls(
            actor_state=actor_state,
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_actor_params=target_actor_params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            get_action_mean=jax.jit(_get_action_mean),
            get_action_sample=jax.jit(_get_action_sample, static_argnames=("std", "noise_clip")),
            get_q=jax.jit(_get_q),
        )


# region Replay Buffer


class ReplayBuffer(struct.PyTreeNode):
    """Preallocated circular replay buffer for off-policy learning.

    Stores transitions with separate actor and critic observations for
    asymmetric actor-critic training.
    """

    actor_obs: Array = struct.field(pytree_node=True)
    critic_obs: Array = struct.field(pytree_node=True)
    actions: Array = struct.field(pytree_node=True)
    rewards: Array = struct.field(pytree_node=True)
    next_actor_obs: Array = struct.field(pytree_node=True)
    next_critic_obs: Array = struct.field(pytree_node=True)
    dones: Array = struct.field(pytree_node=True)
    ptr: Array = struct.field(pytree_node=True)
    size: Array = struct.field(pytree_node=True)
    capacity: int = struct.field(pytree_node=False)

    # Jitted callables
    add: Callable = struct.field(pytree_node=False)
    sample: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity: int, actor_obs_dim: int, critic_obs_dim: int, act_dim: int) -> "ReplayBuffer":
        """Create an empty preallocated replay buffer."""

        def _add(
            buffer: "ReplayBuffer",
            actor_obs: Array,
            critic_obs: Array,
            action: Array,
            reward: Array,
            next_actor_obs: Array,
            next_critic_obs: Array,
            done: Array,
        ) -> "ReplayBuffer":
            """Add a batch of transitions to the buffer."""
            batch_size = actor_obs.shape[0]
            indices = (buffer.ptr + jp.arange(batch_size)) % capacity
            return buffer.replace(
                actor_obs=buffer.actor_obs.at[indices].set(actor_obs),
                critic_obs=buffer.critic_obs.at[indices].set(critic_obs),
                actions=buffer.actions.at[indices].set(action),
                rewards=buffer.rewards.at[indices].set(reward),
                next_actor_obs=buffer.next_actor_obs.at[indices].set(next_actor_obs),
                next_critic_obs=buffer.next_critic_obs.at[indices].set(next_critic_obs),
                dones=buffer.dones.at[indices].set(done),
                ptr=(buffer.ptr + batch_size) % capacity,
                size=jp.minimum(buffer.size + batch_size, capacity),
            )

        def _sample(buffer: "ReplayBuffer", batch_size: int, key: Array) -> dict[str, Array]:
            """Sample a random batch of transitions."""
            # Clamp size to capacity for safety (should already be bounded)
            valid_size = jp.minimum(buffer.size, capacity)
            indices = jax.random.randint(key, (batch_size,), 0, valid_size)
            return {
                "actor_obs": buffer.actor_obs[indices],
                "critic_obs": buffer.critic_obs[indices],
                "actions": buffer.actions[indices],
                "rewards": buffer.rewards[indices],
                "next_actor_obs": buffer.next_actor_obs[indices],
                "next_critic_obs": buffer.next_critic_obs[indices],
                "dones": buffer.dones[indices],
            }

        def _reset(buffer: "ReplayBuffer") -> "ReplayBuffer":
            """Reset buffer by setting ptr and size to 0."""
            return buffer.replace(ptr=jp.array(0, dtype=jp.int32), size=jp.array(0, dtype=jp.int32))

        return cls(
            actor_obs=jp.zeros((capacity, actor_obs_dim), dtype=jp.float32),
            critic_obs=jp.zeros((capacity, critic_obs_dim), dtype=jp.float32),
            actions=jp.zeros((capacity, act_dim), dtype=jp.float32),
            rewards=jp.zeros((capacity,), dtype=jp.float32),
            next_actor_obs=jp.zeros((capacity, actor_obs_dim), dtype=jp.float32),
            next_critic_obs=jp.zeros((capacity, critic_obs_dim), dtype=jp.float32),
            dones=jp.zeros((capacity,), dtype=jp.bool_),
            ptr=jp.array(0, dtype=jp.int32),
            size=jp.array(0, dtype=jp.int32),
            capacity=capacity,
            add=jax.jit(_add),
            sample=jax.jit(_sample, static_argnames=("batch_size",)),
            reset=jax.jit(_reset),
        )


# region Updates


@functools.partial(jax.jit, static_argnames=("gamma", "policy_noise", "noise_clip"))
def update_critics(
    agent: TD3Agent, batch: dict[str, Array], gamma: float, policy_noise: float, noise_clip: float, key: Array
) -> tuple[TD3Agent, float, Array]:
    """Update twin critics using TD3 loss with target policy smoothing.

    Returns:
        Updated agent, critic loss, new random key
    """
    # Unpack batch
    critic_obs = batch["critic_obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_critic_obs = batch["next_critic_obs"]
    next_actor_obs = batch["next_actor_obs"]
    dones = batch["dones"]

    # Target policy smoothing: add clipped noise to target actions
    target_actions, key = agent.get_action_sample(
        agent.target_actor_params, next_actor_obs, key, std=policy_noise, noise_clip=noise_clip
    )

    # Compute target Q: min of twin Q targets
    target_q1 = agent.get_q(agent.target_critic1_params, next_critic_obs, target_actions)
    target_q2 = agent.get_q(agent.target_critic2_params, next_critic_obs, target_actions)
    target_q = jp.minimum(target_q1, target_q2)
    target_q = rewards + gamma * (1.0 - dones.astype(jp.float32)) * target_q

    # Shared MSE loss function for both critics
    def mse_loss(params: dict) -> Array:
        q = agent.get_q(params, critic_obs, actions)
        return jp.mean((q - target_q) ** 2)

    c_grad_fn = jax.value_and_grad(mse_loss)

    # Update both critics using the same loss function
    c1_loss, c1_grads = c_grad_fn(agent.critic1_state.params)
    c2_loss, c2_grads = c_grad_fn(agent.critic2_state.params)

    critic1_state = agent.critic1_state.apply_gradients(grads=c1_grads)
    critic2_state = agent.critic2_state.apply_gradients(grads=c2_grads)

    agent = agent.replace(critic1_state=critic1_state, critic2_state=critic2_state)
    critic_loss = (c1_loss + c2_loss) / 2.0

    return agent, critic_loss, key


@functools.partial(jax.jit, static_argnames=("tau",))
def update_actor(agent: TD3Agent, batch: dict[str, Array], tau: float) -> tuple[TD3Agent, float]:
    """Update actor to maximize Q1(s, actor(s)), then soft-update all target networks.

    Returns:
        Updated agent, actor loss
    """
    actor_obs = batch["actor_obs"]
    critic_obs = batch["critic_obs"]

    def actor_loss_fn(params: dict) -> Array:
        actions = agent.get_action_mean(params, actor_obs)
        # Use critic1 for actor update (TD3 uses Q1 only)
        q1 = agent.get_q(agent.critic1_state.params, critic_obs, actions)
        return -jp.mean(q1)  # Maximize Q

    actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(agent.actor_state.params)
    actor_state = agent.actor_state.apply_gradients(grads=actor_grads)

    # Soft update all target networks
    agent = agent.replace(
        actor_state=actor_state,
        target_actor_params=optax.incremental_update(actor_state.params, agent.target_actor_params, tau),
        target_critic1_params=optax.incremental_update(
            agent.critic1_state.params, agent.target_critic1_params, tau
        ),
        target_critic2_params=optax.incremental_update(
            agent.critic2_state.params, agent.target_critic2_params, tau
        ),
    )

    return agent, actor_loss


if __name__ == "__main__":
    """Test the TD3 agent implementation."""
    actor_obs_dim, critic_obs_dim, act_dim = 17, 21, 4

    # Create agent
    key = jax.random.PRNGKey(0)
    agent = TD3Agent.create(
        key=key,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        actor_lr=3e-4,
        critic_lr=3e-4,
    )
    print("Agent created successfully")

    # Test action inference
    obs = jp.ones((2, actor_obs_dim), dtype=jp.float32)
    action = agent.get_action_mean(agent.actor_state.params, obs)
    print(f"Deterministic action shape: {action.shape}")

    # Test noisy action
    key = jax.random.PRNGKey(1)
    noisy_action, key = agent.get_action_sample(agent.actor_state.params, obs, key, std=0.1)
    print(f"Noisy action shape: {noisy_action.shape}")

    # Test Q-value inference
    critic_obs = jp.ones((2, critic_obs_dim), dtype=jp.float32)
    q1 = agent.get_q(agent.critic1_state.params, critic_obs, action)
    q2 = agent.get_q(agent.critic2_state.params, critic_obs, action)
    print(f"Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")

    # Test replay buffer
    buffer = ReplayBuffer.create(
        capacity=1000, actor_obs_dim=actor_obs_dim, critic_obs_dim=critic_obs_dim, act_dim=act_dim
    )
    buffer = buffer.add(
        buffer,
        actor_obs=jp.ones((10, actor_obs_dim)),
        critic_obs=jp.ones((10, critic_obs_dim)),
        action=jp.ones((10, act_dim)),
        reward=jp.ones((10,)),
        next_actor_obs=jp.ones((10, actor_obs_dim)),
        next_critic_obs=jp.ones((10, critic_obs_dim)),
        done=jp.zeros((10,), dtype=jp.bool_),
    )
    print(f"Buffer size: {buffer.size}, ptr: {buffer.ptr}")

    batch = buffer.sample(buffer, 4, jax.random.PRNGKey(2))
    print(f"Sampled batch shapes: {jax.tree_util.tree_map(lambda x: x.shape, batch)}")

    # Test updates
    agent, critic_loss, key = update_critics(
        agent, batch, gamma=0.99, policy_noise=0.2, noise_clip=0.5, key=key
    )
    print(f"Critic loss: {critic_loss:.4f}")

    agent, actor_loss = update_actor(agent, batch, tau=0.005)
    print(f"Actor loss: {actor_loss:.4f}")
    print("Target networks updated")

    print("\nAll tests passed!")
