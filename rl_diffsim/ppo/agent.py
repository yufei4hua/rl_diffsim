""""PPO agent implementation using Flax."""
import jax
import jax.numpy as jp
from flax import linen as nn
from flax.linen.initializers import orthogonal, zeros
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


def init_agent(key: Array, obs_dim: int, act_dim: int, hidden_size: int = 64) -> dict:
    """Initialize the PPO agent's actor and critic networks."""
    actor = ActorNet(hidden_size=hidden_size, act_dim=act_dim)
    critic = CriticNet(hidden_size=hidden_size)
    k1, k2 = jax.random.split(key)
    dummy_obs = jp.zeros((1, obs_dim), dtype=jp.float32)
    actor_params = actor.init(k1, dummy_obs)
    critic_params = critic.init(k2, dummy_obs)
    return actor, critic, {"actor": actor_params, "critic": critic_params}

def get_value(critic: CriticNet, params: dict, obs: Array) -> Array:
    """Get value from the critic network."""
    return critic.apply(params["critic"], obs)

def get_action_and_value(
    actor: ActorNet,
    critic: CriticNet,
    params: dict,
    obs: Array,
    key: Array,
    action: Array = None,
    deterministic: bool = False,
):
    """Get action, log probability, entropy, and value from the actor-critic networks."""
    mean, logstd = actor.apply(params["actor"], obs)
    std = jp.exp(logstd)

    if action is None:
        if deterministic:
            action = mean
            new_key = key
        else:
            new_key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, mean.shape, dtype=mean.dtype)
            action = mean + std * eps

    logp = -0.5 * (((action - mean) / (std + 1e-8)) ** 2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
    logp = jp.sum(logp, axis=-1)
    entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)

    value = critic.apply(params["critic"], obs)

    return (action, logp, entropy, value), new_key


if __name__ == "__main__":
    # initialization
    obs_dim, act_dim = 13, 4
    actor, critic, params = init_agent(
        jax.random.PRNGKey(0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
    )

    # forward pass
    obs = jp.zeros((obs_dim,), dtype=jp.float32)
    v = get_value(critic, params, obs)
    print("Value:", v)

    key = jax.random.PRNGKey(1)
    (action, logp, ent, value), key = get_action_and_value(actor, critic, params, obs, key)
    print("Action:", action)

    # JIT compilation
    get_av_jit = jax.jit(lambda p, o, k: get_action_and_value(actor, critic, p, o, k))
    (action2, logp2, ent2, value2), key = get_av_jit(params, obs, key)

    get_v_jit = jax.jit(lambda p, o: get_value(critic, p, o))
    v2 = get_v_jit(params, obs)
