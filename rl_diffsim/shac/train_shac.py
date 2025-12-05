"""An SHAC implementation based on Flax."""

import functools
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fire
import flax
import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
import optax
from jax import Array

import wandb
from rl_diffsim.envs.figure_8_env_jittable import FigureEightJittableEnv  # noqa: F401
from rl_diffsim.envs.reach_pos_env_jittable import ReachPosJittableEnv
from rl_diffsim.envs.wrappers_jittable import (
    ActionPenaltyJittable,
    AngleRewardJittable,
    FlattenJaxObservationJittable,
    NormalizeActionsJittable,
    RecordDataJittable,
)
from rl_diffsim.shac.shac_agent import Agent


# region Arguments
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
    """environment device"""
    wandb_project_name: str = "rl-shac-rp"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 500_000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    actor_lr: float = 1.4e-2
    """the learning rate of the actor optimizer"""
    critic_lr: float = 1.0e-3
    """the learning rate of the critic optimizer"""
    gamma: float = 0.98
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the TD-lambd a calculation"""
    update_epochs: int = 12
    """the K epochs to update the policy"""
    clip_coef: float = 0.4
    """the surrogate clipping coefficient"""
    hidden_size: int = 64
    """the hidden size of actor and critic networks"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    rpy_coef: float = 0.1
    d_act_th_coef: float = 1.0
    d_act_xy_coef: float = 1.0
    act_coef: float = 0.0
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        batch_size = int(args.num_envs * args.num_steps)
        minibatch_size = int(batch_size // args.num_minibatches)
        num_iterations = args.total_timesteps // batch_size
        return replace(
            args,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            num_iterations=num_iterations,
        )


# region MakeEnvs
def make_jitted_envs(
    num_envs: int = None, jax_device: str = "cpu", coefs: dict = {}, reset_rotor: bool = False
) -> ReachPosJittableEnv:
    """Make environments for training RL policy."""
    env: ReachPosJittableEnv = ReachPosJittableEnv.create(
        # n_samples=10,
        num_envs=num_envs,
        freq=50,
        drone_model="cf21B_500",
        physics="first_principles",
        control="rotor_vel",
        device=jax_device,
        reset_rotor=reset_rotor,
    )

    env = NormalizeActionsJittable.create(env)
    env = AngleRewardJittable.create(env, rpy_coef=coefs.get("rpy_coef", 0.04))
    env = ActionPenaltyJittable.create(
        env,
        act_coef=coefs.get("act_coef", 0.04),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )
    env = FlattenJaxObservationJittable.create(env)
    return env


# region Utils
@flax.struct.dataclass
class RolloutData:
    """Class for storing rollout data."""

    observations: Array
    actions: Array
    rewards: Array
    dones: Array
    values: Array
    sum_rewards: Array
    entropy: Array
    returns: Array
    losses: Array


def global_max(pytree: dict) -> Array:
    """Compute the global max abs value in a pytree."""
    leaf_max = [jp.max(jp.abs(x)) for x in jax.tree.leaves(pytree)]
    return jp.max(jp.stack(leaf_max))


# region Policy Update
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(
    envs: struct.PyTreeNode,
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    sum_rewards: Array,
    key: Array,
) -> float:
    """SHAC policy updates."""

    def collect_rollout(
        envs: struct.PyTreeNode,
        actor_params: dict,
        critic_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[RolloutData, Array, Array, Array, Array]:
        """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""
        discounts = jp.ones((args.num_envs,))

        # loop over args.num_steps
        def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
            env, key, sum_rewards, discounts, obs, dones = carry
            # 1. get action from policy
            (action, logprob, entropy), key = agent.get_action_sample(actor_params, obs, key)
            value = agent.get_value(critic_params, obs)

            # 2. step environment & compute stepwise loss
            env, (next_obs, reward, terminations, truncations, info) = env.step(env, action)

            sum_rewards = sum_rewards + reward
            next_sum_rewards = jp.where(dones, 0.0, sum_rewards)
            loss = -discounts * jp.where(dones, value, reward)
            next_discounts = jp.where(dones, 1.0, discounts * args.gamma)
            next_dones = terminations | truncations

            return (env, key, next_sum_rewards, next_discounts, next_obs, next_dones), RolloutData(
                observations=obs,
                actions=action,
                rewards=reward,
                dones=dones,
                values=value,
                sum_rewards=sum_rewards,
                entropy=jp.mean(entropy),
                returns=jp.zeros_like(reward),
                losses=loss,
            )

        (envs, key, sum_rewards, next_discounts, next_obs, next_done), rollout_data = jax.lax.scan(
            step_once,
            (envs, key, sum_rewards, discounts, next_obs, next_done),
            length=args.num_steps,
        )

        return envs, rollout_data, next_discounts, next_obs, next_done, sum_rewards, key

    def policy_loss_fn(
        envs: struct.PyTreeNode,
        actor_params: dict,
        critic_params: dict,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        # shac policy loss
        envs, data, next_discounts, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs, actor_params, critic_params, next_obs, next_done, sum_rewards, key
        )
        # compute loss as in SHAC paper Eq(5)
        last_value = agent.get_value(critic_params, next_obs)  # (args.num_envs, )
        losses = jp.sum(data.losses) - jp.sum(next_discounts * last_value)  # terminal value loss
        return losses / (args.num_envs * args.num_steps), (
            envs,
            data,
            next_obs,
            next_done,
            sum_rewards,
            key,
        )

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, argnums=(1,), has_aux=True)
    (p_loss, (envs, data, next_obs, next_done, sum_rewards, key)), (g_actor,) = policy_grad_fn(
        envs,
        agent.actor_states.params,
        agent.critic_states.params,
        next_obs,
        next_done,
        sum_rewards,
        key,
    )

    agent = agent.replace(actor_states=agent.actor_states.apply_gradients(grads=g_actor))
    return (envs, agent, key), (p_loss, (data, next_obs, next_done, sum_rewards))


# region TD-λ
@functools.partial(jax.jit, static_argnames=("args",))
def compute_td_lambda(
    args: Args, agent: Agent, next_obs: Array, next_done: Array, data: RolloutData
) -> RolloutData:
    """Compute TD-λ returns."""
    last_value = agent.get_value(agent.critic_states.params, next_obs)

    returns = last_value.reshape(args.num_envs)
    dones = jp.concatenate([data.dones, next_done[None, :]], axis=0)
    values = jp.concatenate([data.values, last_value[None, :]], axis=0)

    def compute_td_lambda_once(
        carry: Array, inp: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Compute one step of TD(lambda) in scan."""
        returns = carry
        nextdone, nextvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        returns = (
            reward
            + args.gamma
            * ((1.0 - args.gae_lambda) * nextvalues + args.gae_lambda * returns)
            * nextnonterminal
        )
        return returns, returns

    _, returns = jax.lax.scan(
        compute_td_lambda_once, returns, (dones[1:], values[1:], data.rewards), reverse=True
    )
    data = data.replace(returns=returns)
    return data


# region Value Update
@functools.partial(jax.jit, static_argnames=("args",))
def update_value(args: Args, agent: Agent, data: RolloutData, key: Array) -> float:
    """SHAC value updates."""

    def value_loss_fn(
        critic_params: dict, obs_mb: Array, ret_mb: Array, val_mb: Array
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        # ppo manner value loss (clip by default)
        value = agent.get_value(critic_params, obs_mb).reshape(-1)
        v_unclipped = (value - ret_mb) ** 2
        v_clipped = val_mb + jp.clip(value - val_mb, -args.clip_coef, args.clip_coef)
        v_clipped_loss = (v_clipped - ret_mb) ** 2
        v_loss = 0.5 * jp.mean(jp.maximum(v_unclipped, v_clipped_loss))
        return v_loss

    value_grad_fn = jax.value_and_grad(value_loss_fn)

    # batch: loop over epochs
    def update_epoch(
        carry: tuple[Agent, jax.random.PRNGKey], inp: int
    ) -> tuple[Agent, jax.random.PRNGKey]:
        agent, key = carry
        key, subkey = jax.random.split(key)

        def process_data(x: Array) -> Array:
            x = x.reshape((-1,) + x.shape[2:])
            x = jax.random.permutation(subkey, x)
            x = jp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x

        minibatches = jax.tree_util.tree_map(process_data, data)

        # minibatch: scan minibatches
        def update_minibatch(carry: Agent, minibatch: int) -> Agent:
            agent = carry
            v_loss, g_critic = value_grad_fn(
                agent.critic_states.params,
                minibatch.observations,
                minibatch.returns,
                minibatch.values,
            )
            return agent.replace(
                critic_states=agent.critic_states.apply_gradients(grads=g_critic)
            ), v_loss

        agent, v_loss = jax.lax.scan(update_minibatch, agent, minibatches)

        return (agent, key), v_loss

    (agent, key), v_loss = jax.lax.scan(update_epoch, (agent, key), (), length=args.update_epochs)

    y_pred = data.values.reshape(-1)
    y_true = data.returns.reshape(-1)
    var_y = jp.var(y_true)
    explained_var = jp.where(var_y == 0, jp.nan, 1.0 - jp.var(y_true - y_pred) / var_y)
    return (agent, key), (v_loss, explained_var)


# region Train
def train_shac(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> None:
    """Train.

    JAX/Flax training: collect rollouts using the numpy env API, compute GAE in JAX,
    and update policy/value via Flax TrainState and Optax.
    """
    # setup training
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    sum_rewards_hist = []

    setup_start_time = time.time()
    key = jax.random.PRNGKey(args.seed)
    print("Training on device:", jax_device)

    # make envs
    r_coefs = {
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    envs = make_jitted_envs(
        num_envs=args.num_envs, jax_device=jax_device, coefs=r_coefs, reset_rotor=True
    )

    # setup annealing learning rate
    train_steps = args.num_iterations * args.update_epochs * args.num_minibatches
    if args.anneal_lr:
        actor_lr = optax.linear_schedule(
            init_value=args.actor_lr, end_value=0.0, transition_steps=train_steps
        )
        critic_lr = optax.linear_schedule(
            init_value=args.critic_lr, end_value=0.0, transition_steps=train_steps
        )
    else:
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr

    # setup agent
    init_key, key = jax.random.split(key)
    agent = Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=args.hidden_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )
    print("Make envs and agent took {:.5f} s".format(time.time() - setup_start_time))

    # warmup jax compile
    start_warmup_time = time.time()
    envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    for _ in range(2):
        (_, _, _), (p_loss, (data, next_obs, next_done, sum_rewards)) = update_policy(
            envs=envs,
            args=args,
            agent=agent,
            next_obs=next_obs,
            next_done=jp.zeros(args.num_envs, dtype=bool),
            sum_rewards=jp.zeros((args.num_envs,)),
            key=key,
        )
        data = compute_td_lambda(args, agent, next_obs, next_done, data)
        (_, _), (v_loss, explained_var) = update_value(args, agent, data, key)
    print("JAX warmup took {:.5f} s".format(time.time() - start_warmup_time))

    # start the game
    train_start_time = time.time()
    global_step = 0
    # envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iter {iteration}/{args.num_iterations}", end=": ")
        start_time = time.time()
        # 1. rollout and policy update
        (envs, agent, key), (p_loss, (data, next_obs, next_done, sum_rewards)) = update_policy(
            envs=envs,
            args=args,
            agent=agent,
            next_obs=next_obs,
            next_done=next_done,
            sum_rewards=sum_rewards,
            key=key,
        )
        print(f"Policy {time.time() - start_time:.5f} s", end=", ")
        # 2. compute TD-λ
        start_gae_time = time.time()
        data = compute_td_lambda(args, agent, next_obs, next_done, data)
        print(f"TD-λ {time.time() - start_gae_time:.5f} s", end=", ")
        # 3. value update
        start_value_time = time.time()
        (agent, key), (v_loss, explained_var) = update_value(args, agent, data, key)
        print(f"Value {time.time() - start_value_time:.5f} s", end=", ")
        print(f"total {time.time() - start_time:.5f} s")
        # 4. logging
        if wandb_enabled:
            for batch_step, (sum_reward, done) in enumerate(zip(data.sum_rewards, data.dones)):
                if jp.any(done):
                    wandb.log(
                        {"train/reward": jp.mean(sum_reward[done])},
                        step=global_step + batch_step * args.num_envs,
                    )
                    sum_rewards_hist.append(jp.mean(sum_reward[done]))
            wandb.log(
                {
                    "losses/p_loss": jp.mean(p_loss),
                    "losses/v_loss": jp.mean(v_loss),
                    "losses/entropy": jp.mean(data.entropy),
                    "losses/explained_variance": jp.mean(explained_var),
                    "charts/SPS": int(global_step / (time.time() - train_start_time)),
                },
                step=global_step + args.batch_size,
            )

        global_step += args.batch_size

    training_time = time.time() - train_start_time
    print(f"Training for {global_step} steps took {training_time:.2f} seconds.")
    if model_path is not None:
        params = {"actor": agent.actor_states.params, "critic": agent.critic_states.params}
        with open(model_path, "wb") as f:
            import pickle

            pickle.dump(params, f)
        print(f"model saved to {model_path}")

    return sum_rewards_hist, training_time


# region Evaluate
def evaluate_shac(
    args: Args, n_eval: int, model_path: Path, render: bool
) -> tuple[float, float, list, list]:
    """Evaluate the trained policy (Flax/Agent).

    Loads params from `model_path` (pickle of {'actor':..., 'critic':...}) and runs
    `n_eval` episodes with deterministic actions.
    """
    r_coefs = {
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_jitted_envs(num_envs=1, jax_device=args.jax_device, coefs=r_coefs)
    eval_env = RecordDataJittable.create(eval_env)

    agent = Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=eval_env.single_observation_space.shape[0],
        act_dim=eval_env.single_action_space.shape[0],
        hidden_size=args.hidden_size,
    )
    with open(model_path, "rb") as f:
        import pickle

        params = pickle.load(f)
    agent = agent.replace(
        actor_states=agent.actor_states.replace(params=params["actor"]),
        critic_states=agent.critic_states.replace(params=params["critic"]),
    )

    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed

    for episode in range(n_eval):
        eval_env, (obs, info) = eval_env.reset(eval_env, seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action = agent.get_action_mean(agent.actor_states.params, obs)
            eval_env, (obs, reward, terminated, truncated, info) = eval_env.step(eval_env, action)
            if render:
                eval_env.render()
            done = terminated | truncated
            episode_reward += float(np.asarray(reward).item())
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    fig = eval_env.plot_eval(save_path="shac_eval_plot.png") if render else None
    rmse_pos = eval_env.calc_rmse()

    eval_env.close()

    return fig, rmse_pos, episode_rewards, episode_lengths


# region Main
def main(wandb_enabled: bool = True, train: bool = True, n_eval: int = 1, render: bool = True):
    """Main entry.

    Flags:
      wandb_enabled: log metrics to wandb
      train: run training
      n_eval: number of evaluation episodes to run after training (or standalone)
      render: whether to render the environment during evaluation
    """
    args = Args.create()
    model_path = Path(__file__).parents[2] / "saves/shac_model_flax.ckpt"
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_shac(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:  # use "--n_eval <N>" to perform N evaluation episodes
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_shac(
            args, n_eval, model_path, render
        )
        if wandb_enabled and train:
            logs = {
                "eval/mean_rewards": np.mean(episode_rewards),
                "eval/mean_steps": np.mean(episode_lengths),
            }
            if fig is not None:
                logs["eval/eval_plot"] = wandb.Image(fig)
            if rmse_pos is not None:
                logs["eval/pos_rmse_mm"] = rmse_pos
            wandb.log(logs)
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
