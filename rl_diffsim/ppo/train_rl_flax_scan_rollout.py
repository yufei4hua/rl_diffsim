"""A naive RL pipeline for drone racing."""
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
from ppo_agent import Agent

import wandb
from rl_diffsim.envs.figure_8_env_jittable import FigureEightJittableEnv
from rl_diffsim.ppo.wrappers_jittable import (
    ActionPenaltyJittable,
    AngleRewardJittable,
    FlattenJaxObservationJittable,
    NormalizeActionsJittable,
    RecordDataJittable,
    RenderSimJittable,
)


# region Arguments
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""
    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "gpu"
    """environment device"""
    wandb_project_name: str = "rl-ppo-f8"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    actor_lr: float = 1.5e-3
    """the learning rate of the actor optimizer"""
    critic_lr: float = 1.5e-3
    """the learning rate of the critic optimizer"""
    gamma: float = 0.94
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.007
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 1.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 0.06
    d_act_th_coef: float = 0.4
    d_act_xy_coef: float = 1.0
    act_coef: float = 0.02
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
        num_envs: int = None,
        jax_device: str = "cpu",
        coefs: dict = {},
        reset_rotor: bool = False,
    ) -> FigureEightJittableEnv:
    """Make environments for training RL policy."""
    env: FigureEightJittableEnv = FigureEightJittableEnv.create(
        n_samples=10,
        num_envs=num_envs,
        freq=50,
        drone_model="cf21B_500",
        physics="so_rpy_rotor_drag",
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
    obs: Array
    actions: Array
    logprobs: Array
    rewards: Array
    dones: Array
    values: Array
    advantages: Array
    returns: Array

# region Rollout
@functools.partial(jax.jit, static_argnames=("args",))
def collect_rollout(
        envs: struct.PyTreeNode,
        args: Args,
        agent: Agent,
        next_obs: Array,
        next_done: Array,
        sum_rewards: Array,
        key: Array,
    ) -> tuple[RolloutData, Array, Array, Array, Array]:
    """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""

    # loop over args.num_steps
    def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
        env, key, sum_rewards, obs, dones = carry
        # 1. get action from policy
        (action, logprob, entropy), key = agent.get_action_sample(agent.actor_states.params, obs, key)
        value = agent.get_value(agent.critic_states.params, obs)

        # 2. step environment
        env, (next_obs, reward, terminations, truncations, info) = env.step(env, action)
        sum_rewards = sum_rewards + reward
        sum_rewards = jp.where(dones, 0.0, sum_rewards)
        next_dones = terminations | truncations

        return (env, key, sum_rewards, next_obs, next_dones), (obs, action, logprob, reward, dones, value)

    (envs, key, sum_rewards, next_obs, next_done), rollout_data = jax.lax.scan(
        step_once, 
        (envs, key, sum_rewards, next_obs, next_done), 
        length=args.num_steps
    )

    # construct rollout data
    obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf = rollout_data
    data = RolloutData(
        obs=obs_buf,
        actions=actions_buf,
        logprobs=logprobs_buf,
        rewards=rewards_buf,
        dones=dones_buf,
        values=values_buf,
        returns=jp.zeros_like(rewards_buf),
        advantages=jp.zeros_like(rewards_buf),
    )
    return envs, data, next_obs, next_done, sum_rewards, key

# region GAE
@functools.partial(jax.jit, static_argnames=("args",))
def compute_gae(
    args: Args,
    agent: Agent,
    next_obs: Array,
    next_done: Array,
    data: RolloutData,
) -> RolloutData:
    """Compute GAE advantages and returns."""
    last_value = agent.get_value(agent.critic_states.params, next_obs)

    advantages = jp.zeros((args.num_envs,))
    dones = jp.concatenate([data.dones, next_done[None, :]], axis=0)
    values = jp.concatenate([data.values, last_value[None, :]], axis=0)

    def compute_gae_once(carry: Array, inp: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        """Compute one step of GAE in scan."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    _, advantages = jax.lax.scan(
        compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], data.rewards), reverse=True
    )
    data = data.replace(
        advantages=advantages,
        returns=advantages + data.values,
    )
    return data

# region PG
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(
        args: Args,
        agent: Agent,
        data: RolloutData,
        key: Array,
    ) -> float:
    """Performs PPO updates. Returns losses."""

    # loss function
    def loss_fn(actor_params, critic_params, obs_mb, acts_mb, old_logprobs_mb, adv_mb, ret_mb, val_mb):  # noqa
        # ppo policy loss
        logp, entropy = agent.get_action_logprob(actor_params, obs_mb, acts_mb)
        ratio = jp.exp(logp - old_logprobs_mb)
        pg_loss1 = -adv_mb * ratio
        pg_loss2 = -adv_mb * jp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jp.mean(jp.maximum(pg_loss1, pg_loss2))
        entropy_loss = -jp.mean(entropy)

        # ppo value loss (clip by default)
        value = agent.get_value(critic_params, obs_mb).reshape(-1)
        v_unclipped = (value - ret_mb) ** 2
        v_clipped = val_mb + jp.clip(value - val_mb, -args.clip_coef, args.clip_coef)
        v_clipped_loss = (v_clipped - ret_mb) ** 2
        v_loss = 0.5 * jp.mean(jp.maximum(v_unclipped, v_clipped_loss))

        loss = pg_loss + args.ent_coef * entropy_loss + args.vf_coef * v_loss
        approx_kl = jp.mean((ratio - 1.0) - (logp - old_logprobs_mb))
        return loss, (pg_loss, v_loss, entropy_loss, approx_kl)
    
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    # batch: loop over epochs
    def update_epoch(carry: tuple[Agent, jax.random.PRNGKey], inp: int) -> tuple[Agent, jax.random.PRNGKey]:
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
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (g_actor, g_critic) = grad_fn(
                agent.actor_states.params, 
                agent.critic_states.params, 
                minibatch.obs, 
                minibatch.actions, 
                minibatch.logprobs, 
                minibatch.advantages, 
                minibatch.returns, 
                minibatch.values
            )
            return agent.replace(
                actor_states=agent.actor_states.apply_gradients(grads=g_actor),
                critic_states=agent.critic_states.apply_gradients(grads=g_critic)
            ), (pg_loss, v_loss, entropy_loss, approx_kl)
        
        agent, (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(update_minibatch, agent, minibatches)

        return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl)
    
    (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(update_epoch, (agent, key), (), length=args.update_epochs)
    
    y_pred = data.values.reshape(-1)
    y_true = data.returns.reshape(-1)
    var_y = jp.var(y_true)
    explained_var = jp.where(var_y == 0, jp.nan, 1.0 - jp.var(y_true - y_pred) / var_y)
    return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var)

# region Train
def train_ppo(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> None:
    """Train.

    JAX/Flax training: collect rollouts using the numpy env API, compute GAE in JAX,
    and update policy/value via Flax TrainState and Optax.
    """
    # setup training
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))

    train_start_time = time.time()
    key = jax.random.PRNGKey(args.seed)
    print("Training on device:", jax_device)

    # make envs
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    envs = make_jitted_envs(num_envs=args.num_envs, jax_device=jax_device, coefs=r_coefs, reset_rotor=True)
    envs = RenderSimJittable.create(envs)

    # setup annealing learning rate
    train_steps = args.num_iterations * args.update_epochs * args.num_minibatches
    if args.anneal_lr:
        actor_lr = optax.linear_schedule(init_value=args.actor_lr, end_value=0.0, transition_steps=train_steps)
        critic_lr = optax.linear_schedule(init_value=args.critic_lr, end_value=0.0, transition_steps=train_steps)
    else:
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr

    # setup agent
    init_key, key = jax.random.split(key)
    agent = Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=64,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )

    # start the game
    global_step = 0
    envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iter {iteration}/{args.num_iterations}", end=": ")
        start_time = time.time()
        # 1. collect rollouts
        envs, data, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs=envs, 
            args=args, 
            agent=agent,
            next_obs=next_obs, 
            next_done=next_done, 
            sum_rewards=sum_rewards,
            key=key
        )
        envs.render(envs)
        print(f"Rollouts {time.time() - start_time:.5f} s", end=", ")
        # 2. compute GAE
        start_gae_time = time.time()
        data = compute_gae(args, agent, next_obs, next_done, data)
        print(f"GAE {time.time() - start_gae_time:.5f} s", end=", ")
        # 3. update policy
        start_pg_time = time.time()
        (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var) = update_policy(args, agent, data, key)
        print(f"PG {time.time() - start_pg_time:.5f} s", end=", ")
        # 4. logging
        print(f"total {time.time() - start_time:.5f} s")

        if wandb_enabled:
            wandb.log({"train/reward": jp.mean(sum_rewards[next_done])}, step=global_step + args.batch_size)
            wandb.log({
                "losses/pg_loss": jp.mean(pg_loss),
                "losses/v_loss": jp.mean(v_loss),
                "losses/entropy": -jp.mean(entropy_loss),
                "losses/approx_kl": jp.mean(approx_kl),
                "losses/explained_variance": jp.mean(explained_var),
                "charts/SPS": int(global_step / (time.time() - train_start_time)),
            }, step=global_step + args.batch_size)

        global_step += args.batch_size

    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    if model_path is not None:
        params = {"actor": agent.actor_states.params, "critic": agent.critic_states.params}
        with open(model_path, "wb") as f:
            import pickle
            pickle.dump(params, f)
        print(f"model saved to {model_path}")
    envs.close()


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float, list, list]:
    """Evaluate the trained policy (Flax/Agent).

    Loads params from `model_path` (pickle of {'actor':..., 'critic':...}) and runs
    `n_eval` episodes with deterministic actions.
    """
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_jitted_envs(num_envs=1, jax_device=args.jax_device, coefs=r_coefs)
    eval_env = RecordDataJittable.create(eval_env)
    eval_env = RenderSimJittable.create(eval_env)

    # create Agent & load params
    agent = Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=eval_env.single_observation_space.shape[0],
        act_dim=eval_env.single_action_space.shape[0],
        hidden_size=64,
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
        # jitted env reset returns (env, (obs, info))
        eval_env, (obs, info) = eval_env.reset(eval_env, seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            action = agent.get_action_mean(agent.actor_states.params, obs)
            # step env using jittable API: env, (obs, reward, terminated, truncated, info)
            eval_env, (obs, reward, terminated, truncated, info) = eval_env.step(eval_env, action)
            # render using the RenderSimJittable API
            eval_env.render(eval_env)
            done = (terminated | truncated)
            # reward is a jax Array shaped (1,) — convert to float
            episode_reward += float(np.asarray(reward).item())
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    # plot figures, record RMSE if available
    try:
        fig, _, _ = eval_env.base.plot_eval()
        rmse_pos = eval_env.base.calc_rmse()
    except Exception:
        fig, rmse_pos = None, None

    eval_env.close()

    return fig, rmse_pos, episode_rewards, episode_lengths


# region Main
def main(wandb_enabled: bool = True, train: bool = True, n_eval: int = 1):
    """Main entry.

    Flags:
      wandb_enabled: log metrics to wandb
      train: run training
      n_eval: number of evaluation episodes to run after training (or standalone)
    """
    args = Args.create()
    model_path = Path(__file__).parents[2] / "saves/ppo_model_flax.ckpt"
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_ppo(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:  # use "--n_eval <N>" to perform N evaluation episodes
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_ppo(args, n_eval, model_path)
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
