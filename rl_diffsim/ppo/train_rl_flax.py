"""A naive RL pipeline for drone racing."""
import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import optax
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium.vector import VectorEnv
from jax import Array
from ppo_agent import Agent

import wandb
from rl_diffsim.envs.rand_traj import RandTrajEnv
from rl_diffsim.ppo.wrappers import (
    ActionPenalty,
    AngleReward,
    FlattenJaxObservation,
    RecordData,
)


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"
    """environment device"""
    wandb_project_name: str = "rl-ppo-f8"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
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
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# region MakeEnvs
def make_envs(
        num_envs: int = None,
        jax_device: str = "cpu",
        coefs: dict = {},
        reset_rotor: bool = False,
    ) -> VectorEnv:
    """Make environments for training RL policy."""
    env: RandTrajEnv = RandTrajEnv(
        n_samples=10,
        num_envs=num_envs,
        freq=50,
        drone_model="cf21B_500",
        physics="so_rpy_rotor_drag",
        device=jax_device,
        reset_rotor=reset_rotor,
    )
    
    env = NormalizeActions(env)
    # env = ActionTransform(env)
    env = AngleReward(env, rpy_coef=coefs.get("rpy_coef", 0.04))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.04),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )
    env = FlattenJaxObservation(env)
    # env = ObsNoise(env, noise_std=0.01)
    return env


# region Utils
def set_seeds(seed: int) -> jax.Array:
    """Seed everything."""
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key

# region Train
def compute_gae(rewards: Array, values: Array, dones: Array, last_value: Array, gamma: float, lam: float):
    """Compute GAE advantages and returns (pure JAX)."""
    T = rewards.shape[0]
    advantages = jp.zeros_like(rewards)
    lastgaelam = 0.0
    for t in range(T - 1, -1, -1):
        nextvalues = last_value if t == T - 1 else values[t + 1]
        nextnonterminal = 1.0 - dones[t + 1] if t < T - 1 else 1.0 - dones[-1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages = advantages.at[t].set(lastgaelam)
    returns = advantages + values
    return advantages, returns


def collect_rollout(envs, agent: Agent, actor_params, critic_params, next_obs, next_done, args: Args, key: Array):
    """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""
    # buffers
    obs_buf = jp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions_buf = jp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs_buf = jp.zeros((args.num_steps, args.num_envs))
    rewards_buf = jp.zeros((args.num_steps, args.num_envs))
    dones_buf = jp.zeros((args.num_steps, args.num_envs))
    values_buf = jp.zeros((args.num_steps, args.num_envs))

    sum_rewards = jp.zeros((args.num_envs,))
    sum_rewards_hist: list[tuple[int, float]] = []
    global_step = 0

    obs = next_obs
    dones = next_done
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs_buf = obs_buf.at[step].set(obs)
        dones_buf = dones_buf.at[step].set(dones)

        (action, logprob, entropy), key = agent.get_action(actor_params, obs, key)
        value = agent.get_value(critic_params, obs)

        values_buf = values_buf.at[step].set(jp.squeeze(value))
        actions_buf = actions_buf.at[step].set(action)
        logprobs_buf = logprobs_buf.at[step].set(logprob)

        # step envs (envs uses numpy interface)
        next_obs_np, reward, terminations, truncations, infos = envs.step(action)
        next_obs = jp.array(next_obs_np)
        reward = jp.array(reward)
        terminations = jp.array(terminations)
        truncations = jp.array(truncations)

        rewards_buf = rewards_buf.at[step].set(reward)
        sum_rewards = sum_rewards + reward
        sum_rewards = jp.where(dones, 0.0, sum_rewards)
        dones = terminations | truncations

        if dones.any():
            for r in sum_rewards[dones]:
                sum_rewards_hist.append((global_step, float(r)))

        obs = next_obs

    data = {
        "obs": obs_buf,
        "actions": actions_buf,
        "logprobs": logprobs_buf,
        "rewards": rewards_buf,
        "dones": dones_buf,
        "values": values_buf,
        "next_obs": obs,
        "next_done": dones,
        "sum_rewards_hist": sum_rewards_hist,
    }
    return data, obs, dones, key


def update_policy(agent: Agent, args: Args, data: dict, envs) -> float:
    """Performs PPO updates using optax / flax TrainState. Returns explained variance."""
    # bootstrap last value
    last_value = agent.get_value(agent.critic_states.params, data["next_obs"])  # shape (num_envs,)
    advantages, returns = compute_gae(data["rewards"], data["values"], data["dones"], last_value, args.gamma, args.gae_lambda)

    # flatten
    b_obs = data["obs"].reshape((-1,) + envs.single_observation_space.shape)
    b_actions = data["actions"].reshape((-1,) + envs.single_action_space.shape)
    b_logprobs = data["logprobs"].reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = data["values"].reshape(-1)

    def loss_fn(actor_params, critic_params, obs_mb, acts_mb, old_logprobs_mb, adv_mb, ret_mb, val_mb):
        mean, logstd = agent.actor.apply(actor_params, obs_mb)
        std = jp.exp(logstd)
        logp = -0.5 * (((acts_mb - mean) / (std + 1e-8)) ** 2 + 2.0 * logstd + jp.log(2.0 * jp.pi))
        logp = jp.sum(logp, axis=-1)
        ratio = jp.exp(logp - old_logprobs_mb)
        pg_loss1 = -adv_mb * ratio
        pg_loss2 = -adv_mb * jp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jp.mean(jp.maximum(pg_loss1, pg_loss2))
        entropy = jp.sum(0.5 * (1.0 + jp.log(2.0 * jp.pi)) + logstd, axis=-1)
        entropy_loss = jp.mean(entropy)

        value = agent.critic.apply(critic_params, obs_mb).reshape(-1)
        if args.clip_vloss:
            v_unclipped = (value - ret_mb) ** 2
            v_clipped = val_mb + jp.clip(value - val_mb, -args.clip_coef, args.clip_coef)
            v_clipped_loss = (v_clipped - ret_mb) ** 2
            v_loss = 0.5 * jp.mean(jp.maximum(v_unclipped, v_clipped_loss))
        else:
            v_loss = 0.5 * jp.mean((value - ret_mb) ** 2)

        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        approx_kl = jp.mean((ratio - 1.0) - (logp - old_logprobs_mb))
        return loss, (pg_loss, v_loss, entropy_loss, approx_kl)

    b_inds = np.arange(args.batch_size)
    rng = np.random.default_rng()
    for epoch in range(args.update_epochs):
        rng.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            mb_inds = b_inds[start : start + args.minibatch_size]
            obs_mb = b_obs[mb_inds]
            acts_mb = b_actions[mb_inds]
            old_logprobs_mb = b_logprobs[mb_inds]
            adv_mb = b_advantages[mb_inds]
            ret_mb = b_returns[mb_inds]
            val_mb = b_values[mb_inds]

            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            (loss_val, aux), (g_actor, g_critic) = grad_fn(agent.actor_states.params, agent.critic_states.params, obs_mb, acts_mb, old_logprobs_mb, adv_mb, ret_mb, val_mb)

            agent.actor_states = agent.actor_states.apply_gradients(grads=g_actor)
            agent.critic_states = agent.critic_states.apply_gradients(grads=g_critic)

    # explained variance
    y_pred = np.asarray(b_values)
    y_true = np.asarray(b_returns)
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return explained_var


def train_ppo(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> None:
    """Train.

    JAX/Flax training: collect rollouts using the numpy env API, compute GAE in JAX,
    and update policy/value via Flax TrainState and Optax.
    """
    # setup training
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))

    train_start_time = time.time()
    key = set_seeds(args.seed)
    print("Training on device:", jax_device)

    # make envs
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    envs = make_envs(num_envs=args.num_envs, jax_device=jax_device, coefs=r_coefs, reset_rotor=True)

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
    agent = Agent(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        hidden_size=64,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )

    # start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = jp.array(next_obs)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards_hist = []

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()
        data, next_obs, next_done, key = collect_rollout(envs, agent, agent.actor_states.params, agent.critic_states.params, next_obs, next_done, args, key)
        explained_var = update_policy(agent, args, data, envs)
        global_step += args.batch_size
        sum_rewards_hist.extend(data["sum_rewards_hist"])

        if wandb_enabled:
            # log rewards_hist
            for step, reward in data["sum_rewards_hist"]:
                wandb.log({"train/reward": reward}, step=global_step+step)
            wandb.log({"losses/explained_variance": float(explained_var), "charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step)

        print(f"Iter {iteration}/{args.num_iterations} took {time.time() - start_time:.2f} seconds")

    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    if model_path is not None:
        params = {"actor": agent.actor_states.params, "critic": agent.critic_states.params}
        with open(model_path, "wb") as f:
            import pickle
            pickle.dump(params, f)
        print(f"model saved to {model_path}")
    envs.close()

    return sum_rewards_hist

# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float, list, list]:
    """Evaluate the trained policy (Flax/Agent).

    Loads params from `model_path` (pickle of {'actor':..., 'critic':...}) and runs
    `n_eval` episodes with deterministic actions.
    """
    set_seeds(args.seed)
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_envs(num_envs=1, coefs=r_coefs)
    eval_env = RecordData(eval_env)

    # create Agent (weights will be overwritten)
    key = set_seeds(args.seed)
    init_key, _ = jax.random.split(key)
    agent = Agent(key=init_key, obs_dim=eval_env.single_observation_space.shape, act_dim=eval_env.single_action_space.shape)

    # load params
    with open(model_path, "rb") as f:
        import pickle

        params = pickle.load(f)

    # replace params in train states
    if "actor" in params:
        agent.actor_states = agent.actor_states.replace(params=params["actor"])
    if "critic" in params:
        agent.critic_states = agent.critic_states.replace(params=params["critic"])

    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed

    for episode in range(n_eval):
        obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
        obs = jp.array(obs)
        done = False
        episode_reward = 0.0
        steps = 0
        key = set_seeds(ep_seed)
        while not done:
            (action, _, _), key = agent.get_action(agent.actor_states.params, obs, key, deterministic=True)
            # step env (numpy interface)
            obs_np, reward, terminated, truncated, info = eval_env.step(np.asarray(action))
            eval_env.render()
            done = bool(terminated | truncated)
            episode_reward += float(np.asarray(reward).item())
            steps += 1
            obs = jp.array(obs_np)

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    # plot figures, record RMSE if available
    try:
        fig, _, _ = eval_env.plot_eval()
        rmse_pos = eval_env.calc_rmse()
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
    model_path = Path(__file__).parents[2] / "saves/ppo_model.ckpt"
    jax_device = args.jax_device

    if train:  # use "--do_train False" to skip training
        _ = train_ppo(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:  # use "--n_eval <N>" to perform N evaluation episodes
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_ppo(args, n_eval, model_path)
        if wandb_enabled:
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
