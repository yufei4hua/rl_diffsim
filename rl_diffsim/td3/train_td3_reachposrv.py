"""TD3 training script for reach position task with rotor velocity control.

TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation for
quadrotor position control using direct RPM setpoints.
"""

import functools
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import fire
import flax.struct as struct
import jax
import jax.lax as lax
import jax.numpy as jp
import numpy as np
import optax
from jax import Array

import wandb
from rl_diffsim.envs.reach_pos_env import ReachPosEnv
from rl_diffsim.envs.wrappers import (
    ActionPenalty,
    FlattenJaxObservation,
    NormalizeActions,
    PrivilegedCriticObs,
    RecordData,
)
from rl_diffsim.td3.td3_agent import TD3Agent


# region Arguments
@dataclass(frozen=True)
class Args:
    """Configuration for TD3 training."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "gpu"
    """environment device"""
    exp_name: str = "td3_rprv"
    """the name of the experiment"""
    wandb_project_name: str = "rl-td3-rprv"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Training
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments"""
    buffer_size: int = 1_000_00
    """replay buffer capacity"""
    batch_size: int = 256
    """minibatch size for updates"""
    learning_starts: int = 400
    """timesteps before training starts (random exploration)"""
    rollout_steps: int = 100
    """N: number of env steps per rollout before training (macro-iteration)"""
    updates_epochs: int = 100
    """M: number of gradient updates per rollout (controls G/U ratio)"""

    # TD3 hyperparameters
    actor_lr: float = 1e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 2e-4
    """the learning rate of the critic optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target network update rate (Polyak averaging)"""
    policy_delay: int = 2
    """update actor every N critic updates"""
    exploration_noise: float = 0.1
    """std of exploration noise during data collection"""
    policy_noise: float = 0.2
    """std of noise added to target policy (smoothing)"""
    noise_clip: float = 0.5
    """clip target policy noise"""

    # Network architecture
    hidden_size: int = 64
    """hidden size of actor and critic networks"""
    num_layers: int = 2
    """number of hidden layers in networks"""

    # to be filled in runtime
    rollout_batchsize: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Reward coefficients
    C_rp: float = 1.0
    """position error coefficient"""
    C_rq: float = 0.1
    """orientation error coefficient"""
    C_rv: float = 0.1
    """velocity penalty coefficient"""
    C_rw: float = 0.1
    """angular velocity penalty coefficient"""
    C_ra: float = 0.01
    """action deviation from hover coefficient"""
    C_rab: float = 0.25
    """hover action baseline (normalized)"""
    C_rs: float = 0.1
    """survival bonus"""

    # Wrapper settings (reuse ActionPenalty for action history)
    num_actions: int = 1
    """number of previous actions to include in observation"""
    act_coefs: tuple = (0.0,) * 4
    """action energy penalty coefficients (handled in PrivilegedCriticObs)"""
    d_act_coefs: tuple = (0.0,) * 4
    """action smoothness penalty coefficients"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class with computed values."""
        args = Args(**kwargs)
        assert args.updates_epochs % args.policy_delay == 0, (
            f"updates_epochs ({args.updates_epochs}) must be divisible by policy_delay ({args.policy_delay})"
        )
        rollout_batchsize = int(args.rollout_steps * args.num_envs)
        num_iterations = (args.total_timesteps - args.learning_starts) // rollout_batchsize

        return replace(args, rollout_batchsize=rollout_batchsize, num_iterations=num_iterations)


# region MakeEnvs
def make_jitted_envs(num_envs: int, jax_device: str, args: Args, reset_rotor: bool = True) -> ReachPosEnv:
    """Create wrapped environments for TD3 training."""
    env: ReachPosEnv = ReachPosEnv.create(
        num_envs=num_envs,
        freq=100,
        sim_freq=500,
        drone_model="cf21B_500",
        physics="first_principles",
        control="rotor_vel",
        device=jax_device,
        reset_rotor=reset_rotor,
    )

    env = NormalizeActions.create(env)
    hover_action = jp.array([args.C_rab] * 4, dtype=jp.float32)
    env = ActionPenalty.create(
        env,
        num_actions=args.num_actions,
        init_last_actions=None,
        hover_action=hover_action,
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )
    env = PrivilegedCriticObs.create(
        env,
        hover_action=hover_action,
        C_rp=args.C_rp,
        C_rq=args.C_rq,
        C_rv=args.C_rv,
        C_rw=args.C_rw,
        C_ra=args.C_ra,
        C_rs=args.C_rs,
    )
    env = FlattenJaxObservation.create(env)

    return env


# region RolloutData
@struct.dataclass
class RolloutData:
    """Class for storing rollout data."""

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    sum_rewards: Array


# region ReplayBuffer
class ReplayBuffer(struct.PyTreeNode):
    """Preallocated circular replay buffer."""

    observations: Array = struct.field(pytree_node=True)
    actions: Array = struct.field(pytree_node=True)
    rewards: Array = struct.field(pytree_node=True)
    next_observations: Array = struct.field(pytree_node=True)
    dones: Array = struct.field(pytree_node=True)
    ptr: Array = struct.field(pytree_node=True)
    size: Array = struct.field(pytree_node=True)
    capacity: int = struct.field(pytree_node=False)

    add: Callable = struct.field(pytree_node=False)
    sample: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, capacity: int, obs_dim: int, act_dim: int) -> "ReplayBuffer":
        """Create an empty preallocated replay buffer."""

        def _add(buffer: "ReplayBuffer", data: RolloutData) -> "ReplayBuffer":
            """Add batched RolloutData (n_steps, n_envs, ...) to buffer."""
            flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), data)
            batch_size = flat.observations.shape[0]
            indices = (buffer.ptr + jp.arange(batch_size)) % capacity
            return buffer.replace(
                observations=buffer.observations.at[indices].set(flat.observations),
                actions=buffer.actions.at[indices].set(flat.actions),
                rewards=buffer.rewards.at[indices].set(flat.rewards),
                next_observations=buffer.next_observations.at[indices].set(flat.next_observations),
                dones=buffer.dones.at[indices].set(flat.dones),
                ptr=(buffer.ptr + batch_size) % capacity,
                size=jp.minimum(buffer.size + batch_size, capacity),
            )

        def _sample(buffer: "ReplayBuffer", batch_size: int, key: Array) -> RolloutData:
            """Sample a random batch of transitions."""
            indices = jax.random.randint(key, (batch_size,), 0, jp.minimum(buffer.size, capacity))
            return RolloutData(
                observations=buffer.observations[indices],
                actions=buffer.actions[indices],
                rewards=buffer.rewards[indices],
                next_observations=buffer.next_observations[indices],
                dones=buffer.dones[indices],
                sum_rewards=jp.zeros((batch_size,)),  # Placeholder, not used in training
            )

        def _reset(buffer: "ReplayBuffer") -> "ReplayBuffer":
            """Reset buffer by setting ptr and size to 0."""
            return buffer.replace(ptr=jp.array(0, dtype=jp.int32), size=jp.array(0, dtype=jp.int32))

        return cls(
            observations=jp.zeros((capacity, obs_dim), dtype=jp.float32),
            actions=jp.zeros((capacity, act_dim), dtype=jp.float32),
            rewards=jp.zeros((capacity,), dtype=jp.float32),
            next_observations=jp.zeros((capacity, obs_dim), dtype=jp.float32),
            dones=jp.zeros((capacity,), dtype=jp.bool_),
            ptr=jp.array(0, dtype=jp.int32),
            size=jp.array(0, dtype=jp.int32),
            capacity=capacity,
            add=jax.jit(_add),
            sample=jax.jit(_sample, static_argnames=("batch_size",)),
            reset=jax.jit(_reset),
        )


# region Rollout
@functools.partial(jax.jit, static_argnames=("args", "random_action"))
def collect_rollout(
    envs: struct.PyTreeNode,
    args: Args,
    agent: TD3Agent,
    obs: Array,
    done: Array,
    sum_rewards: Array,
    key: Array,
    buffer: ReplayBuffer,
    random_action: bool,
) -> tuple[struct.PyTreeNode, ReplayBuffer, Array, Array, Array, Array, RolloutData]:
    """Collect rollout data and add to buffer.

    Returns:
        (envs, buffer, next_obs, next_done, sum_rewards, key, rollout_data)
        rollout_data: RolloutData with sum_rewards for logging completed episodes
    """
    n_steps = args.rollout_steps if not random_action else args.learning_starts // args.num_envs

    def step_once(carry, _):
        env, key, sum_rewards, obs, done = carry

        # Action selection
        if random_action:
            action, key = agent.get_random_action(args.num_envs, key)
        else:
            action, key = agent.get_action_sample(
                agent.actor_state.params, obs, key, std=args.exploration_noise
            )

        # Step environment
        env, (next_obs, reward, terminated, truncated, _) = env.step(env, action)
        next_done = terminated | truncated

        # Track episode rewards (accumulate, then reset for done envs)
        sum_rewards = sum_rewards + reward
        next_sum_rewards = jp.where(next_done, 0.0, sum_rewards)

        data = RolloutData(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            dones=next_done,
            sum_rewards=sum_rewards,  # Store before reset for logging
        )

        return (env, key, next_sum_rewards, next_obs, next_done), data

    (envs, key, sum_rewards, obs, done), rollout_data = lax.scan(
        step_once, (envs, key, sum_rewards, obs, done), length=n_steps
    )

    # Add to buffer
    buffer = buffer.add(buffer, rollout_data)

    return envs, buffer, obs, done, sum_rewards, key, rollout_data


# region update_policy
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(
    args: Args, agent: TD3Agent, buffer: ReplayBuffer, key: Array
) -> tuple[TD3Agent, Array, Array, Array]:
    """Perform M TD3 policy updates with delayed actor updates.

    Returns:
        (agent, mean_critic_loss, mean_actor_loss, key)
    """

    def update_epoch(carry, idx):
        agent, key = carry

        # Sample batch
        key, sample_key = jax.random.split(key)
        batch = buffer.sample(buffer, args.batch_size, sample_key)

        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_observations
        dones = batch.dones

        # === Critic Update ===
        # Target policy smoothing
        key, noise_key = jax.random.split(key)
        target_actions = agent.get_action_mean(agent.target_actor_params, next_obs)
        noise = jp.clip(
            jax.random.normal(noise_key, target_actions.shape) * args.policy_noise, -args.noise_clip, args.noise_clip
        )
        target_actions = jp.clip(target_actions + noise, -1.0, 1.0)

        # Compute target Q
        target_q1 = agent.get_q(agent.target_critic1_params, next_obs, target_actions)
        target_q2 = agent.get_q(agent.target_critic2_params, next_obs, target_actions)
        target_q = rewards + args.gamma * (1.0 - dones.astype(jp.float32)) * jp.minimum(target_q1, target_q2)

        def critic_loss_fn(params, target_q=target_q):
            q = agent.get_q(params, obs, actions)
            return jp.mean((q - target_q) ** 2)

        c1_loss, c1_grads = jax.value_and_grad(critic_loss_fn)(agent.critic1_state.params)
        c2_loss, c2_grads = jax.value_and_grad(critic_loss_fn)(agent.critic2_state.params)

        critic1_state = agent.critic1_state.apply_gradients(grads=c1_grads)
        critic2_state = agent.critic2_state.apply_gradients(grads=c2_grads)
        agent = agent.replace(critic1_state=critic1_state, critic2_state=critic2_state)
        critic_loss = (c1_loss + c2_loss) / 2.0

        # === Actor Update (delayed) ===
        def do_actor_update(ag):
            def actor_loss_fn(params):
                a = ag.get_action_mean(params, obs)
                return -jp.mean(ag.get_q(ag.critic1_state.params, obs, a))

            a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(ag.actor_state.params)
            actor_state = ag.actor_state.apply_gradients(grads=a_grads)

            # Soft update targets
            ag = ag.replace(
                actor_state=actor_state,
                target_actor_params=optax.incremental_update(actor_state.params, ag.target_actor_params, args.tau),
                target_critic1_params=optax.incremental_update(
                    ag.critic1_state.params, ag.target_critic1_params, args.tau
                ),
                target_critic2_params=optax.incremental_update(
                    ag.critic2_state.params, ag.target_critic2_params, args.tau
                ),
            )
            return ag, a_loss

        def skip_actor_update(ag):
            return ag, jp.array(0.0)

        agent, actor_loss = lax.cond(idx % args.policy_delay == 0, do_actor_update, skip_actor_update, agent)

        return (agent, key), (critic_loss, actor_loss)

    (agent, key), (critic_losses, actor_losses) = lax.scan(
        update_epoch, (agent, key), jp.arange(args.updates_epochs), unroll=args.policy_delay
    )

    mean_critic_loss = jp.mean(critic_losses)
    mean_actor_loss = jp.mean(actor_losses) * args.policy_delay

    return agent, mean_critic_loss, mean_actor_loss, key


# region train_td3
def train_td3(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> tuple:
    """Train TD3 agent.

    Structure:
    1. Warmup JIT compilation
    2. Pre-exploration with random actions
    3. Training loop: collect_rollout -> update_policy
    """
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))

    setup_start = time.time()
    key = jax.random.PRNGKey(args.seed)
    print(f"Training on device: {jax_device}")

    # Create environments
    envs = make_jitted_envs(args.num_envs, jax_device, args)
    print(envs.single_action_space.shape)

    # Create agent
    key, init_key = jax.random.split(key)
    agent = TD3Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        actor_obs_dim=envs.single_observation_space.shape[0] - 4,  # last 4 obs are privileged critic obs
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )

    # Create buffer
    buffer = ReplayBuffer.create(
        capacity=args.buffer_size,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
    )
    buffer = jax.device_put(buffer)
    print(f"Setup took {time.time() - setup_start:.2f}s")

    # Reset environments
    envs, (obs, _) = envs.reset(envs, seed=args.seed)
    done = jp.zeros((args.num_envs,), dtype=jp.bool_)
    sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)

    print(f"Pre-exploration: {args.rollout_batchsize} rollouts ({args.learning_starts} timesteps)")
    print(f"Training: {args.num_iterations} iterations (N={args.rollout_steps}, M={args.updates_epochs})")

    # JIT warmup
    print("JIT compiling... ", end="", flush=True)
    warmup_start = time.time()

    # Warmup collect_rollout
    _, _, _, _, sum_rewards, key, _ = collect_rollout(
        envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=True
    )
    _, _, _, _, sum_rewards, key, _ = collect_rollout(
        envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=False
    )

    # Warmup update_policy
    agent, _, _, key = update_policy(args, agent, buffer, key)
    agent.actor_state.params["params"]["Dense_0"]["kernel"].block_until_ready()
    print(f"done ({time.time() - warmup_start:.2f}s)")

    # Reset for actual training
    envs, (obs, _) = envs.reset(envs, seed=args.seed)
    done = jp.zeros((args.num_envs,), dtype=jp.bool_)
    sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)
    buffer = ReplayBuffer.create(
        capacity=args.buffer_size,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
    )
    buffer = jax.device_put(buffer)

    reward_history = []
    train_start = time.time()

    # === Pre-exploration ===
    print("Phase 1: Pre-exploration...")

    envs, buffer, obs, done, sum_rewards, key, rollout_data = collect_rollout(
        envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=True
    )

    # Log completed episodes (like PPO)
    for step_idx in range(args.rollout_steps):
        step_dones = rollout_data.dones[step_idx]
        if jp.any(step_dones):
            step_rewards = rollout_data.sum_rewards[step_idx]
            mean_reward = float(jp.mean(step_rewards[step_dones]))
            reward_history.append(mean_reward)

    if len(reward_history) > 0:
        print(f"Pre-exploration: {len(reward_history)} episodes, mean reward: {np.mean(reward_history):.2f}")
        if wandb_enabled:
            wandb.log({"train/episode_reward": np.mean(reward_history)}, step=args.learning_starts)

    # === Training ===
    print("Phase 2: Training...")

    for i in range(args.num_iterations):
        # Collect rollout
        envs, buffer, obs, done, sum_rewards, key, rollout_data = collect_rollout(
            envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=False
        )

        # Update policy
        agent, critic_loss, actor_loss, key = update_policy(args, agent, buffer, key)

        # Logging
        global_step = args.learning_starts + (i + 1) * args.rollout_batchsize

        # Log completed episodes (like PPO)
        for step_idx in range(args.rollout_steps):
            step_dones = rollout_data.dones[step_idx]
            if jp.any(step_dones):
                step_rewards = rollout_data.sum_rewards[step_idx]
                mean_reward = float(jp.mean(step_rewards[step_dones]))
                reward_history.append(mean_reward)
                if wandb_enabled:
                    wandb.log({"train/episode_reward": mean_reward}, step=global_step)

        if i % 10 == 0:
            if wandb_enabled:
                wandb.log(
                    {"losses/critic_loss": float(critic_loss), "losses/actor_loss": float(actor_loss)},
                    step=global_step,
                )

        # Progress print every 100 iterations
        if i % 100 == 0:
            elapsed = time.time() - train_start
            sps = global_step / elapsed
            print(f"Iter {i}/{args.num_iterations}, step {global_step}, SPS: {sps:.0f}")

    # Final sync
    agent.actor_state.params["params"]["Dense_0"]["kernel"].block_until_ready()
    training_time = time.time() - train_start
    sps = args.total_timesteps / training_time
    print(f"Training {args.total_timesteps} steps took {training_time:.2f}s ({sps:.0f} SPS)")

    if wandb_enabled:
        wandb.log({"charts/SPS": sps}, step=args.total_timesteps)

    # Save model
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "actor": agent.actor_state.params,
            "critic1": agent.critic1_state.params,
            "critic2": agent.critic2_state.params,
        }
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Model saved to {model_path}")

    return reward_history, training_time


# region Evaluate
def evaluate_td3(
    args: Args, n_eval: int, model_path: Path, render: bool, plot: bool = True
) -> tuple[Any, float, list, list]:
    """Evaluate trained TD3 policy."""
    eval_env = make_jitted_envs(num_envs=1, jax_device=args.jax_device, args=args)
    eval_env = RecordData.create(eval_env)

    agent = TD3Agent.create(
        key=jax.random.PRNGKey(0),
        obs_dim=eval_env.single_observation_space.shape[0],
        act_dim=eval_env.single_action_space.shape[0],
        actor_obs_dim=eval_env.single_observation_space.shape[0] - 4,  # last 4 obs are privileged critic obs
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    with open(model_path, "rb") as f:
        params = pickle.load(f)
    agent = agent.replace(actor_state=agent.actor_state.replace(params=params["actor"]))

    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed

    for ep in range(n_eval):
        eval_env, (obs, _) = eval_env.reset(eval_env, seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = agent.get_action_mean(agent.actor_state.params, obs)
            eval_env, (obs, reward, terminated, truncated, _) = eval_env.step(eval_env, action)

            if render:
                fps = 50
                if ((steps * fps) % eval_env.unwrapped.freq) < fps:
                    eval_env.render()

            done = bool(terminated | truncated)
            episode_reward += float(np.asarray(reward).item())
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep + 1}/{n_eval}: reward={episode_reward:.2f}, length={steps}")

    fig = eval_env.plot_eval(save_path=f"{args.exp_name}_eval_plot.png") if plot else None
    rmse_pos = eval_env.calc_rmse()

    print(f"Eval Mean Reward: {np.mean(episode_rewards):.2f}, RMSE: {rmse_pos * 1000:.3f} mm")
    eval_env.close()

    return fig, rmse_pos, episode_rewards, episode_lengths


# region Main
def main(
    wandb_enabled: bool = True, train: bool = True, n_eval: int = 1, render: bool = False, plot: bool = True
):
    """Main entry point."""
    args = Args.create()
    model_path = Path(__file__).parents[2] / f"saves/{args.exp_name}_model.ckpt"
    jax_device = args.jax_device

    if train:
        reward_history, training_time = train_td3(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_td3(args, n_eval, model_path, render, plot)
        if wandb_enabled and train:
            logs = {
                "eval/mean_rewards": np.mean(episode_rewards),
                "eval/mean_steps": np.mean(episode_lengths),
            }
            if fig is not None:
                logs["eval/eval_plot"] = wandb.Image(fig)
            if rmse_pos is not None:
                logs["eval/pos_rmse_mm"] = rmse_pos * 1000
            wandb.log(logs)
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
