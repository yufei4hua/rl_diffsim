"""TD3 training script for reach position task with rotor velocity control.

TD3 (Twin Delayed Deep Deterministic Policy Gradient) implementation for
quadrotor position control using direct RPM setpoints.
"""

import functools
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import flax.struct as struct
import jax
import jax.lax as lax
import jax.numpy as jp
import numpy as np
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
from rl_diffsim.td3.td3_agent import ReplayBuffer, TD3Agent, TD3RolloutData, update_actor, update_critics


# region Arguments
@dataclass(frozen=True)
class Args:
    """Configuration for TD3 training."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
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
    num_envs: int = 64
    """the number of parallel game environments"""
    buffer_size: int = 1_000_000
    """replay buffer capacity"""
    batch_size: int = 256
    """minibatch size for updates"""
    learning_starts: int = 25000
    """timesteps before training starts (random exploration)"""

    # TD3 hyperparameters
    actor_lr: float = 3e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 3e-4
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

    # Computed at runtime
    actor_obs_dim: int = 0
    critic_obs_dim: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class with computed values."""
        args = Args(**kwargs)
        return args


# region MakeEnvs
def make_jitted_envs(num_envs: int, jax_device: str, args: Args, reset_rotor: bool = True) -> ReachPosEnv:
    """Create wrapped environments for TD3 training.

    Wrapper chain: NormalizeActions -> ActionPenalty -> PrivilegedCriticObs -> FlattenJaxObservation
    """
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

    # 1. Normalize actions to [-1, 1]
    env = NormalizeActions.create(env)

    # 2. Add action history for motor delay compensation
    hover_action = jp.array([args.C_rab] * 4, dtype=jp.float32)
    env = ActionPenalty.create(
        env,
        num_actions=args.num_actions,
        init_last_actions=None,
        hover_action=hover_action,
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )

    # 3. Add privileged critic observations and TD3 reward
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

    # 4. Flatten dict observations to array
    env = FlattenJaxObservation.create(env)

    return env


# region Train


@struct.dataclass
class TD3CarryState:
    """State carried between scan iterations."""

    obs: Array
    critic_obs: Array
    episode_rewards: Array
    episode_lengths: Array
    total_updates: Array
    global_step: Array
    key: Array


def train_td3(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> tuple:
    """Train TD3 agent using jax.lax.scan for the training loop.

    Returns:
        Tuple of (reward_history, training_time)
    """
    # Setup
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    reward_history = []

    setup_start_time = time.time()
    key = jax.random.PRNGKey(args.seed)
    print(f"Training on device: {jax_device}")

    # Create environments
    envs = make_jitted_envs(args.num_envs, jax_device, args)

    # Get observation dimensions
    actor_obs_dim = envs.single_observation_space.shape[0]
    # Critic obs = actor obs + rotor_vel (4D)
    critic_obs_dim = actor_obs_dim + 4
    act_dim = envs.single_action_space.shape[0]
    print(f"Actor obs dim: {actor_obs_dim}, Critic obs dim: {critic_obs_dim}, Action dim: {act_dim}")

    # Create agent
    init_key, key = jax.random.split(key)
    agent = TD3Agent.create(
        key=init_key,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )

    # Create replay buffer
    buffer = ReplayBuffer.create(
        capacity=args.buffer_size, actor_obs_dim=actor_obs_dim, critic_obs_dim=critic_obs_dim, act_dim=act_dim
    )
    print(f"Setup took {time.time() - setup_start_time:.2f}s")

    # Reset environments
    envs, (obs, info) = envs.reset(envs, seed=args.seed)
    critic_obs = info["critic_obs"]

    # Define the TD3 step function for scan
    @functools.partial(jax.jit, static_argnames=("args",))
    def td3_step(
        carry: tuple[struct.PyTreeNode, TD3Agent, ReplayBuffer, TD3CarryState], _: int, args: Args
    ) -> tuple[tuple, TD3RolloutData]:
        """Single TD3 training step for use with jax.lax.scan."""
        env, agent, buffer, state = carry

        # 1. Select action (random or policy-based)
        def random_action(inputs):
            _, key = inputs
            action_key, new_key = jax.random.split(key)
            action = jax.random.uniform(action_key, (args.num_envs, act_dim), minval=-1.0, maxval=1.0)
            return action, new_key

        def policy_action(inputs):
            obs, key = inputs
            return agent.get_action_sample(agent.actor_state.params, obs, key, std=args.exploration_noise)

        action, key = lax.cond(
            state.global_step < args.learning_starts, random_action, policy_action, (state.obs, state.key)
        )

        # 2. Step environment
        env, (next_obs, reward, terminated, truncated, env_info) = env.step(env, action)
        next_critic_obs = env_info["critic_obs"]
        done = terminated | truncated

        # 3. Store transition in replay buffer
        buffer = buffer.add(
            buffer,
            actor_obs=state.obs,
            critic_obs=state.critic_obs,
            action=action,
            reward=reward,
            next_actor_obs=next_obs,
            next_critic_obs=next_critic_obs,
            done=done,
        )

        # 4. Update episode tracking
        new_episode_rewards = state.episode_rewards + reward
        new_episode_lengths = state.episode_lengths + 1

        # Store values before reset for logging
        episode_rewards_out = new_episode_rewards
        episode_lengths_out = new_episode_lengths

        # Reset episode stats for done environments
        new_episode_rewards = jp.where(done, 0.0, new_episode_rewards)
        new_episode_lengths = jp.where(done, 0, new_episode_lengths)

        # 5. Sample batch and maybe update networks
        sample_key, key = jax.random.split(key)
        batch = buffer.sample(buffer, args.batch_size, sample_key)

        def do_update(inputs):
            agent, batch, key, total_updates = inputs

            # Update critics
            agent, critic_loss, key = update_critics(
                agent, batch, args.gamma, args.policy_noise, args.noise_clip, key
            )
            new_total_updates = total_updates + 1

            # Conditionally update actor based on policy_delay
            def do_actor_update(agent_batch):
                ag, ba = agent_batch
                return update_actor(ag, ba, args.tau)

            def skip_actor_update(agent_batch):
                ag, _ = agent_batch
                return ag, jp.array(0.0)

            agent, actor_loss = lax.cond(
                new_total_updates % args.policy_delay == 0, do_actor_update, skip_actor_update, (agent, batch)
            )

            return agent, critic_loss, actor_loss, new_total_updates, key, jp.array(True)

        def skip_update(inputs):
            agent, _, key, total_updates = inputs
            return (
                agent,
                jp.array(0.0),  # critic_loss
                jp.array(0.0),  # actor_loss
                total_updates,
                key,
                jp.array(False),  # did_update
            )

        agent, critic_loss, actor_loss, total_updates, key, did_update = lax.cond(
            state.global_step >= args.learning_starts,
            do_update,
            skip_update,
            (agent, batch, key, state.total_updates),
        )

        # 6. Update global step
        new_global_step = state.global_step + args.num_envs

        # 7. Build new carry state
        new_state = TD3CarryState(
            obs=next_obs,
            critic_obs=next_critic_obs,
            episode_rewards=new_episode_rewards,
            episode_lengths=new_episode_lengths,
            total_updates=total_updates,
            global_step=new_global_step,
            key=key,
        )

        # 8. Build rollout data for logging
        rollout_data = TD3RolloutData(
            rewards=reward,
            dones=done,
            episode_rewards=episode_rewards_out,
            episode_lengths=episode_lengths_out,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            did_update=did_update,
        )

        return (env, agent, buffer, new_state), rollout_data

    # Initialize carry state
    initial_state = TD3CarryState(
        obs=obs,
        critic_obs=critic_obs,
        episode_rewards=jp.zeros((args.num_envs,)),
        episode_lengths=jp.zeros((args.num_envs,), dtype=jp.int32),
        total_updates=jp.array(0, dtype=jp.int32),
        global_step=jp.array(0, dtype=jp.int32),
        key=key,
    )
    initial_carry = (envs, agent, buffer, initial_state)

    # JIT warmup
    print("JIT compiling... ", end="", flush=True)
    warmup_start = time.time()

    step_fn = functools.partial(td3_step, args=args)

    # Warmup scan (small number of steps)
    warmup_carry, _ = lax.scan(step_fn, initial_carry, jp.arange(10))
    warmup_carry[2].size.block_until_ready()  # buffer.size
    print(f"done ({time.time() - warmup_start:.2f}s)")

    # Main training loop with chunked scan for memory efficiency
    train_start_time = time.time()
    num_steps = args.total_timesteps // args.num_envs
    chunk_size = min(10000, num_steps)  # Process in chunks to avoid memory issues
    num_chunks = (num_steps + chunk_size - 1) // chunk_size

    carry = initial_carry
    for chunk_idx in range(num_chunks):
        steps_in_chunk = min(chunk_size, num_steps - chunk_idx * chunk_size)
        if steps_in_chunk <= 0:
            break

        # Run chunk with scan
        carry, chunk_data = lax.scan(step_fn, carry, jp.arange(steps_in_chunk))

        # Block until this chunk completes
        carry[2].size.block_until_ready()

        # Post-chunk logging
        for step_idx in range(steps_in_chunk):
            global_step = (chunk_idx * chunk_size + step_idx) * args.num_envs

            dones = chunk_data.dones[step_idx]
            episode_rewards_step = chunk_data.episode_rewards[step_idx]
            episode_lengths_step = chunk_data.episode_lengths[step_idx]

            # Log completed episodes
            if jp.any(dones):
                done_mask = dones
                mean_reward = jp.mean(episode_rewards_step[done_mask])
                mean_length = jp.mean(episode_lengths_step[done_mask].astype(jp.float32))
                reward_history.append(float(mean_reward))

                if wandb_enabled:
                    wandb.log(
                        {
                            "train/episode_reward": float(mean_reward),
                            "train/episode_length": float(mean_length),
                        },
                        step=global_step,
                    )

            # Log training losses
            if chunk_data.did_update[step_idx] and step_idx % 100 == 0:
                if wandb_enabled:
                    wandb.log(
                        {
                            "losses/critic_loss": float(chunk_data.critic_loss[step_idx]),
                            "losses/actor_loss": float(chunk_data.actor_loss[step_idx]),
                        },
                        step=global_step,
                    )

        # Progress logging
        elapsed = time.time() - train_start_time
        current_step = (chunk_idx + 1) * chunk_size * args.num_envs
        current_step = min(current_step, args.total_timesteps)
        sps = current_step / elapsed if elapsed > 0 else 0
        print(
            f"Step {current_step}/{args.total_timesteps} ({100 * current_step / args.total_timesteps:.1f}%) "
            f"- SPS: {sps:.0f}"
        )
        if wandb_enabled:
            wandb.log({"charts/SPS": sps}, step=current_step)

    training_time = time.time() - train_start_time
    print(
        f"Training {args.total_timesteps} steps took {training_time:.2f}s "
        f"({args.total_timesteps / training_time:.0f} SPS)"
    )

    # Extract final agent from carry
    _, final_agent, _, _ = carry

    # Save model
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "actor": final_agent.actor_state.params,
            "critic1": final_agent.critic1_state.params,
            "critic2": final_agent.critic2_state.params,
        }
        with open(model_path, "wb") as f:
            pickle.dump(params, f)
        print(f"Model saved to {model_path}")

    return reward_history, training_time


# region Evaluate
def evaluate_td3(
    args: Args, n_eval: int, model_path: Path, render: bool, plot: bool = True
) -> tuple[Any, float, list, list]:
    """Evaluate trained TD3 policy.

    Returns:
        Tuple of (figure, rmse_pos, episode_rewards, episode_lengths)
    """
    # Create evaluation environment
    eval_env = make_jitted_envs(num_envs=1, jax_device=args.jax_device, args=args)
    eval_env = RecordData.create(eval_env)

    # Get dimensions
    actor_obs_dim = eval_env.single_observation_space.shape[0]
    critic_obs_dim = actor_obs_dim + 4
    act_dim = eval_env.single_action_space.shape[0]

    # Create agent and load parameters
    agent = TD3Agent.create(
        key=jax.random.PRNGKey(0),
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    with open(model_path, "rb") as f:
        params = pickle.load(f)
    agent = agent.replace(actor_state=agent.actor_state.replace(params=params["actor"]))

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed

    for ep in range(n_eval):
        eval_env, (obs, info) = eval_env.reset(eval_env, seed=(ep_seed := ep_seed + 1))
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            # Deterministic action (no noise)
            action = agent.get_action_mean(agent.actor_state.params, obs)
            eval_env, (obs, reward, terminated, truncated, info) = eval_env.step(eval_env, action)

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

    # Generate evaluation plot
    fig = eval_env.plot_eval(save_path=f"{args.exp_name}_eval_plot.png") if plot else None
    rmse_pos = eval_env.calc_rmse()

    print(f"Eval Mean Reward: {np.mean(episode_rewards):.2f}, RMSE: {rmse_pos * 1000:.3f} mm")

    eval_env.close()

    return fig, rmse_pos, episode_rewards, episode_lengths


# region Main
def main(
    wandb_enabled: bool = True, train: bool = True, n_eval: int = 1, render: bool = False, plot: bool = True
):
    """Main entry point.

    Args:
        wandb_enabled: Log metrics to wandb
        train: Run training
        n_eval: Number of evaluation episodes (0 to skip)
        render: Render environment during evaluation
        plot: Generate evaluation plots
    """
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
