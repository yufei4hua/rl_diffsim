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
    AngleReward,
    FlattenJaxObservation,
    NormalizeActions,
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

    # Algorithm specific arguments
    total_timesteps: int = 300_000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 24
    """N: number of steps per env per rollout (macro-iteration)"""
    updates_epochs: int = 48
    """M: number of gradient updates per rollout (controls G/U ratio)"""
    buffer_size: int = 131_072  # 262_144
    """replay buffer capacity"""
    batch_size: int = 512
    """minibatch size for updates"""
    learning_starts: int = 65_536
    """timesteps before training starts (random exploration)"""
    actor_lr: float = 0.0017489005211975934
    """the learning rate of the actor optimizer"""
    critic_lr: float = 0.0011526032458357583
    """the learning rate of the critic optimizer"""
    gamma: float = 0.982896493822861
    """the discount factor gamma"""
    tau: float = 0.19481974159090312
    """target network update rate (Polyak averaging)"""
    policy_delay: int = 2
    """update actor every N critic updates"""
    exploration_noise: float = 0.2
    """std of exploration noise during data collection"""
    policy_noise: float = 0.2
    """std of noise added to target policy (smoothing)"""
    noise_clip: float = 0.10550942818281409
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

    # Envs & Wrapper settings
    pos_min: Array = (-0.5, -0.5, 1.0)
    pos_max: Array = (0.5, 0.5, 2.0)
    goal_pmin: Array = (-0.0, -0.0, 1.5)
    goal_pmax: Array = (0.0, 0.0, 1.5)
    vel_min: float = -1.0
    vel_max: float = 1.0
    ang_vel_min: Array = (-1.0, -1.0, -1.0)
    ang_vel_max: Array = (1.0, 1.0, 1.0)
    num_last_actions: int = 1
    rpy_coef: float = 0.20506035936210104
    act_coefs: tuple = (0.07514871103816087,) * 4
    d_act_coefs: tuple = (0.10373663103609186,) * 4

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class with computed values."""
        args = Args(**kwargs)
        assert args.updates_epochs % args.policy_delay == 0, (
            f"updates_epochs ({args.updates_epochs}) must be divisible by policy_delay ({args.policy_delay})"
        )
        rollout_batchsize = int(args.num_steps * args.num_envs)
        num_iterations = (args.total_timesteps - args.learning_starts) // rollout_batchsize
        act_coefs = (args.act_coefs[0],) * 4  # make sure all four coefficients are the same
        d_act_coefs = (args.d_act_coefs[0],) * 4
        return replace(
            args,
            rollout_batchsize=rollout_batchsize,
            num_iterations=num_iterations,
            act_coefs=act_coefs,
            d_act_coefs=d_act_coefs,
        )


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
        max_episode_time=5.0,
        pos_min=jp.array(args.pos_min),
        pos_max=jp.array(args.pos_max),
        goal_pmin=jp.array(args.goal_pmin),
        goal_pmax=jp.array(args.goal_pmax),
        ang_vel_min=jp.array(args.ang_vel_min),
        ang_vel_max=jp.array(args.ang_vel_max),
        vel_min=args.vel_min,
        vel_max=args.vel_max,
    )

    env = NormalizeActions.create(env)
    env = AngleReward.create(env, rpy_coef=args.rpy_coef)
    env = ActionPenalty.create(
        env,
        num_actions=args.num_last_actions,
        init_last_actions=jp.array([[0.0, 0.0, 0.0, 0.0]]),
        hover_action=jp.array([0.25, 0.25, 0.25, 0.25]),
        act_coefs=args.act_coefs,
        d_act_coefs=args.d_act_coefs,
    )
    env = FlattenJaxObservation.create(env)

    return env


# region Utils
@struct.dataclass
class RolloutData:
    """Class for storing rollout data."""

    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    dones: Array
    sum_rewards: Array


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
    n_steps = args.num_steps if not random_action else args.learning_starts // args.num_envs

    def step_once(
        carry: tuple[struct.PyTreeNode, Array, Array, Array, Array], _: Any
    ) -> tuple[tuple[struct.PyTreeNode, Array, Array, Array, Array], RolloutData]:
        env, key, sum_rewards, obs, done = carry

        # Action selection
        if random_action:
            action, key = agent.get_random_action(args.num_envs, key)
            # action, key = agent.get_action_sample(
            #     agent.actor_state.params, obs, key, std=args.exploration_noise
            # ) # only sample policy distribution
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
    """Perform M TD3 policy updates with delayed actor updates."""
    # Pre-sample all batches
    sample_keys = jax.random.split(key, args.updates_epochs + 1)
    key = sample_keys[0]
    sample_keys = sample_keys[1:]

    def sample_batch(key: Array) -> RolloutData:
        return buffer.sample(buffer, args.batch_size, key)

    batches = jax.vmap(sample_batch)(sample_keys)

    # Update Batch Epochs (update_critic * policy_delay + update_actor * 1)
    def update_batch_epochs(
        carry: tuple[TD3Agent, Array], batch_offset: int
    ) -> tuple[tuple[TD3Agent, Array], tuple[Array, Array]]:
        agent, key = carry

        # Critic Update (*policy_delay)
        def update_critic(
            carry: tuple[TD3Agent, Array], local_idx: int
        ) -> tuple[tuple[TD3Agent, Array], Array]:
            agent, key = carry
            idx = batch_offset * args.policy_delay + local_idx

            obs = batches.observations[idx]
            actions = batches.actions[idx]
            rewards = batches.rewards[idx]
            next_obs = batches.next_observations[idx]
            dones = batches.dones[idx]

            # Target policy smoothing
            target_actions, key = agent.get_action_sample(
                agent.target_actor_params, next_obs, key, std=args.policy_noise, noise_clip=args.noise_clip
            )

            # Compute target Q
            target_q1 = agent.get_q(agent.target_critic1_params, next_obs, target_actions)
            target_q2 = agent.get_q(agent.target_critic2_params, next_obs, target_actions)
            target_q = rewards + args.gamma * (1.0 - dones.astype(jp.float32)) * jp.minimum(
                target_q1, target_q2
            )

            # Twin critic loss
            def critic_loss_fn(c1_params: dict[str, Any], c2_params: dict[str, Any]) -> Array:
                q1 = agent.get_q(c1_params, obs, actions)
                q2 = agent.get_q(c2_params, obs, actions)
                return jp.mean((q1 - target_q) ** 2) + jp.mean((q2 - target_q) ** 2)

            (critic_loss, (c1_grads, c2_grads)) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1))(
                agent.critic1_state.params, agent.critic2_state.params
            )

            critic1_state = agent.critic1_state.apply_gradients(grads=c1_grads)
            critic2_state = agent.critic2_state.apply_gradients(grads=c2_grads)
            agent = agent.replace(critic1_state=critic1_state, critic2_state=critic2_state)

            return (agent, key), critic_loss

        (agent, key), critic_losses = lax.scan(
            update_critic, (agent, key), jp.arange(args.policy_delay), unroll=args.policy_delay
        )  # unroll update_critic() loop

        # Actor Update (*1)
        def update_actor(agent: TD3Agent, obs: Array) -> tuple[TD3Agent, Array]:
            def actor_loss_fn(params: dict[str, Any]) -> Array:
                a = agent.get_action_mean(params, obs)
                return -jp.mean(agent.get_q(agent.critic1_state.params, obs, a))

            actor_loss, a_grads = jax.value_and_grad(actor_loss_fn)(agent.actor_state.params)
            actor_state = agent.actor_state.apply_gradients(grads=a_grads)

            # Soft update targets
            agent = agent.replace(
                actor_state=actor_state,
                target_actor_params=optax.incremental_update(
                    actor_state.params, agent.target_actor_params, args.tau
                ),
                target_critic1_params=optax.incremental_update(
                    agent.critic1_state.params, agent.target_critic1_params, args.tau
                ),
                target_critic2_params=optax.incremental_update(
                    agent.critic2_state.params, agent.target_critic2_params, args.tau
                ),
            )
            return agent, actor_loss

        last_idx = batch_offset * args.policy_delay + args.policy_delay - 1
        obs = batches.observations[last_idx]
        agent, actor_loss = update_actor(agent, obs)

        return (agent, key), (jp.mean(critic_losses), actor_loss)

    num_actor_updates = args.updates_epochs // args.policy_delay
    (agent, key), (critic_losses, actor_losses) = lax.scan(
        update_batch_epochs, (agent, key), jp.arange(num_actor_updates)
    )

    mean_critic_loss = jp.mean(critic_losses)
    mean_actor_loss = jp.mean(actor_losses)

    return agent, mean_critic_loss, mean_actor_loss, key


# region train_td3
def train_td3(args: Args, model_path: Path, jax_device: str, wandb_enabled: bool = False) -> tuple:
    """Train TD3 agent."""
    if wandb_enabled and wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))

    setup_start = time.time()
    key = jax.random.PRNGKey(args.seed)
    print(f"Training on device: {jax_device}")

    # Create environments
    envs = make_jitted_envs(args.num_envs, jax_device, args)

    # Create agent
    key, init_key = jax.random.split(key)
    agent = TD3Agent.create(
        key=init_key,
        obs_dim=envs.single_observation_space.shape[0],
        act_dim=envs.single_action_space.shape[0],
        # actor_obs_dim=envs.single_observation_space.shape[0] - 4,  # last 4 obs are privileged critic obs
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

    print(f"Pre-exploration: {args.learning_starts} timesteps")
    print(f"Training: {args.num_iterations} iterations (N={args.num_steps}, M={args.updates_epochs})")

    @jax.jit
    def train_iteration(carry: tuple, _) -> tuple[tuple, tuple]:
        envs, agent, buffer, obs, done, sum_rewards, key = carry

        # 1. Collect rollout
        envs, buffer, obs, done, sum_rewards, key, rollout_data = collect_rollout(
            envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=False
        )

        # 2. Update policy
        agent, critic_loss, actor_loss, key = update_policy(args, agent, buffer, key)

        return (envs, agent, buffer, obs, done, sum_rewards, key), (rollout_data, critic_loss, actor_loss)

    # JIT warmup (only what's needed for training)
    print("JIT compiling... ", end="", flush=True)
    warmup_start = time.time()
    envs, buffer, obs, done, sum_rewards, key, pre_rollout_data = collect_rollout(
        envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=True
    )
    (
        (envs, warmup_agent, buffer, obs, done, sum_rewards, key),
        (all_rollout_data, all_critic_loss, all_actor_loss),
    ) = lax.scan(train_iteration, (envs, agent, buffer, obs, done, sum_rewards, key), length=2)
    warmup_agent.actor_state.params["params"]["Dense_0"]["kernel"].block_until_ready()
    print(f"done ({time.time() - warmup_start:.2f}s)")

    # Training
    envs, (obs, _) = envs.reset(envs, seed=args.seed)
    done = jp.zeros((args.num_envs,), dtype=jp.bool_)
    sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)
    buffer = buffer.reset(buffer)

    train_start = time.time()
    print("Phase 1: Pre-exploration...")
    envs, buffer, obs, done, sum_rewards, key, pre_rollout_data = collect_rollout(
        envs, args, agent, obs, done, sum_rewards, key, buffer, random_action=True
    )

    print("Phase 2: Training...")
    # Run training loop using scan
    (
        (envs, agent, buffer, obs, done, sum_rewards, key),
        (all_rollout_data, all_critic_loss, all_actor_loss),
    ) = lax.scan(
        train_iteration, (envs, agent, buffer, obs, done, sum_rewards, key), jp.arange(args.num_iterations)
    )

    obs.block_until_ready()
    training_time = time.time() - train_start
    sps = args.total_timesteps / training_time
    print(f"Training {args.total_timesteps} steps took {training_time:.2f}s ({sps:.0f} SPS)")

    # === Post-training logging ===
    reward_history = []

    # Log pre-exploration episodes
    for step_idx in range(pre_rollout_data.dones.shape[0]):
        step_dones = pre_rollout_data.dones[step_idx]
        if jp.any(step_dones):
            step_rewards = pre_rollout_data.sum_rewards[step_idx]
            reward_history.append(float(jp.mean(step_rewards[step_dones])))

    if len(reward_history) > 0:
        print(f"Pre-exploration: {len(reward_history)} episodes, mean reward: {np.mean(reward_history):.2f}")
        if wandb_enabled:
            wandb.log({"train/episode_reward": np.mean(reward_history)}, step=args.learning_starts)

    # Log training iterations
    if wandb_enabled:
        for iter_idx in range(args.num_iterations):
            global_step = args.learning_starts + (iter_idx + 1) * args.rollout_batchsize

            # Log episode rewards
            for step_idx in range(args.num_steps):
                step_dones = all_rollout_data.dones[iter_idx, step_idx]
                if jp.any(step_dones):
                    step_rewards = all_rollout_data.sum_rewards[iter_idx, step_idx]
                    mean_reward = float(jp.mean(step_rewards[step_dones]))
                    reward_history.append(mean_reward)
                    wandb.log({"train/episode_reward": mean_reward}, step=global_step)

            # Log losses every 10 iterations
            if iter_idx % 10 == 0:
                wandb.log(
                    {
                        "losses/critic_loss": float(all_critic_loss[iter_idx]),
                        "losses/actor_loss": float(all_actor_loss[iter_idx]),
                    },
                    step=global_step,
                )

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
        # actor_obs_dim=eval_env.single_observation_space.shape[0] - 4,
        # last 4 obs are privileged critic obs
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
