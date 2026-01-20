"""A PPO implementation based on Flax."""

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
from ml_collections import ConfigDict

import wandb
from rl_diffsim.envs.drone_race_env import DroneRaceEnv
from rl_diffsim.envs.race_utils import load_config
from rl_diffsim.envs.wrappers import ActionPenalty, FlattenJaxObservation, NormalizeActions, ZeroYaw
from rl_diffsim.envs.wrappers_race import RaceWrapper, RecordRaceData
from rl_diffsim.ppo.ppo_agent import Agent


# region Arguments
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "gpu"
    """environment device"""
    exp_name: str = "ppo_race"
    """the name of the experiment"""
    wandb_project_name: str = "rl-ppo-race"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 15_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    actor_lr: float = 8e-4
    """the learning rate of the actor optimizer"""
    critic_lr: float = 2.5e-3
    """the learning rate of the critic optimizer"""
    gamma: float = 0.95
    """the discount factor gamma"""
    gae_lambda: float = 0.81
    """the lambda for the general advantage estimation"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.08
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 1.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
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
    min_vel: float = 0.4
    max_vel: float = 1.0
    cont_floor_safe_dist: float = 0.05
    cont_gate_safe_dist: float = 0.17
    cont_obst_safe_dist: float = 0.18
    gate_size: tuple = (0.6, 0.25)
    gate_pos_coef: float = 0.0
    gate_vel_coef: tuple = (2.7, 0.0)
    gate_pass_coef: tuple = (5.0, 15.0)
    contact_coef: tuple = (20.0, 64.0)
    act_coefs: tuple = (0.2, 0.2, 0.0, 0.1)
    d_act_coefs: tuple = (1.0, 1.0, 0.0, 0.4)
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
    total_timesteps: int = 0,
    coefs: dict = {},
    config: ConfigDict = ConfigDict(),
    check_contacts: bool = True,
) -> DroneRaceEnv:
    """Make environments for training RL policy."""
    env: DroneRaceEnv = DroneRaceEnv.create(
        num_envs=num_envs, device=jax_device, check_contacts=check_contacts, **config
    )

    env = RaceWrapper.create(
        env,
        gate_pos_coef=coefs.get("gate_pos_coef", 0.0),
        gate_vel_coef=coefs.get("gate_vel_coef", 0.0),
        gate_pass_coef=coefs.get("gate_pass_coef", 0.0),
        min_vel=coefs.get("min_vel", 0.0),
        max_vel=coefs.get("max_vel", 0.0),
        cont_floor_safe_dist=coefs.get("cont_floor_safe_dist", 0.0),
        cont_gate_safe_dist=coefs.get("cont_gate_safe_dist", 0.0),
        cont_obst_safe_dist=coefs.get("cont_obst_safe_dist", 0.0),
        contact_coef=coefs.get("contact_coef", 0.0),
        gate_size=coefs.get("gate_size", 0.45),
        total_timesteps=total_timesteps,
    )
    env = NormalizeActions.create(env)
    env = ZeroYaw.create(env)
    env = ActionPenalty.create(
        env,
        num_actions=1,
        init_last_actions=(0.0,) * 4,
        act_coefs=coefs.get("act_coefs", (0.0,) * 4),
        d_act_coefs=coefs.get("d_act_coefs", (0.0,) * 4),
    )
    env = FlattenJaxObservation.create(env)
    return env


# region Utils
@flax.struct.dataclass
class RolloutData:
    """Class for storing rollout data."""

    observations: Array
    actions: Array
    logprobs: Array
    rewards: Array
    dones: Array
    values: Array
    sum_rewards: Array
    advantages: Array
    returns: Array
    gates_passed: Array


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
) -> tuple[struct.PyTreeNode, RolloutData, Array, Array, Array, Array]:
    """Collect a rollout of length args.num_steps. Returns data dict and next obs/done/key."""

    # loop over args.num_steps
    def step_once(carry: tuple, _) -> tuple[tuple, tuple]:
        env, key, sum_rewards, obs, dones = carry
        # 1. get action from policy
        (action, logprob, entropy), key = agent.get_action_sample(
            agent.actor_states.params, obs, key
        )
        value = agent.get_value(agent.critic_states.params, obs)

        # 2. step environment
        env, (next_obs, reward, terminations, truncations, info) = env.step(env, action)
        gates_passed = env.unwrapped.race_data.target_gate[:, 0]
        sum_rewards = sum_rewards + reward
        next_sum_rewards = jp.where(dones, 0.0, sum_rewards)
        next_dones = terminations | truncations

        return (env, key, next_sum_rewards, next_obs, next_dones), RolloutData(
            observations=obs,
            actions=action,
            logprobs=logprob,
            rewards=reward,
            dones=dones,
            values=value,
            sum_rewards=sum_rewards,
            advantages=jp.zeros_like(reward),
            returns=jp.zeros_like(reward),
            gates_passed=gates_passed,
        )

    (envs, key, sum_rewards, next_obs, next_done), rollout_data = jax.lax.scan(
        step_once, (envs, key, sum_rewards, next_obs, next_done), length=args.num_steps
    )

    return envs, rollout_data, next_obs, next_done, sum_rewards, key


# region GAE
@functools.partial(jax.jit, static_argnames=("args",))
def compute_gae(
    args: Args, agent: Agent, next_obs: Array, next_done: Array, data: RolloutData
) -> RolloutData:
    """Compute GAE advantages and returns."""
    last_value = agent.get_value(agent.critic_states.params, next_obs)

    advantages = jp.zeros((args.num_envs,))
    dones = jp.concatenate([data.dones, next_done[None, :]], axis=0)
    values = jp.concatenate([data.values, last_value[None, :]], axis=0)

    def compute_gae_once(
        carry: Array, inp: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Compute one step of GAE in scan."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + args.gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    _, advantages = jax.lax.scan(
        compute_gae_once,
        advantages,
        (dones[1:], values[1:], values[:-1], data.rewards),
        reverse=True,
    )
    data = data.replace(advantages=advantages, returns=advantages + data.values)
    return data


# region PG
@functools.partial(jax.jit, static_argnames=("args",))
def update_policy(args: Args, agent: Agent, data: RolloutData, key: Array) -> float:
    """Performs PPO updates. Returns losses."""

    # loss function
    def loss_fn(
        actor_params: dict,
        critic_params: dict,
        obs_mb: Array,
        acts_mb: Array,
        old_logprobs_mb: Array,
        adv_mb: Array,
        ret_mb: Array,
        val_mb: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
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
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (g_actor, g_critic) = grad_fn(
                agent.actor_states.params,
                agent.critic_states.params,
                minibatch.observations,
                minibatch.actions,
                minibatch.logprobs,
                minibatch.advantages,
                minibatch.returns,
                minibatch.values,
            )
            return agent.replace(
                actor_states=agent.actor_states.apply_gradients(grads=g_actor),
                critic_states=agent.critic_states.apply_gradients(grads=g_critic),
            ), (pg_loss, v_loss, entropy_loss, approx_kl)

        agent, (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_minibatch, agent, minibatches
        )

        return (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl)

    (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        update_epoch, (agent, key), (), length=args.update_epochs
    )

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
    sum_rewards_hist = []

    setup_start_time = time.time()
    key = jax.random.PRNGKey(args.seed)
    print("Training on device:", jax_device)

    # make envs
    r_coefs = {
        "gate_pos_coef": args.gate_pos_coef,
        "gate_vel_coef": args.gate_vel_coef,
        "gate_pass_coef": args.gate_pass_coef,
        "min_vel": args.min_vel,
        "max_vel": args.max_vel,
        "cont_floor_safe_dist": args.cont_floor_safe_dist,
        "cont_gate_safe_dist": args.cont_gate_safe_dist,
        "cont_obst_safe_dist": args.cont_obst_safe_dist,
        "contact_coef": args.contact_coef,
        "gate_size": args.gate_size,
        "act_coefs": args.act_coefs,
        "d_act_coefs": args.d_act_coefs,
    }
    config = load_config(Path(__file__).parents[2] / "scripts/config_race.toml")
    envs = make_jitted_envs(
        num_envs=args.num_envs,
        jax_device=jax_device,
        total_timesteps=args.total_timesteps,
        coefs=r_coefs,
        config=config.env,
        check_contacts=False,
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
    # for _ in range(2):
    #     _, data, next_obs, next_done, _, _ = collect_rollout(
    #         envs=envs,
    #         args=args,
    #         agent=agent,
    #         next_obs=next_obs,
    #         next_done=jp.zeros(args.num_envs, dtype=bool),
    #         sum_rewards=jp.zeros((args.num_envs,)),
    #         key=key,
    #     )
    #     data = compute_gae(args, agent, next_obs, next_done, data)
    #     (_, _), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var) = update_policy(
    #         args, agent, data, key
    #     )
    print("JAX warmup took {:.5f} s".format(time.time() - start_warmup_time))

    # start the game
    train_start_time = time.time()
    # envs, (next_obs, _) = envs.reset(envs, seed=args.seed)
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))

    # Define single iteration function for scan
    def train_iteration(carry: tuple, _) -> tuple[tuple, tuple]:
        envs, agent, key, next_obs, next_done, sum_rewards = carry

        # 1. collect rollouts
        envs, data, next_obs, next_done, sum_rewards, key = collect_rollout(
            envs=envs,
            args=args,
            agent=agent,
            next_obs=next_obs,
            next_done=next_done,
            sum_rewards=sum_rewards,
            key=key,
        )

        # 2. compute GAE
        data = compute_gae(args, agent, next_obs, next_done, data)

        # 3. update policy
        (agent, key), (pg_loss, v_loss, entropy_loss, approx_kl, explained_var) = update_policy(
            args, agent, data, key
        )

        # Return carry and metrics for logging
        return (envs, agent, key, next_obs, next_done, sum_rewards), (
            data,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            explained_var,
        )

    # Run training loop using scan
    (
        (envs, agent, key, next_obs, next_done, sum_rewards),
        (all_data, all_pg_loss, all_v_loss, all_entropy_loss, all_approx_kl, all_explained_var),
    ) = jax.lax.scan(
        train_iteration,
        (envs, agent, key, next_obs, next_done, sum_rewards),
        jp.arange(args.num_iterations),
    )

    next_obs.block_until_ready()
    training_time = time.time() - train_start_time
    total_steps = args.num_iterations * args.batch_size
    print(f"Training for {total_steps} steps took {training_time:.2f} seconds.")

    # Post-training logging
    if wandb_enabled:
        # Log aggregated metrics
        for iter_idx in range(args.num_iterations):
            global_step = iter_idx * args.batch_size
            data = jax.tree_util.tree_map(lambda x: x[iter_idx], all_data)

            # Log episode rewards
            for batch_step, (sum_reward, done, gates_passed) in enumerate(
                zip(data.sum_rewards, data.dones, data.gates_passed)
            ):
                gates_passed = jp.arange(envs.unwrapped.race_data.n_gates + 1)[gates_passed]
                if jp.any(done):
                    wandb.log(
                        {
                            "train/gates_passed": jp.max(gates_passed),
                            "train/reward": jp.mean(sum_reward[done]),
                        },
                        step=global_step + batch_step * args.num_envs,
                    )
                    sum_rewards_hist.append(float(jp.mean(sum_reward[done])))

            # Log training metrics
            wandb.log(
                {
                    "losses/pg_loss": float(jp.mean(all_pg_loss[iter_idx])),
                    "losses/v_loss": float(jp.mean(all_v_loss[iter_idx])),
                    "losses/entropy": float(-jp.mean(all_entropy_loss[iter_idx])),
                    "losses/approx_kl": float(jp.mean(all_approx_kl[iter_idx])),
                    "losses/explained_variance": float(jp.mean(all_explained_var[iter_idx])),
                    "charts/SPS": int(global_step / training_time) if training_time > 0 else 0,
                },
                step=global_step + args.batch_size,
            )

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        params = {"actor": agent.actor_states.params, "critic": agent.critic_states.params}
        with open(model_path, "wb") as f:
            import pickle

            pickle.dump(params, f)
        print(f"model saved to {model_path}")

    return sum_rewards_hist, training_time


# region Evaluate
def evaluate_ppo(
    args: Args, n_eval: int, model_path: Path, render: bool, plot: bool = True
) -> tuple[float, float, list, list]:
    """Evaluate the trained policy (Flax/Agent).

    Loads params from `model_path` (pickle of {'actor':..., 'critic':...}) and runs
    `n_eval` episodes with deterministic actions.
    """
    r_coefs = {
        "gate_pos_coef": 0.0,
        "gate_vel_coef": 0.0,
        "gate_pass_coef": 0.0,
        "min_vel": args.min_vel,
        "max_vel": args.max_vel,
        "cont_floor_safe_dist": -1.0,
        "cont_gate_safe_dist": 0.0,
        "cont_obst_safe_dist": 0.0,
        "contact_coef": 1000.0,
        "gate_size": 0.45,
        "act_coefs": (0.0,) * 4,
        "d_act_coefs": (0.0,) * 4,
    }
    config = load_config(Path(__file__).parents[2] / "scripts/config_race.toml")
    eval_env = make_jitted_envs(
        num_envs=1,
        jax_device=args.jax_device,
        coefs=r_coefs,
        config=config.env,
        check_contacts=True,
    )
    eval_env = RecordRaceData.create(eval_env)

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
    success_mask = np.zeros(n_eval, dtype=bool)

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
            done = np.asarray(terminated | truncated, dtype=bool).any()
            episode_reward += np.mean(np.asarray(reward))
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        gates_passed = jp.arange(eval_env.unwrapped.race_data.n_gates + 1)[
            eval_env.unwrapped.race_data.target_gate[:, 0]
        ]
        success_mask[episode] = np.max(gates_passed) == eval_env.unwrapped.race_data.n_gates
        print(
            f"Collision cost: {episode_reward:.2f}, Gates passed: {np.max(gates_passed)}, Lap time: {steps / config.env.freq:.2f} s"
        )
        fig = eval_env.plot_eval(save_path=f"{args.exp_name}_eval_plot.png") if plot else None
    
    success_count = np.sum(success_mask)
    episode_lengths = np.array(episode_lengths)
    avg_lap_time = np.mean(episode_lengths[success_mask]) / config.env.freq if success_count > 0 else 10.0
    print(
        f"Success rate: {success_count}/{n_eval}, Average lap time: {avg_lap_time:.2f} s"
    )

    eval_env.close()

    return fig, success_count, episode_rewards, avg_lap_time


# region Main
def main(
    wandb_enabled: bool = True,
    train: bool = True,
    n_eval: int = 1,
    render: bool = True,
    plot: bool = True,
):
    """Main entry.

    Flags:
      wandb_enabled: log metrics to wandb
      train: run training
      n_eval: number of evaluation episodes to run after training (or standalone)
      render: whether to render the environment during evaluation
    """
    args = Args.create()
    model_path = Path(__file__).parents[2] / f"saves/{args.exp_name}_model.ckpt"
    jax_device = args.jax_device

    if train:  # use "--train False" to skip training
        train_ppo(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:  # use "--n_eval <N>" to perform N evaluation episodes
        fig, success_count, episode_rewards, episode_lengths = evaluate_ppo(
            args, n_eval, model_path, render, plot
        )
        if wandb_enabled and train:
            logs = {
                "eval/mean_rewards": np.mean(episode_rewards),
                "eval/mean_steps": np.mean(episode_lengths),
            }
            if fig is not None:
                logs["eval/eval_plot"] = wandb.Image(fig)
            if success_count is not None:
                logs["eval/success_count"] = success_count
            wandb.log(logs)
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
