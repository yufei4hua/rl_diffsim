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
from ml_collections import ConfigDict

import wandb
from rl_diffsim.envs.drone_race_env import DroneRaceEnv
from rl_diffsim.envs.race_utils import load_config
from rl_diffsim.envs.wrappers import ActionPenalty, FlattenJaxObservation, NormalizeActions, ZeroYaw
from rl_diffsim.envs.wrappers_race import RaceWrapper, RecordRaceData
from rl_diffsim.shac.shac_agent import Agent


# region Arguments
@dataclass(frozen=True)
class Args:
    """Class to store configurations."""

    seed: int = 42
    """seed of the experiment"""
    jax_device: str = "cpu"
    """environment device"""
    exp_name: str = "shac_race_lv2"
    """the name of the experiment"""
    wandb_project_name: str = "rl-shac-race-lv2"
    """the wandb's project name"""
    wandb_entity: str = "fresssack"
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 400_000
    """total timesteps of the experiments"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 48
    """the number of steps to run in each environment per policy rollout"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    anneal_actor_lr: bool = True
    """Toggle learning rate annealing for policy networks"""
    anneal_critic_lr: bool = True
    """Toggle learning rate annealing for value networks"""
    actor_lr: float = 1e-2
    """the learning rate of the actor optimizer"""
    critic_lr: float = 8e-4
    """the learning rate of the critic optimizer"""
    gamma: float = 0.96
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the TD-lambda calculation"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    clip_coef: float = 0.6
    """the surrogate clipping coefficient"""
    hidden_size: int = 32
    """the hidden size of actor and critic networks"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    min_vel: float = 0.66
    max_vel: float = 2.2
    cont_floor_safe_dist: float = 0.05
    cont_gate_safe_dist: float = 0.12
    cont_obst_safe_dist: float = 0.22
    gate_size: float = 0.20
    gate_pos_coef: float = 1.5
    gate_vel_coef: tuple = (3.2, 1.1)
    gate_pass_coef: float = 52.0
    gate_pass_pos_coef: float = 8.0
    gate_pass_vel_coef: float = 10.0
    contact_coef: tuple = (8.0, 99.5)
    act_coefs: tuple = (0.3, 0.3, 0.0, 0.1)
    d_act_coefs: tuple = (0.6, 0.6, 0.0, 0.3)
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        batch_size = int(args.num_envs * args.num_steps)
        num_iterations = args.total_timesteps // batch_size
        act_coefs = (args.act_coefs[0],) * 2 + (0.0, args.act_coefs[3])
        d_act_coefs = (args.d_act_coefs[0],) * 2 + (0.0, args.d_act_coefs[3])
        return replace(
            args,
            batch_size=batch_size,
            num_iterations=num_iterations,
            act_coefs=act_coefs,
            d_act_coefs=d_act_coefs,
        )


# region MakeEnvs
def make_jitted_envs(
    num_envs: int = None,
    jax_device: str = "cpu",
    total_timesteps: int = 0,
    coefs: dict = {},
    config: ConfigDict = ConfigDict(),
    check_contacts: bool = True,
    end_on_gate_bypass: bool = False,
) -> DroneRaceEnv:
    """Make environments for training RL policy."""
    env: DroneRaceEnv = DroneRaceEnv.create(
        num_envs=num_envs,
        device=jax_device,
        check_contacts=check_contacts,
        end_on_gate_bypass=end_on_gate_bypass,
        **config,
    )

    env = RaceWrapper.create(
        env,
        gate_pos_coef=coefs.get("gate_pos_coef", 0.0),
        gate_vel_coef=coefs.get("gate_vel_coef", 0.0),
        gate_pass_coef=coefs.get("gate_pass_coef", 0.0),
        gate_pass_pos_coef=coefs.get("gate_pass_pos_coef", 0.0),
        gate_pass_vel_coef=coefs.get("gate_pass_vel_coef", 0.0),
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
    rewards: Array
    dones: Array
    values: Array
    sum_rewards: Array
    entropy: Array
    returns: Array
    losses: Array
    gates_passed: Array


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

            gates_passed = env.unwrapped.race_data.target_gate[:, 0]
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
                gates_passed=gates_passed,
            )

        (envs, key, sum_rewards, next_discounts, next_obs, next_done), rollout_data = jax.lax.scan(
            step_once, (envs, key, sum_rewards, discounts, next_obs, next_done), length=args.num_steps
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
        return losses / (args.num_envs * args.num_steps), (envs, data, next_obs, next_done, sum_rewards, key)

    policy_grad_fn = jax.value_and_grad(policy_loss_fn, argnums=(1,), has_aux=True)
    (p_loss, (envs, data, next_obs, next_done, sum_rewards, key)), (g_actor,) = policy_grad_fn(
        envs, agent.actor_states.params, agent.critic_states.params, next_obs, next_done, sum_rewards, key
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

    def compute_td_lambda_once(carry: Array, inp: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
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
            v_loss, g_critic = value_grad_fn(
                agent.critic_states.params, minibatch.observations, minibatch.returns, minibatch.values
            )
            return agent.replace(critic_states=agent.critic_states.apply_gradients(grads=g_critic)), v_loss

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
        "gate_pos_coef": args.gate_pos_coef,
        "gate_vel_coef": args.gate_vel_coef,
        "gate_pass_coef": args.gate_pass_coef,
        "gate_pass_pos_coef": args.gate_pass_pos_coef,
        "gate_pass_vel_coef": args.gate_pass_vel_coef,
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
    config = load_config(Path(__file__).parents[2] / "scripts/config_race_lv2.toml")
    envs = make_jitted_envs(
        num_envs=args.num_envs,
        jax_device=jax_device,
        total_timesteps=args.total_timesteps,
        coefs=r_coefs,
        config=config.env,
        check_contacts=False,
        end_on_gate_bypass=False,
    )

    # setup annealing learning rate
    if args.anneal_actor_lr:
        actor_lr = optax.linear_schedule(
            init_value=args.actor_lr, end_value=0.0, transition_steps=args.num_iterations
        )
    else:
        actor_lr = args.actor_lr
    if args.anneal_critic_lr:
        critic_lr = optax.linear_schedule(
            init_value=args.critic_lr,
            end_value=0.0,
            transition_steps=args.num_iterations * args.update_epochs * args.num_minibatches,
        )
    else:
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
    # for _ in range(1):
    #     (_, _, _), (p_loss, (data, _, next_done, sum_rewards)) = update_policy(
    #         envs=envs,
    #         args=args,
    #         agent=agent,
    #         next_obs=next_obs,
    #         next_done=jp.zeros(args.num_envs, dtype=bool),
    #         sum_rewards=jp.zeros((args.num_envs,)),
    #         key=key,
    #     )
    #     data = compute_td_lambda(args, agent, next_obs, next_done, data)
    #     (_, _), (v_loss, explained_var) = update_value(args, agent, data, key)
    # v_loss.block_until_ready()
    print("JAX warmup took {:.5f} s".format(time.time() - start_warmup_time))

    def training_step(carry: tuple, _) -> tuple[tuple, tuple]:
        """Single training iteration using scan."""
        envs, agent, key, next_obs, next_done, sum_rewards = carry
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
        # 2. compute TD-λ
        data = compute_td_lambda(args, agent, next_obs, next_done, data)
        # 3. value update
        (agent, key), (v_loss, explained_var) = update_value(args, agent, data, key)

        return (envs, agent, key, next_obs, next_done, sum_rewards), (p_loss, v_loss, explained_var, data)

    # start the game
    train_start_time = time.time()
    global_step = 0
    next_done = jp.zeros(args.num_envs, dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,))
    # envs, (next_obs, _) = envs.reset(envs, seed=args.seed) # already done in warmup

    ((envs, agent, key, next_obs, next_done, sum_rewards), (p_losses, v_losses, explained_vars, all_data)) = (
        jax.lax.scan(
            training_step, (envs, agent, key, next_obs, next_done, sum_rewards), length=args.num_iterations
        )
    )

    next_obs.block_until_ready()
    training_time = time.time() - train_start_time
    global_step = args.num_iterations * args.batch_size
    print(f"Training for {global_step} steps took {training_time:.2f} seconds.")

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
                    "losses/p_loss": float(jp.mean(p_losses[iter_idx])),
                    "losses/v_loss": float(jp.mean(v_losses[iter_idx])),
                    "losses/entropy": float(jp.mean(data.entropy)),
                    "losses/explained_variance": float(jp.mean(explained_vars[iter_idx])),
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
def evaluate_shac(
    args: Args, n_eval: int, model_path: Path, render: bool, plot: bool = True
) -> tuple[float, float, list, list]:
    """Evaluate the trained policy (Flax/Agent).

    Loads params from `model_path` (pickle of {'actor':..., 'critic':...}) and runs
    `n_eval` episodes with deterministic actions.
    """
    r_coefs = {
        "gate_pos_coef": args.gate_pos_coef,
        "gate_vel_coef": args.gate_vel_coef,
        "gate_pass_coef": args.gate_pass_coef,
        "gate_pass_pos_coef": args.gate_pass_pos_coef,
        "gate_pass_vel_coef": args.gate_pass_vel_coef,
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
    config = load_config(Path(__file__).parents[2] / "scripts/config_race_lv2.toml")
    eval_env = make_jitted_envs(
        num_envs=1,
        jax_device=args.jax_device,
        coefs=r_coefs,
        config=config.env,
        check_contacts=True,
        end_on_gate_bypass=False,
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
            done = terminated | truncated
            episode_reward += float(np.asarray(reward).item())
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        gates_passed = jp.arange(eval_env.unwrapped.race_data.n_gates + 1)[
            eval_env.unwrapped.race_data.target_gate[:, 0]
        ]
        success_mask[episode] = np.max(gates_passed) == eval_env.unwrapped.race_data.n_gates
        print(
            f"Collision cost: {episode_reward:.2f}, Gates passed: {np.max(gates_passed)}, \
            Lap time: {steps / config.env.freq:.2f} s"
        )
        fig = eval_env.plot_eval(save_path=f"{args.exp_name}_eval_plot.png") if plot else None

    success_count = np.sum(success_mask)
    episode_lengths = np.array(episode_lengths)
    avg_lap_time = np.mean(episode_lengths[success_mask]) / config.env.freq if success_count > 0 else 10.0
    print(f"Success rate: {success_count}/{n_eval}, Average lap time: {avg_lap_time:.2f} s")

    eval_env.close()

    return fig, success_count, episode_rewards, avg_lap_time


# region Main
def main(
    wandb_enabled: bool = True, train: bool = True, n_eval: int = 1, render: bool = True, plot: bool = True
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
        train_shac(args, model_path, jax_device, wandb_enabled)

    if n_eval > 0:  # use "--n_eval <N>" to perform N evaluation episodes
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_shac(
            args, n_eval, model_path, render, plot
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
