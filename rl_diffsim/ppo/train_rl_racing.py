# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper, VectorWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from torch.distributions.normal import Normal

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config


@dataclass
class Args:
    seed: int = 44
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"
    """environment device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""

    # Algorithm specific arguments
    env_id: str = "DroneRacing-v0"
    """the id of the environment"""
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Crazyflow sim specific arguments
    physic: Literal["first_principles", "so_rpy_rotor_drag"] = "so_rpy_rotor_drag"
    """physics model type"""
    drone_model: str = "cf21B_500"
    """Crazyflie drone model name"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the observations."""
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)
        self.n_gates = env.single_observation_space["target_gate"].n

    def observations(self, observations: dict) -> dict:
        flat_obs = []
        for k, v in observations.items():
            if k == "target_gate" and self.n_gates is not None:
                target_onehot = jp.zeros((v.shape[0], self.n_gates))
                idx = jp.clip(v, 0, self.n_gates - 1) # 0 ~ 3
                target_onehot = target_onehot.at[jp.arange(v.shape[0]), idx.astype(int)].set(1.0)
                flat_obs.append(target_onehot)
            else:
                flat_obs.append(jp.reshape(v, (v.shape[0], -1)))
        return jp.concatenate(flat_obs, axis=-1)
    
class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""
    def __init__(self, env: VectorEnv):
        super().__init__(env)

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        actions = actions.at[..., -1].set(0.0) # optional: block yaw output because we don't need it
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, self.rewards(rewards, observations), terminations, truncations, infos
    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        # apply rpy penalty
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= 0.08 * rpy_norm
        return rewards

class DroneRacingWrapper(VectorWrapper):
    """Wrapper for training policy in Drone Racing Environment.
    
    This wrapper should be applied before FlattenJaxObservation.
    """
    def __init__(
        self, 
        env: VecDroneRaceEnv,
        n_samples: int = 10,
        samples_dt: float = 0.1,
        des_completion_time: float = 15.0,
    ):
        """Initialize the environment and create the reference trajectory.

        Args:
            env: Env to wrap.
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            des_completion_time: Desired time for completing the reference trajectory in seconds.
        """
        super().__init__(env)
        self.env = env
        # initialize trajectory here
        waypoints = np.array(
            [
                [-1.5, 1.0, 0.05],
                [-1.0, 0.8, 0.2],
                [0.3, 0.55, 0.5],
                [1.3, 0.2, 0.65],
                [0.85, 1.1, 1.1],
                [-0.5, 0.2, 0.65],
                [-1.15, 0.0, 0.52],
                [-1.15, 0.0, 1.1],
                [-0.0, -0.4, 1.1],
                [0.5, -0.4, 1.1],
            ]
        )
        # waypoints += np.array([0.0, 0.0, 2.0])
        self.des_completion_time = des_completion_time # sec
        ts = np.linspace(0, self.des_completion_time, int(self.env.unwrapped.freq * self.des_completion_time))
        spline = CubicSpline(np.linspace(0, self.des_completion_time, waypoints.shape[0]), waypoints)
        self.trajectory = spline(ts)  # (n_steps, 3)

        # Define trajectory sampling parameters
        self.n_samples = n_samples
        self.samples_dt = samples_dt
        self.sample_offsets = np.array(np.arange(n_samples) * self.env.unwrapped.freq * samples_dt, dtype=int)

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["traj_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.env.unwrapped.sim.n_worlds)

        # Update reset pipeline
        self.env.unwrapped.sim.reset_pipeline += (self._reset_randomization,)
        self.env.unwrapped.sim.build_reset_fn()

    def reset(self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all environment using seed and options."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.obs(obs), info
    
    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step the environments."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.obs(observations), self.rewards(rewards, observations), terminations, truncations, infos
    
    def render(self):
        """Render."""
        idx = np.clip(self.steps[:, None] + self.sample_offsets[None, ...], 0, self.trajectory.shape[0] - 1)
        next_trajectory = self.trajectory[idx, ...]
        draw_line(self.env.unwrapped.sim, self.trajectory[0:-1:2, :], rgba=np.array([1,1,1,0.4]), start_size=2.0, end_size=2.0)
        draw_line(self.env.unwrapped.sim, next_trajectory[0], rgba=np.array([1,0,0,1]), start_size=3.0, end_size=3.0)
        draw_points(self.env.unwrapped.sim, next_trajectory[0], rgba=np.array([1.0, 0, 0, 1]), size=0.01)
        self.env.unwrapped.sim.render()

    def obs(self, observations: dict[str, Array]) -> dict[str, Array]:
        """Sample some waypoints as extra observations."""
        idx = np.clip(self.steps[:, None] + self.sample_offsets[None, ...], 0, self.trajectory.shape[0] - 1)
        dpos = self.trajectory[idx, ...] - observations["pos"][:, None, :]
        observations["traj_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return observations

    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        """Rewards for tracking the target trajectory."""
        pos = observations["pos"] # (num_envs, 3)
        goal = self.trajectory[self.steps] # (num_envs, 3)

        norm_distance = jp.linalg.norm(pos - goal, axis=-1) # distance to next trajectory point
        rewards += 2.0*jp.exp(-2.0 * norm_distance) # encourage flying close to goal
        # rewards = jp.where(self.env.unwrapped.terminated().squeeze(), -1.0, rewards) # penalize drones that crash
        return rewards
    
    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.env.unwrapped.data.steps
    
    @staticmethod
    def _reset_randomization(data: SimData, mask: Array) -> SimData:
        """Randomize the initial position and velocity of the drones.

        This function will get compiled into the reset function of the simulation. Therefore, it
        must take data and mask as input arguments and must return a SimData object.
        """
        # Sample initial position
        shape = (data.core.n_worlds, data.core.n_drones, 3)
        pmin, pmax = jp.array([-0.1, -0.1, 1.9]), jp.array([0.1, 0.1, 2.1])
        key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
        data = data.replace(core=data.core.replace(rng_key=key))
        pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
        vel = jax.random.uniform(key=vel_key, shape=shape, minval=-0.5, maxval=0.5)
        data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
        return data
    
class RecordData(VectorWrapper):
    """Wrapper to record usefull data for debugging."""

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._record_act  = []
        self._record_pos  = []
        self._record_goal = []
        self._record_rpy  = []
    
    def _find_drone_racing_wrapper(self, env):
        while hasattr(env, "env"):
            if isinstance(env, DroneRacingWrapper):
                return env
            env = env.env
        return None

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        raw_env = self.env.unwrapped
        drone_racing_wrapper = self._find_drone_racing_wrapper(self.env)

        act = np.asarray(actions)
        self._record_act.append(act.copy())

        pos = np.asarray(raw_env.sim.data.states.pos[:, 0, :])   # shape: (n_worlds, 3)
        self._record_pos.append(pos.copy())

        goal = np.asarray(drone_racing_wrapper.trajectory[drone_racing_wrapper.steps])
        self._record_goal.append(goal.copy())

        rpy = np.asarray(R.from_quat(raw_env.sim.data.states.quat[:, 0, :]).as_euler("xyz"))
        self._record_rpy.append(rpy.copy())

        return obs, rewards, terminated, truncated, infos
    
    def calc_rmse(self):
        # compute rmse for all worlds
        pos = np.array(self._record_pos)     # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)   # shape: (T, num_envs, 3)
        pos_err = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        rmse = np.sqrt(np.mean(pos_err ** 2))*1000 # mm

        return rmse
    
    def plot_eval(self, save_path: str = "eval_plot.png"):
        import matplotlib
        matplotlib.use("Agg")  # render to raster images
        import matplotlib.pyplot as plt
        actions = np.array(self._record_act)
        pos = np.array(self._record_pos)
        goal = np.array(self._record_goal)
        rpy = np.array(self._record_rpy)

        # Plot the actions over time
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        for i in range(4):
            axes[i].plot(actions[:, 0, i])
            axes[i].set_title(f"{action_labels[i]} Command")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Action Value")
            axes[i].grid(True)

        # Plot position components
        position_labels = ["X Position", "Y Position", "Z Position"]
        for i in range(3):
            axes[4 + i].plot(pos[:, 0, i])
            axes[4 + i].set_title(position_labels[i])
            axes[4 + i].set_xlabel("Time Step")
            axes[4 + i].set_ylabel("Position (m)")
            axes[4 + i].grid(True)
        # Plot goal position components in same plots
        for i in range(3):
            axes[4 + i].plot(goal[:, 0, i], linestyle="--")
            axes[4 + i].legend(["Position", "Goal"])
        # Plot error in position
        pos_err = np.linalg.norm(pos[:, 0] - goal[:, 0], axis=1)
        axes[7].plot(pos_err)
        axes[7].set_title("Position Error")
        axes[7].set_xlabel("Time Step")
        axes[7].set_ylabel("Error (m)")
        axes[7].grid(True)

        # Plot angle components (roll, pitch, yaw)
        rpy_labels = ["Roll", "Pitch", "Yaw"]
        for i in range(3):
            axes[8 + i].plot(rpy[:, 0, i])
            axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
            axes[8 + i].set_xlabel("Time Step")
            axes[8 + i].set_ylabel("Angle (rad)")
            axes[8 + i].grid(True)

        # compute RMSE for position
        rmse_pos = np.sqrt(np.mean(pos_err**2))
        axes[11].text(0.1, 0.5, f"Position RMSE: {rmse_pos*1000:.3f} mm", fontsize=14)
        axes[11].axis("off")

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / save_path)

        return fig, axes, rmse_pos
    
def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_envs(
        config: str = "level0.toml",
        num_envs: int = None,
        jax_device: str = "cpu",
        torch_device: str = "cpu"
    ) -> VectorEnv:
    config = load_config(Path(__file__).parents[2] / "config" / config)
    env: VecDroneRaceEnv = gym.make_vec(
        config.env.id,
        num_envs = num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        device=jax_device,
    )
    env = NormalizeActions(env)
    env = DroneRacingWrapper(env)
    env = FlattenJaxObservation(env)
    env = JaxToTorch(env, torch_device)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

def main(wandb_enabled: bool = True, train: bool = True, eval: int = 1):
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device
    if train:
        if wandb_enabled:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
        
        train_start_time = time.time()
        
        set_seeds(args.seed) # TRY NOT TO MODIFY: seeding

        print("Training on device:", device, "| Environment device:", jax_device)

        # env setup
        envs = make_envs(num_envs=args.num_envs, jax_device=jax_device, torch_device=device)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            start_time = time.time()

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                # envs.render()
                next_done = terminations | truncations
                rewards[step] = reward
                print(reward[0])

                # TODO: log cummulative rewards
                # if wandb_enabled and next_done.any():
                #     for r in rewards[next_done]:
                #         wandb.log({"train/reward": r.item()}, step=global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if wandb_enabled:
                wandb.log(
                    {
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    }, step=global_step
                )
            end_time = time.time()
            print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
        train_end_time = time.time()
        print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
        if args.save_model:
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        envs.close()

    if eval > 0:
        set_seeds(args.seed)
        with torch.no_grad():
            eval_env = make_envs(num_envs=1, jax_device="cpu", torch_device="cpu")
            eval_env = RecordData(eval_env)
            agent    = Agent(eval_env).to("cpu")
            agent.load_state_dict(torch.load(model_path))
            episode_rewards = []
            episode_lengths = []
            # Evaluate the policy
            for episode in range(eval):
                obs, _ = eval_env.reset(seed=args.seed)
                done = torch.zeros(10, dtype=bool, device=device)
                episode_reward = 0
                steps = 0
                while not done.any():
                    act, _, _, _ = agent.get_action_and_value(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(act)
                    eval_env.render()

                    done = terminated | truncated
                    episode_reward += reward[0].item()
                    steps += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

            print(f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}")

            # plot figures, record RMSE
            fig, _, _ = eval_env.plot_eval()
            rmse_pos = eval_env.calc_rmse()

            if wandb_enabled and train:
                wandb.log(
                    {
                        "eval/eval_plot": wandb.Image(fig),
                        "eval/pos_rmse_mm": rmse_pos * 1000,
                        "eval/mean_rewards": np.mean(episode_rewards),
                        "eval/mean_steps": np.mean(episode_lengths),
                    }
                )
                wandb.finish()
        eval_env.close()



if __name__ == "__main__":
    fire.Fire(main)
