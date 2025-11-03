"""A naive RL pipeline for drone racing."""
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from crazyflow.envs.norm_actions_wrapper import NormalizeActions
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import (
    VectorActionWrapper,  # noqa: F401
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
    VectorWrapper,
)
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline  # noqa: F401
from torch import Tensor
from torch.distributions.normal import Normal

import wandb
from rl_diffsim.envs.rand_traj import RandTrajEnv


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
    learning_rate: float = 1.5e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.94
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
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


    
# region Wrappers
# class ActionTransform(VectorActionWrapper):
#     """Wrapper to transform rotation vector to attitude commands."""

#     def __init__(self, env: VectorEnv):
#         super().__init__(env)
#         self._action_scale = (1 / super().unwrapped.freq) * 5 * jp.pi
#         self._action_scale = jp.array(0.1)
#         # Compute scale and mean for rescaling
#         thrust_low = self.single_action_space.low[-1]
#         thrust_high = self.single_action_space.high[-1]
#         self._scale = jp.array((thrust_high - thrust_low) / 2.0, device=super().unwrapped.device)
#         self._mean = jp.array((thrust_high + thrust_low) / 2.0, device=super().unwrapped.device)
#         # Modify the wrapper's action space to [-1, 1]
#         self.single_action_space.low = -np.ones_like(self.single_action_space.low)
#         self.single_action_space.high = np.ones_like(self.single_action_space.high)
#         self.action_space = batch_space(self.single_action_space, self.num_envs)

#     def actions(self, actions: Array) -> Array:
#         rpy = jp.clip(actions[..., :3], -1.0, 1.0) * jp.pi/2
#         # obs = super().unwrapped.obs()
#         # rpy = self._action_rot2rpy(jp.array(obs["quat"]), actions[..., :3], self._action_scale)
#         # rpy = (R.from_matrix(obs["quat"].reshape(-1,3,3)) * R.from_rotvec(jp.clip(actions[..., :3], -1.0, 1.0) * self._action_scale)).as_euler("xyz")
#         # rpy = (R.from_rotvec(jp.clip(actions[..., :3], -1.0, 1.0) * self._action_scale)).as_euler("xyz")
#         # rpy = R.from_quat(obs["quat"]).as_euler("xyz") + actions[..., :3] * self._action_scale
#         rpy = rpy.at[..., 2].set(0.0)
#         actions = actions.at[..., :3].set(rpy)
#         th = jp.clip(actions[..., -1], -1.0, 1.0) * self._scale + self._mean
#         actions = actions.at[..., -1].set(th)
#         return actions
    
#     @staticmethod
#     @jax.jit
#     def _action_rot2rpy(rot_mat, action_rot, action_scale):
#         rpy = (R.from_matrix(rot_mat.reshape(-1, 3, 3)) * R.from_rotvec(jp.clip(action_rot, -1.0, 1.0) * action_scale)).as_euler("xyz")
#         return rpy
    
class StackObs(VectorObservationWrapper):
    """Wrapper to stack history observations."""
    def __init__(self, env: VectorEnv, n_obs: int = 0):
        """Init."""
        super().__init__(env)
        self.n_obs = n_obs
        print(f"{n_obs=}")
        if self.n_obs > 0:
            # Update observation space
            spec = {k: v for k, v in self.single_observation_space.items()}
            spec["prev_obs"] = spaces.Box(-np.inf, np.inf, shape=(13 * self.n_obs,))
            self.single_observation_space = spaces.Dict(spec)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            # Init obs buffer
            init_obs = env.unwrapped.obs()
            self._prev_obs = jp.zeros((self.num_envs, self.n_obs, 13))
            for _ in range(n_obs):
                self._prev_obs = self._update_prev_obs(self._prev_obs, init_obs)

    def observations(self, observations: dict) -> dict:
        """Override observation."""
        if self.n_obs > 0:
            observations["prev_obs"] = self._prev_obs.reshape(self.num_envs, -1)
            self._prev_obs = self._update_prev_obs(self._prev_obs, observations)
        return observations
    
    @staticmethod
    @jax.jit
    def _update_prev_obs(prev_obs: Array, obs: dict) -> Array:
        """Update previous observations."""
        basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = jp.concatenate(
            [jp.reshape(obs[k], (obs[k].shape[0], -1)) for k in basic_obs_key], axis=-1
        )
        prev_obs = jp.concatenate(
            [prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1
        )
        return prev_obs


class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""
    def __init__(self, env: VectorEnv, rpy_coef: float = 0.08):
        super().__init__(env)
        self.rpy_coef = rpy_coef

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        actions = actions.at[..., 2].set(0.0) # block yaw output because we don't need it
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, self.rewards(rewards, observations), terminations, truncations, infos
    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        # apply rpy penalty
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= self.rpy_coef * rpy_norm
        return rewards
        
class ActionPenalty(VectorObservationWrapper):
    """Wrapper to apply action penalty."""
    def __init__(self, env: VectorEnv, act_coef: float = 0.01, d_act_th_coef: float = 0.2, d_act_xy_coef: float = 0.4):
        """Init."""
        super().__init__(env)
        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["last_action"] = spaces.Box(-np.inf, np.inf, shape=(4,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._last_action = jp.zeros((self.num_envs, 4))
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Override step."""
        obs, reward, terminated, truncated, info = super().step(action)
        # penalty on actions
        action_diff = action - self._last_action
        # energy
        reward -= self.act_coef * action[..., -1] ** 2
        # smoothness
        reward -= self.d_act_th_coef * action_diff[..., -1] ** 2
        reward -= self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)
        self._last_action = action
        return self.observations(obs), reward, terminated, truncated, info
    
    def observations(self, observations: dict) -> dict:
        """Override observation."""
        observations["last_action"] = self._last_action
        return observations

class FlattenJaxObservation(VectorObservationWrapper):
    """Wrapper to flatten the observations."""
    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        self.single_observation_space = flatten_space(env.single_observation_space)
        self.observation_space = flatten_space(env.observation_space)

    def observations(self, observations: dict) -> dict:
        """Flatten observations."""
        return jp.concatenate([jp.reshape(v, (v.shape[0], -1)) for k, v in observations.items()], axis=-1)
    
class ObsNoise(VectorObservationWrapper):
    """Simple wrapper to add noise to the observations."""

    def __init__(self, env: VectorEnv, noise_std: float = 0.01):
        """Wrap the environment to add noise to the observations."""
        super().__init__(env)
        self.noise_std = noise_std
        self.prng_key = jax.random.PRNGKey(0)

    def observations(self, observation: Array) -> Array:
        """Add noise to the observations."""
        self.prng_key, key = jax.random.split(self.prng_key)
        noise = jax.random.normal(key, shape=observation.shape) * self.noise_std
        return observation + noise
    
class RecordData(VectorWrapper):
    """Wrapper to record usefull data for debugging."""

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._record_act  = []
        self._record_pos  = []
        self._record_goal = []
        self._record_rpy  = []

    def step(self, actions: Any):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        raw_env = self.env.unwrapped

        act = np.asarray(actions)
        self._record_act.append(act.copy())
        pos = np.asarray(raw_env.sim.data.states.pos[:, 0, :])   # shape: (n_worlds, 3)
        self._record_pos.append(pos.copy())
        goal = np.asarray(raw_env.trajectories[np.arange(raw_env.steps.shape[0]), raw_env.steps.squeeze(1)])
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

# region MakeEnvs
def make_envs(
        num_envs: int = None,
        jax_device: str = "cpu",
        torch_device: torch.device = torch.device("cpu"),
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
    env = JaxToTorch(env, torch_device)
    return env

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# region Agent
class Agent(nn.Module):
    """RL Agent."""
    def __init__(self, obs_shape:tuple, action_shape:tuple):
        """Init network structures."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, torch.tensor(action_shape).prod()), std=0.01
            ),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, torch.tensor(action_shape).prod())
            # torch.Tensor([[-1, -1, -1, 1]]) # start with smaller std for roll, pitch, yaw
        )

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Action output."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal distribution. The standard deviation is a learnable parameter that should decrease during training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# region Train
def train_ppo(args: Args, model_path: Path, device: torch.device, jax_device: str, wandb_enabled: bool = False) -> None:
        """Train.

        An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
        """
        # train setup
        if wandb_enabled and wandb.run is None:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
        train_start_time = time.time()
        set_seeds(args.seed) # TRY NOT TO MODIFY: seeding
        print("Training on device:", device, "| Environment device:", jax_device)

        # env setup
        r_coefs = {
            "n_obs": args.n_obs,
            "rpy_coef": args.rpy_coef,
            "d_act_xy_coef": args.d_act_xy_coef,
            "d_act_th_coef": args.d_act_th_coef,
            "act_coef": args.act_coef,
        }
        envs = make_envs(num_envs=args.num_envs, jax_device=jax_device, torch_device=device, coefs=r_coefs, reset_rotor=True)
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
        sum_rewards = torch.zeros((args.num_envs)).to(device)
        sum_rewards_hist = []

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
                rewards[step] = reward
                sum_rewards += reward
                sum_rewards[next_done.bool()] = 0
                next_done = terminations | truncations

                if wandb_enabled and next_done.any():
                    for r in sum_rewards[next_done.bool()]:
                        wandb.log({"train/reward": r.item()}, step=global_step)
                        sum_rewards_hist.append(r.item())

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
        if model_path is not None:
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        envs.close()

        return sum_rewards_hist

# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float]:
    """Evaluate."""
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
    }
    eval_env = make_envs(num_envs=1, coefs=r_coefs)
    eval_env = RecordData(eval_env)
    agent = Agent(eval_env.single_observation_space.shape, eval_env.single_action_space.shape).to(device)
    agent.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        episode_rewards = []
        episode_lengths = []
        ep_seed = args.seed
        # Evaluate the policy
        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=(ep_seed:=ep_seed+1))
            done = torch.zeros(10, dtype=bool, device=device)
            episode_reward = 0
            steps = 0
            while not done.any():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(act)
                eval_env.render()
                done = terminated | truncated
                episode_reward += reward[0].item()
                steps += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        # plot figures, record RMSE
        fig, _, _ = eval_env.plot_eval()
        rmse_pos = eval_env.calc_rmse()
        print(f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}, Pos RMSE = {rmse_pos:.2f} mm")

        eval_env.close()

        return fig, rmse_pos, episode_rewards, episode_lengths


# region Main
def main(wandb_enabled: bool = True, train: bool = True, eval: int = 1):
    """Main."""
    args = Args.create()
    model_path = Path(__file__).parents[2] / "saves/ppo_model.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    jax_device = args.jax_device

    if train: # use "--train False" to skip training
        train_ppo(args, model_path, device, jax_device, wandb_enabled)

    if eval > 0: # use "--eval <N>" to perform N evaluation episodes
        fig, rmse_pos, episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_path)
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/eval_plot": wandb.Image(fig),
                    "eval/pos_rmse_mm": rmse_pos,
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)
