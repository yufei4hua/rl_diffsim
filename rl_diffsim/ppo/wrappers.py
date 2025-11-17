"""Wrappers for PPO training."""
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import matplotlib
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.vector import (
    VectorActionWrapper,
    VectorEnv,
    VectorObservationWrapper,
    VectorRewardWrapper,
    VectorWrapper,
)
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")  # render to raster images
import matplotlib.pyplot as plt
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from scipy.interpolate import CubicSpline


class StackObs(VectorObservationWrapper):
    """Wrapper to stack history observations."""

    def __init__(self, env: VectorEnv, n_obs: int = 0):
        """Init."""
        super().__init__(env)
        self.n_obs = n_obs
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
        prev_obs = jp.concatenate([prev_obs[:, 1:, :], basic_obs[:, None, :]], axis=1)
        return prev_obs


class AngleReward(VectorRewardWrapper):
    """Wrapper to penalize orientation in the reward."""

    def __init__(self, env: VectorEnv, rpy_coef: float = 0.08):
        """Init."""
        super().__init__(env)
        self.rpy_coef = rpy_coef

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Set yaw command to zero."""
        actions = actions.at[..., 2].set(0.0)  # block yaw output because we don't need it
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, self.rewards(rewards, observations), terminations, truncations, infos

    def rewards(self, rewards: Array, observations: dict[str, Array]) -> Array:
        """Additional angular rewards."""
        # apply rpy penalty
        rpy_norm = jp.linalg.norm(R.from_quat(observations["quat"]).as_euler("xyz"), axis=-1)
        rewards -= self.rpy_coef * rpy_norm
        return rewards


class ActionPenalty(VectorObservationWrapper):
    """Wrapper to apply action penalty."""

    def __init__(
        self,
        env: VectorEnv,
        act_coef: float = 0.01,
        d_act_th_coef: float = 0.2,
        d_act_xy_coef: float = 0.4,
    ):
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
        self._obs_keys = sorted(env.single_observation_space.spaces.keys())

    def observations(self, observations: dict) -> dict:
        """Flatten observations."""
        return jp.concatenate([jp.reshape(observations[k], (observations[k].shape[0], -1)) for k in self._obs_keys], axis=-1)
    
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
        """Init."""
        super().__init__(env)
        self._record_act  = []
        self._record_pos  = []
        self._record_goal = []
        self._record_rpy  = []

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Step and record data."""
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

    def calc_rmse(self) -> float:
        """Compute RMSE between position and goal over the recorded data."""
        # compute rmse for all worlds
        pos = np.array(self._record_pos)     # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)   # shape: (T, num_envs, 3)
        pos_err = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        rmse = np.sqrt(np.mean(pos_err ** 2))*1000 # mm

        return rmse

    def plot_eval(self, save_path: str = "eval_plot.png") -> tuple[plt.Figure, list[plt.Axes], float]:
        """Plot the recorded data and save to file."""
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
    
class DroneRacingWrapper(VectorWrapper):
    """Wrapper for training policy in Drone Racing Environment.
    
    This wrapper should be applied before FlattenJaxObservation.
    """
    def __init__(
        self, 
        env: VectorEnv,
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

class ActionTransform(VectorActionWrapper):
    """Wrapper to transform rotation vector to attitude commands."""

    def __init__(self, env: VectorEnv):
        """Init."""
        super().__init__(env)
        self._action_scale = (1 / super().unwrapped.freq) * 5 * jp.pi
        self._action_scale = jp.array(0.1)
        # Compute scale and mean for rescaling
        thrust_low = self.single_action_space.low[-1]
        thrust_high = self.single_action_space.high[-1]
        self._scale = jp.array((thrust_high - thrust_low) / 2.0, device=super().unwrapped.device)
        self._mean = jp.array((thrust_high + thrust_low) / 2.0, device=super().unwrapped.device)
        # Modify the wrapper's action space to [-1, 1]
        self.single_action_space.low = -np.ones_like(self.single_action_space.low)
        self.single_action_space.high = np.ones_like(self.single_action_space.high)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def actions(self, actions: Array) -> Array:
        """Transform rotation vector to attitude commands."""
        rpy = jp.clip(actions[..., :3], -1.0, 1.0) * jp.pi/2
        # obs = super().unwrapped.obs()
        # rpy = self._action_rot2rpy(jp.array(obs["quat"]), actions[..., :3], self._action_scale)
        # rpy = (R.from_matrix(obs["quat"].reshape(-1,3,3)) * R.from_rotvec(jp.clip(actions[..., :3], -1.0, 1.0) * self._action_scale)).as_euler("xyz")
        # rpy = (R.from_rotvec(jp.clip(actions[..., :3], -1.0, 1.0) * self._action_scale)).as_euler("xyz")
        # rpy = R.from_quat(obs["quat"]).as_euler("xyz") + actions[..., :3] * self._action_scale
        rpy = rpy.at[..., 2].set(0.0)
        actions = actions.at[..., :3].set(rpy)
        th = jp.clip(actions[..., -1], -1.0, 1.0) * self._scale + self._mean
        actions = actions.at[..., -1].set(th)
        return actions
    
    @staticmethod
    @jax.jit
    def _action_rot2rpy(rot_mat, action_rot, action_scale) -> Array:
        """Convert rotation vector action to rpy commands."""
        rpy = (R.from_matrix(rot_mat.reshape(-1, 3, 3)) * R.from_rotvec(jp.clip(action_rot, -1.0, 1.0) * action_scale)).as_euler("xyz")
        return rpy