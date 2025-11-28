"""Drone environment for following a random trajectory."""

from typing import Callable, Literal

import jax
import jax.numpy as jp
import numpy as np
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array


class FigureEightEnv(DroneEnv):
    """Drone environment for following a random trajectory.

    This environment is used to follow a random trajectory. The observations contain the
    relative position errors to the next `n_samples` points that are distanced by `samples_dt`. The
    reward is based on the distance to the next trajectory point.
    """

    def __init__(
        self,
        n_samples: int = 10,
        trajectory_time: float = 10.0,
        samples_dt: float = 0.1,
        *,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Physics = Physics.first_principles,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        device: str = "cpu",
        reset_rotor: bool = False,
    ):
        """Initialize the environment and create the figure-eight trajectory.

        Args:
            n_samples: Number of next trajectory points to sample for observations.
            samples_dt: Time between trajectory sample points in seconds.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            num_envs: Number of environments to run in parallel.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation.
            device: Device to use for the simulation.
            reset_rotor: Whether to reset rotor speeds on environment reset.
        """
        # Override reset randomization function
        self._reset_randomization = self.build_reset_randomization_fn(
            physics if reset_rotor else "no_reset_rotor"
        )

        super().__init__(
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            drone_model=drone_model,
            freq=freq,
            device=device,
        )
        if trajectory_time < self.max_episode_time:
            raise ValueError("Trajectory time must be greater than max episode time")

        # Define trajectory sampling parameters
        self.num_waypoints = 10
        self.n_samples = n_samples
        self.samples_dt = samples_dt
        self.trajectory_time = trajectory_time
        # number of simulation steps for the trajectory
        self.n_steps = int(jp.ceil(self.trajectory_time * self.freq).item())
        # offsets (in steps) for the samples returned in observations
        self.sample_offsets = jp.array(jp.arange(n_samples) * self.freq * samples_dt, dtype=int)
        self.trajectories = jp.zeros((self.num_envs, self.n_steps, 3))

        # Create the figure eight trajectory
        n_steps = int(jp.ceil(trajectory_time * self.freq).item())
        t = jp.linspace(0, 2 * jp.pi, n_steps)
        offset = jp.linspace(0, 2 * jp.pi, self.num_envs, endpoint=False)
        ts = t[None, :] + offset[:, None]  # random phase shift
        radius = 1  # Radius for the circles
        x = radius * jp.sin(ts)  # Scale amplitude for 1-meter diameter
        y = jp.zeros_like(ts)  # y is 0 everywhere
        z = radius / 2 * jp.sin(2 * ts) + 1.5  # Scale amplitude for 1-meter diameter
        self.trajectories = jp.array([x.T, y.T, z.T]).T  # (num_envs, n_steps, 3)

        # Set takeoff position and build default reset position
        # self.takeoff_pos = jp.array([-1.5, 1.0, 0.07])
        self.takeoff_pos = self.trajectories[:, :1, :]
        data = self.sim.data
        self.sim.data = data.replace(states=data.states.replace(pos=self.takeoff_pos))
        # self.sim.data = data.replace(states=data.states.replace(pos=jp.broadcast_to(self.takeoff_pos, (data.core.n_worlds, data.core.n_drones, 3))))
        self.sim.build_default_data()

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        # use Python floats for infinity (compatible with gym spaces)
        spec["local_samples"] = spaces.Box(-float("inf"), float("inf"), shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset."""
        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options)  # call jax reset function
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        """Render."""
        idx = jp.clip(self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1)
        next_trajectory = self.trajectories[jp.arange(self.trajectories.shape[0])[:, None], idx]
        trajectories = np.array(self.trajectories)
        next_trajectory = np.array(next_trajectory)
        draw_line(self.sim, trajectories[0, 0:-1:2, :], rgba=jp.array([1, 1, 1, 0.4]), start_size=2.0, end_size=2.0)
        draw_line(self.sim, next_trajectory[0], rgba=jp.array([1, 0, 0, 1]), start_size=3.0, end_size=3.0)
        draw_points(self.sim, next_trajectory[0], rgba=jp.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        """Observations."""
        obs = super().obs()
        obs["local_samples"] = self._aux_obs(
            self.trajectories, self.steps, self.sim.data.states.pos, self.sample_offsets
        )
        return obs

    @staticmethod
    @jax.jit
    def _aux_obs(
        trajectories: Array, steps: Array, pos: Array, sample_offsets: Array
    ) -> dict[str, Array]:
        """Static method version of obs for jitting."""
        idx = jp.clip(steps + sample_offsets[None, ...], 0, trajectories.shape[1] - 1)
        dpos = trajectories[jp.arange(trajectories.shape[0])[:, None], idx] - pos
        local_samples = dpos.reshape(dpos.shape[0], dpos.shape[1] * dpos.shape[2])
        return local_samples

    def reward(self) -> Array:
        """Rewards."""
        pos = self.sim.data.states.pos[:, 0, :]
        goal = self.trajectories[jp.arange(self.trajectories.shape[0])[:, None], self.steps][
            :, 0, :
        ]  # (num_envs, 3)
        return self._reward(self.terminated(), pos, goal)

    @staticmethod
    @jax.jit
    def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
        # distance to next trajectory point
        norm_distance = jp.linalg.norm(pos - goal, axis=-1)
        reward = jp.exp(-2.0 * norm_distance)
        reward = jp.where(terminated, -1.0, reward)
        return reward

    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1

    @staticmethod
    @jax.jit
    def _terminated(pos: Array) -> Array:
        lower_bounds = jp.array([-4.0, -4.0, -0.0])
        upper_bounds = jp.array([4.0, 4.0, 4.0])
        terminate = jp.any((pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1)
        return terminate

    def build_reset_randomization_fn(self, physics: str) -> Callable[[SimData, Array], SimData]:
        """Reset randomization."""

        # Spin up rotors to help takeoff
        def _reset_randomization_so_rpy(data: SimData, mask: Array) -> SimData:
            rotor_vel = 0.05 * jp.ones(
                (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
            )
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data

        def _reset_randomization_first_principles(data: SimData, mask: Array) -> SimData:
            rotor_vel = 10000.0 * jp.ones(
                (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
            )
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data

        def _reset_randomization_no_reset_rotor(data: SimData, mask: Array) -> SimData:
            return data

        match physics:
            case "first_principles":
                return _reset_randomization_first_principles
            case "so_rpy" | "so_rpy_rotor" | "so_rpy_rotor_drag":
                return _reset_randomization_so_rpy
            case "no_reset_rotor":
                return _reset_randomization_no_reset_rotor
