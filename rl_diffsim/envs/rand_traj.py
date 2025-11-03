
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


class RandTrajEnv(DroneEnv):
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
        physics: Literal["so_rpy_rotor_drag", "first_principles"] | Physics = Physics.first_principles,
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
        """
        # Override reset randomization function
        self._reset_randomization = self.build_reset_randomization_fn(physics if reset_rotor else "no_reset_rotor")

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
        self.n_steps = int(np.ceil(self.trajectory_time * self.freq))
        self.sample_offsets = np.array(np.arange(n_samples) * self.freq * samples_dt, dtype=int)
        self.trajectories = np.zeros((self.num_envs, self.n_steps, 3))

        # Create the figure eight trajectory
        n_steps = int(np.ceil(trajectory_time * self.freq))
        t = np.linspace(0, 2 * np.pi, n_steps)
        offset = np.linspace(0, 2*np.pi, self.num_envs, endpoint=False)
        ts = t[None, :] + offset[:, None]  # random phase shift
        radius = 1  # Radius for the circles
        x = radius * np.sin(ts)  # Scale amplitude for 1-meter diameter
        y = np.zeros_like(ts)  # y is 0 everywhere
        z = radius / 2 * np.sin(2 * ts) + 1.5  # Scale amplitude for 1-meter diameter
        self.trajectories = np.array([x.T, y.T, z.T]).T # (num_envs, n_steps, 3)

        # Set takeoff position and build default reset position
        # self.takeoff_pos = np.array([-1.5, 1.0, 0.07])
        self.takeoff_pos = self.trajectories[:, :1, :]
        data = self.sim.data
        self.sim.data = data.replace(states=data.states.replace(pos=self.takeoff_pos))
        # self.sim.data = data.replace(states=data.states.replace(pos=np.broadcast_to(self.takeoff_pos, (data.core.n_worlds, data.core.n_drones, 3))))
        self.sim.build_default_data()

        # # Create a random trajectory based on spline interpolation
        # n_steps = int(np.ceil(self.trajectory_time * self.freq))
        # t = np.linspace(0, self.trajectory_time, n_steps)
        # # Random control points
        # scale = np.array([1.2, 1.2, 0.5])
        # waypoints = np.random.uniform(-1, 1, size=(self.sim.n_worlds, self.num_waypoints, 3)) * scale
        # waypoints = waypoints + 0.3*self.takeoff_pos + np.array([0.0, 0.0, 0.7]) # shift up in z direction
        # waypoints[:, 0, :] = self.takeoff_pos # set start point to takeoff position
        # spline = CubicSpline(np.linspace(0, self.trajectory_time, self.num_waypoints), waypoints, axis=1)
        # self.trajectories = spline(t)  # (n_worlds, n_steps, 3)

        # Update observation space
        spec = {k: v for k, v in self.single_observation_space.items()}
        spec["local_samples"] = spaces.Box(-np.inf, np.inf, shape=(3 * self.n_samples,))
        self.single_observation_space = spaces.Dict(spec)
        self.observation_space = batch_space(self.single_observation_space, self.sim.n_worlds)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        """Reset."""
        # # Create a random trajectory based on spline interpolation
        # t = np.linspace(0, self.trajectory_time, self.n_steps)
        # scale = np.array([1.2, 1.2, 0.5])
        # waypoints = np.random.uniform(-1, 1, size=(self.sim.n_worlds, self.num_waypoints, 3)) * scale
        # waypoints = waypoints + 0.3*self.takeoff_pos + np.array([0.0, 0.0, 0.7]) # shift up in z direction
        # waypoints[:, :3, :] = np.array([[-1.5, 1.0, 0.07],[-1.0, 0.55, 0.4],[0.3, 0.35, 0.7]]) # set first three waypoints
        # v0 = np.tile(np.array([[0.0, 0.0, 0.4]]), (self.sim.n_worlds, 1)) # takeoff velocity
        # spline = CubicSpline(np.linspace(0, self.trajectory_time, self.num_waypoints), waypoints, axis=1, bc_type=((1, v0), 'not-a-knot'))
        # self.trajectories = spline(t)  # (n_worlds, n_steps, 3)

        super().reset(seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options) # call jax reset function
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}

    def render(self):
        """Render."""
        idx = np.clip(self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1)
        next_trajectory = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx]
        draw_line(self.sim, self.trajectories[0, 0:-1:2, :], rgba=np.array([1,1,1,0.4]), start_size=2.0, end_size=2.0)
        draw_line(self.sim, next_trajectory[0], rgba=np.array([1,0,0,1]), start_size=3.0, end_size=3.0)
        draw_points(self.sim, next_trajectory[0], rgba=np.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.render()

    def obs(self) -> dict[str, Array]:
        """Observations."""
        obs = super().obs()
        idx = np.clip(self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1)
        dpos = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], idx] - self.sim.data.states.pos
        obs["local_samples"] = dpos.reshape(-1, 3 * self.n_samples)
        return obs

    def reward(self) -> Array:
        """Rewards."""
        obs = self.obs()
        pos = obs["pos"] # (num_envs, 3)
        goal = self.trajectories[np.arange(self.trajectories.shape[0])[:, None], self.steps][:, 0, :] # (num_envs, 3)
        # distance to next trajectory point
        norm_distance = jp.linalg.norm(pos - goal, axis=-1)
        reward = jp.exp(-2.0 * norm_distance) # encourage flying close to goal
        reward = jp.where(self.terminated(), -1.0, reward) # penalize drones that crash into the ground
        return reward

    def apply_action(self, action: Array):
        """Apply the commanded state action to the simulation."""
        action = action.reshape((self.sim.n_worlds, self.sim.n_drones, -1))
        if "action" in self.disturbances:
            key, subkey = jax.random.split(self.sim.data.core.rng_key)
            action += self.disturbances["action"](subkey, action.shape)
            self.sim.data = self.sim.data.replace(core=self.sim.data.core.replace(rng_key=key))
        match self.sim.control:
            case "attitude":
                self.sim.attitude_control(action)
            case "state":
                self.sim.state_control(action)
            case _:
                raise ValueError(f"Unsupported control mode: {self.sim.control}")

    @property
    def steps(self) -> Array:
        """The current step in the trajectory."""
        return self.sim.data.core.steps // (self.sim.freq // self.freq) - 1
    
    @staticmethod
    @jax.jit
    def _terminated(pos: Array) -> Array:
        lower_bounds = jp.array([-4.0, -4.0, -0.0])
        upper_bounds = jp.array([ 4.0,  4.0, 4.0])
        terminate = jp.any((pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1)
        return terminate

    def build_reset_randomization_fn(self, physics: str) -> Callable[[SimData, Array], SimData]:
        """Reset randomization."""
        # Spin up rotors to help takeoff
        def _reset_randomization_so_rpy(data: SimData, mask: Array) -> SimData:
            rotor_vel = 0.05 * jp.ones((data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1]))
            data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
            return data
        def _reset_randomization_first_principles(data: SimData, mask: Array) -> SimData:
            rotor_vel = 10000.0 * jp.ones((data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1]))
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