"""Jittable Drone Environment Implementation."""

import functools
import os
from typing import Callable, Literal

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
import scipy
from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from rl_diffsim.envs.drone_env_jittable import DroneJittableEnv, create_action_space


class RandTrajJittableEnv(DroneJittableEnv):
    """Jittable Random Trajectory Environment.

    This class defines a subclass of DroneJittableEnv that contains environment data and jittable functions,
    allowing for efficient execution with JAX's JIT compilation. Pass env as an argument to jitted functions.

    Attributes:
        num_envs: Number of parallel environments.
        max_episode_steps: Maximum number of steps per episode.
        obs_space: Observation space of the random trajectory environment.
        act_space: Action space of the random trajectory environment.
        reset_fn: Jittable reset function.
        step_fn: Jittable step function.
        obs_fn: Jittable observation extraction function.
        data: Simulation data structure.
    """

    # Random trajectory parameters
    trajectories: Array = struct.field(pytree_node=False)
    sample_offsets: Array = struct.field(pytree_node=False)

    # Non-jittable functions
    def render(self, world: int = 0, **kwargs: dict) -> None:
        """Override base class render to show random trajectory."""
        idx = jp.clip(
            self.steps + self.sample_offsets[None, ...], 0, self.trajectories[0].shape[0] - 1
        )
        next_trajectory = self.trajectories[jp.arange(self.trajectories.shape[0])[:, None], idx]
        trajectories = np.array(self.trajectories)
        next_trajectory = np.array(next_trajectory)
        draw_line(
            self.sim,
            trajectories[world, 0:-1:2, :],
            rgba=jp.array([1, 1, 1, 0.4]),
            start_size=2.0,
            end_size=2.0,
        )
        draw_line(
            self.sim,
            next_trajectory[world],
            rgba=jp.array([1, 0, 0, 1]),
            start_size=3.0,
            end_size=3.0,
        )
        draw_points(self.sim, next_trajectory[world], rgba=jp.array([1.0, 0, 0, 1]), size=0.01)
        self.sim.data = self.data
        return self.sim.render(world=world, **kwargs)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Physics = Physics.so_rpy_rotor_drag,
        control: Control | str = Control.default,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        device: str = "cpu",
        num_waypoints: int = 7,
        n_samples: int = 10,
        trajectory_time: float = 10.0,
        samples_dt: float = 0.1,
        reset_rotor: bool = False,
        reset_randomization: Callable[[SimData, Array], SimData] | None = None,
    ) -> "RandTrajJittableEnv":
        """Create a jittable drone environment without render support.

        Args:
            num_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            control: Control interface to use.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation.
            sim_freq: Simulation frequency.
            device: Device to use for the simulation.
            num_waypoints: Number of random sampled waypoints for the random trajectory.
            n_samples: Number of next trajectory points to sample for observations.
            trajectory_time: Total time for completing the random trajectory in seconds.
            samples_dt: Time between trajectory sample points in seconds.
            reset_rotor: Whether to reset rotor speeds on environment reset.
            reset_randomization: A function that randomizes the initial state of the simulation. If
                None, the default randomization for pos and vel is used.

        Returns:
            An instance of RandTrajJittableEnv with jittable functions and data.
        """
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=1,
            drone_model=drone_model,
            physics=physics,
            control=control if control in [c.value for c in Control] else Control.default,
            device=device,
            freq=sim_freq,
        )

        # Modify the step pipeline if needed
        if control == "rotor_vel":
            sim.step_pipeline = sim.step_pipeline[2:]  # remove all controllers
            sim.build_step_fn()

        # Override reset randomization function
        def build_reset_rotor_fn(physics: str) -> Callable[[SimData, Array], SimData]:
            """Reset rotor."""

            # Spin up rotors to help takeoff
            def _reset_rotor_so_rpy(data: SimData, mask: Array) -> SimData:
                rotor_vel = 0.05 * jp.ones(
                    (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
                )
                data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
                return data

            def _reset_rotor_first_principles(data: SimData, mask: Array) -> SimData:
                rotor_vel = 10000.0 * jp.ones(
                    (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
                )
                data = data.replace(states=leaf_replace(data.states, mask, rotor_vel=rotor_vel))
                return data

            def _no_reset_rotor(data: SimData, mask: Array) -> SimData:
                return data

            match physics:
                case "first_principles":
                    return _reset_rotor_first_principles
                case "so_rpy" | "so_rpy_rotor" | "so_rpy_rotor_drag":
                    return _reset_rotor_so_rpy
                case "no_reset_rotor":
                    return _no_reset_rotor

        reset_rotor_randomization = build_reset_rotor_fn(
            physics if reset_rotor else "no_reset_rotor"
        )

        if reset_randomization is None:

            def _reset_randomization(
                data: SimData, mask: Array, pmin: Array, pmax: Array, vmin: float, vmax: float
            ) -> SimData:
                shape = (data.core.n_worlds, data.core.n_drones, 3)
                key, pos_key, vel_key = jax.random.split(data.core.rng_key, 3)
                data = data.replace(core=data.core.replace(rng_key=key))
                original_pos = data.states.pos
                pos = jax.random.uniform(
                    key=pos_key, shape=shape, minval=original_pos - pmin, maxval=original_pos + pmax
                )
                vel = jax.random.uniform(key=vel_key, shape=shape, minval=vmin, maxval=vmax)
                data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel))
                return data

            reset_randomization = functools.partial(
                _reset_randomization, pmin=-0.1, pmax=0.1, vmin=-0.5, vmax=0.5
            )

        sim.reset_pipeline += (reset_randomization, reset_rotor_randomization)
        sim.build_reset_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(sim.control, sim.drone_model)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = spaces.Dict(
            {
                "pos": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "quat": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "ang_vel": spaces.Box(-np.inf, np.inf, shape=(3,)),
            }
        )
        n_substeps = sim.freq // freq

        # Create a random trajectory based on spline interpolation
        sample_offsets = jp.array(jp.arange(n_samples) * freq * samples_dt, dtype=int)
        n_steps = int(np.ceil(trajectory_time * freq))
        t = np.linspace(0, trajectory_time, n_steps)
        # Random control points
        scale = np.array([0.7, 0.7, 0.5])
        waypoints = np.random.uniform(-1, 1, size=(sim.n_worlds, num_waypoints, 3)) * scale
        waypoints[:, 0, :] = np.zeros(3)  # set start point to [0, 0, 0]
        waypoints[:, -1, :] = np.zeros(3)  # set end point to [0, 0, 0]
        waypoints = waypoints + np.array([0, 0, 2])  # shift up in z direction
        spline = scipy.interpolate.CubicSpline(
            np.linspace(0, trajectory_time, num_waypoints), waypoints, axis=1
        )
        trajectories = spline(t)  # (n_worlds, n_steps, 3)
        trajectories = jp.array(trajectories, dtype=jp.float32)

        # Set takeoff position and build default reset position
        takeoff_pos = trajectories[:, :1, :]
        sim.data = sim.data.replace(states=sim.data.states.replace(pos=takeoff_pos))
        sim.build_default_data()

        # Update observation space
        spec = {k: v for k, v in single_observation_space.items()}
        # use Python floats for infinity (compatible with gym spaces)
        spec["local_samples"] = spaces.Box(-float("inf"), float("inf"), shape=(3 * n_samples,))
        single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _sanitize_action_STE(action: Array, low: Array, high: Array) -> Array:
            action_clipped = jp.clip(action, low, high)
            action = action + jax.lax.stop_gradient(action_clipped - action)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _aux_obs(
            trajectories: Array, steps: Array, pos: Array, sample_offsets: Array
        ) -> dict[str, Array]:
            """Static method version of obs for jitting."""
            idx = jp.clip(steps + sample_offsets[None, ...], 0, trajectories.shape[1] - 1)
            dpos = trajectories[jp.arange(trajectories.shape[0])[:, None], idx] - pos
            local_samples = dpos.reshape(dpos.shape[0], dpos.shape[1] * dpos.shape[2])
            return local_samples

        def _obs(data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            steps = data.core.steps // (sim.freq // freq) - 1
            obs["local_samples"] = _aux_obs(trajectories, steps, data.states.pos, sample_offsets)
            return obs

        def _reset(
            env: "RandTrajJittableEnv", *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            _marked_for_reset = env._marked_for_reset
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(data=data, _marked_for_reset=_marked_for_reset), (_obs(data), {})

        def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
            # distance to next trajectory point
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-2.0 * norm_distance)
            reward = jp.where(terminated, -1.0, reward)
            return reward

        def _terminated(pos: Array) -> Array:
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            terminate = jp.any(
                (pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1
            )
            return terminate

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _step(
            env: "RandTrajJittableEnv", action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data, _marked_for_reset = env.data, env._marked_for_reset
            # 1. apply action: only attitude control
            low, high = action_space.low, action_space.high
            action = _sanitize_action(action, low, high)
            data = data.replace(
                controls=data.controls.replace(
                    attitude=data.controls.attitude.replace(staged_cmd=action)
                )
            )
            # 2. step sim for n_substeps
            data = sim._step(data, n_substeps)
            # 3. handle autoreset & update mask
            data = sim._reset(data, sim.default_data, _marked_for_reset)
            sim_time = data.core.steps / data.core.freq
            terminated, truncated = (
                _terminated(data.states.pos),
                _truncated(sim_time[..., 0], max_episode_time),
            )
            _marked_for_reset = _done(terminated, truncated)
            # 4. construct obs & rewards
            steps = data.core.steps // (sim.freq // freq) - 1
            pos = data.states.pos[:, 0, :]
            goal = trajectories[jp.arange(trajectories.shape[0])[:, None], steps][:, 0, :]

            return env.replace(data=data, steps=steps, _marked_for_reset=_marked_for_reset), (
                _obs(data),
                _reward(terminated, pos, goal),
                terminated,
                truncated,
                {},
            )

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        return cls(
            sim=sim,
            num_envs=num_envs,
            max_episode_time=max_episode_time,
            physics=physics,
            control=control,
            drone_model=drone_model,
            freq=freq,
            device=device,
            single_action_space=single_action_space,
            action_space=action_space,
            single_observation_space=single_observation_space,
            observation_space=observation_space,
            n_substeps=n_substeps,
            trajectories=trajectories,
            sample_offsets=sample_offsets,
            data=sim.data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )


if __name__ == "__main__":
    import time

    """Test the jittable drone environment implementation."""
    # Create the jittable environment
    env = RandTrajJittableEnv.create(
        num_envs=1024,
        max_episode_time=10.0,
        physics=Physics.so_rpy_rotor_drag,
        drone_model="cf21B_500",
        freq=500,
        device="gpu",
        n_samples=10,
        trajectory_time=10.0,
        samples_dt=0.1,
        reset_rotor=True,
    )

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Trajectories:", env.trajectories.shape)
    print("Initial Traj Obs:", obs["local_samples"].shape)

    def step_once(env: RandTrajJittableEnv, _) -> tuple[RandTrajJittableEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(
        env: RandTrajJittableEnv, num_steps: int
    ) -> tuple[RandTrajJittableEnv, tuple[Array, Array]]:
        """Rollout for multiple steps using lax.scan."""
        env, (pos_traj, vel_traj) = jax.lax.scan(step_once, env, xs=None, length=num_steps)
        return env, (pos_traj, vel_traj)

    rollout_jit = jax.jit(rollout, static_argnames=("num_steps",))

    # Warm-up rollout
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Warm-up rollout took {end_time - start_time:.4f} seconds")

    # After jitting
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 8)
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")

    print("\nPos trajectory shape:", pos_traj.shape)
    print("Vel trajectory shape:", vel_traj.shape)

    print("obs.trajectories[0, :5, :]:", env.trajectories[0, :5, :])
