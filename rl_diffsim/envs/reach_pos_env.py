"""Drone Environment Implementation."""

import functools
from typing import Callable, Literal

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.sim.visualize import draw_points
from crazyflow.utils import leaf_replace
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array

from rl_diffsim.envs.drone_env import DroneEnv, create_action_space


class ReachPosEnv(DroneEnv):
    """Reach Position Environment.

    This class defines a subclass of DroneEnv that contains environment data and jittable functions,
    allowing for efficient execution with JAX's JIT compilation. Pass env as an argument to jitted functions.

    Attributes:
        num_envs: Number of parallel environments.
        max_episode_steps: Maximum number of steps per episode.
        obs_space: Observation space of the figure-eight environment.
        act_space: Action space of the figure-eight environment.
        reset_fn:  reset function.
        step_fn:  step function.
        obs_fn:  observation extraction function.
        data: Simulation data structure.
    """

    # Reach position specific attributes
    goal_pos: Array = struct.field(pytree_node=True)

    # Non-jittable functions
    def render(self, world: int = 0) -> None:
        """Override base class render to show reach position."""
        draw_points(self.sim, self.goal_pos[None, world], rgba=jp.array([1.0, 0, 0, 1.0]), size=0.01)
        self.sim.data = self.data
        self.sim.render(world=world)

    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        max_episode_time: float = 5.0,
        physics: Literal["so_rpy_rotor_drag", "first_principles"] | Physics = Physics.first_principles,
        control: Control | str = Control.default,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        device: str = "cpu",
        reset_rotor: bool = False,
        pos_min: Array = jp.array([-1.0, -1.0, 1.0]),
        pos_max: Array = jp.array([1.0, 1.0, 2.0]),
        goal_pmin: Array = jp.array([-1.0, -1.0, 0.5]),
        goal_pmax: Array = jp.array([1.0, 1.0, 1.5]),
        ang_vel_min: Array = jp.zeros(3),
        ang_vel_max: Array = jp.zeros(3),
        vel_min: float = -1.0,
        vel_max: float = 1.0,
    ) -> "ReachPosEnv":
        """Create a jittable drone environment without render support.

        Args:
            num_envs: Number of parallel environments.
            max_episode_time: Maximum episode time in seconds.
            physics: Physics backend to use.
            control: Control interface.
            drone_model: Drone model of the environment.
            freq: Frequency of the simulation.
            sim_freq: Simulation frequency.
            device: Device to use for the simulation.
            n_samples: Number of next trajectory points to sample for observations.
            trajectory_time: Total time for completing the figure-eight trajectory in seconds.
            samples_dt: Time between trajectory sample points in seconds.
            reset_rotor: Whether to reset rotor speeds on environment reset.
            pos_min: Minimum position for randomization on reset.
            pos_max: Maximum position for randomization on reset.
            goal_pmin: Minimum goal position for random goal sampling.
            goal_pmax: Maximum goal position for random goal sampling.
            ang_vel_min: Minimum initial angular velocity for randomization.
            ang_vel_max: Maximum initial angular velocity for randomization.
            vel_min: Minimum velocity for randomization on reset.
            vel_max: Maximum velocity for randomization on reset.

        Returns:
            An instance of ReachPosEnv with jittable functions and data.
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
                rotor_vel = 18967.0 * jp.ones(
                    (data.core.n_worlds, data.core.n_drones, data.states.rotor_vel.shape[-1])
                )  # TODO: calculate hover rotor velocity based on drone parameters
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

        def _reset_randomization(
            data: SimData,
            mask: Array,
            pmin: Array,
            pmax: Array,
            vmin: float,
            vmax: float,
            wmin: Array,
            wmax: Array,
        ) -> SimData:
            shape = (data.core.n_worlds, data.core.n_drones, 3)
            key, pos_key, vel_key, ang_vel_key = jax.random.split(data.core.rng_key, 4)
            data = data.replace(core=data.core.replace(rng_key=key))
            pos = jax.random.uniform(key=pos_key, shape=shape, minval=pmin, maxval=pmax)
            vel = jax.random.uniform(key=vel_key, shape=shape, minval=vmin, maxval=vmax)
            ang_vel = jax.random.uniform(key=ang_vel_key, shape=shape, minval=wmin, maxval=wmax)
            data = data.replace(states=leaf_replace(data.states, mask, pos=pos, vel=vel, ang_vel=ang_vel))
            return data

        reset_rotor_randomization = build_reset_rotor_fn(physics if reset_rotor else "no_reset_rotor")

        reset_randomization = functools.partial(
            _reset_randomization,
            pmin=pos_min,
            pmax=pos_max,
            vmin=vel_min,
            vmax=vel_max,
            wmin=ang_vel_min,
            wmax=ang_vel_max,
        )
        sim.reset_pipeline += (reset_randomization, reset_rotor_randomization)
        sim.build_reset_fn()

        # Prepare immutable constants
        single_action_space = create_action_space(control, sim.drone_model)
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

        # Update observation space
        # spec = {k: v for k, v in single_observation_space.items()}
        # spec["difference_to_goal"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        # single_observation_space = spaces.Dict(spec)
        observation_space = batch_space(single_observation_space, sim.n_worlds)

        # Build jittable functions
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _sanitize_action_STE(action: Array, low: Array, high: Array) -> Array:
            action_clipped = jp.clip(action, low, high)
            action = action + jax.lax.stop_gradient(action_clipped - action)
            return jp.array(action, device=jax_device).reshape((num_envs, 1, -1))

        def _obs(goal_pos: Array, data: SimData) -> dict[str, Array]:
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
            }
            # obs["difference_to_goal"] = goal_pos - data.states.pos[:, 0, :] # legacy approach
            obs["pos"] = data.states.pos[:, 0, :] - goal_pos  # agent only sees relative position
            return obs

        def _sample_goal(key: Array, goal: Array, mask: Array | None) -> Array:
            new_goal = jax.random.uniform(key, shape=goal.shape, minval=goal_pmin, maxval=goal_pmax)
            if mask is not None:
                new_goal = jp.where(mask[..., None], new_goal, goal)
            return new_goal

        def _reset(
            env: "ReachPosEnv", *, seed: int | None = None, options: dict | None = None
        ) -> tuple[tuple[SimData, Array, Array], tuple[dict[str, Array], dict]]:
            data = env.data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            # 1. re-sample goal position
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, None)
            # 2. reset sim
            data = sim._reset(data, sim.default_data, None)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(data=data, _marked_for_reset=_marked_for_reset, goal_pos=goal_pos), (
                _obs(goal_pos, data),
                {},
            )

        def _reward(terminated: Array, pos: Array, goal: Array) -> Array:
            # distance to next trajectory point
            norm_distance = jp.linalg.norm(pos - goal, axis=-1)
            reward = jp.exp(-6.0 * norm_distance)
            reward = jp.where(terminated, -1.0, reward)
            return reward

        def _terminated(pos: Array) -> Array:
            lower_bounds = jp.array([-4.0, -4.0, 0.0])
            upper_bounds = jp.array([4.0, 4.0, 4.0])
            terminate = jp.any((pos[:, 0, :] < lower_bounds) | (pos[:, 0, :] > upper_bounds), axis=-1)
            return terminate

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _apply_action(data: SimData, action: Array, control: Control) -> SimData:
            low, high = action_space.low, action_space.high
            action = _sanitize_action_STE(action, low, high)
            match control:
                case Control.state:
                    raise NotImplementedError("State control currently not supported")
                case Control.attitude:
                    data = data.replace(
                        controls=data.controls.replace(
                            attitude=data.controls.attitude.replace(staged_cmd=action)
                        )
                    )
                case Control.force_torque:
                    data = data.replace(
                        controls=data.controls.replace(
                            force_torque=data.controls.force_torque.replace(staged_cmd=action)
                        )
                    )
                case "rotor_vel":
                    data = data.replace(controls=data.controls.replace(rotor_vel=action))
                case _:
                    raise ValueError(f"Invalid control type {control}")
            return data

        _apply_action = functools.partial(_apply_action, control=control)

        def _step(
            env: "ReachPosEnv", action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data, _marked_for_reset = env.data, env._marked_for_reset
            # 1. apply action: only attitude control
            data = _apply_action(data, action)
            # 2. step sim for n_substeps
            data = sim._step(data, n_substeps)
            # 3. handle autoreset & update mask
            rng_key, subkey = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=rng_key))
            goal_pos = _sample_goal(subkey, env.goal_pos, _marked_for_reset)
            data = sim._reset(data, sim.default_data, _marked_for_reset)
            sim_time = data.core.steps / data.core.freq
            terminated, truncated = (
                _terminated(data.states.pos),
                _truncated(sim_time[..., 0], max_episode_time),
            )
            _marked_for_reset = _done(terminated, truncated)
            # 4. construct obs & rewards
            steps = data.core.steps // (sim.freq // freq)
            pos = data.states.pos[:, 0, :]
            goal = goal_pos

            return env.replace(
                data=data, steps=steps, _marked_for_reset=_marked_for_reset, goal_pos=goal_pos
            ), (_obs(goal_pos, data), _reward(terminated, pos, goal), terminated, truncated, {})

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
            goal_pos=jp.zeros((sim.n_worlds, 3), dtype=jp.float32, device=jax_device),
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
    env = ReachPosEnv.create(
        num_envs=1024,
        max_episode_time=10.0,
        physics=Physics.first_principles,
        control="rotor_vel",
        drone_model="cf21B_500",
        freq=500,
        device="gpu",
        reset_rotor=True,
    )

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("obs.goal_pos[:5, :]:", env.goal_pos[:5, :])
    print("Goal Pos:", env.goal_pos.shape)
    print("Initial Goal Obs:", obs["difference_to_goal"].shape)

    def step_once(env: ReachPosEnv, _) -> tuple[ReachPosEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(env: ReachPosEnv, num_steps: int) -> tuple[ReachPosEnv, tuple[Array, Array]]:
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

    # test rendering
    print("Action_space:", env.action_space.low[0], env.action_space.high[0])
    env, (obs, info) = env.reset(env)
    for step in range(500):
        base_action = jp.array([15000, 20000, 15000, 20000], dtype=jp.float32)  # fixed action
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)
        env, _ = env.step(env, action)
        env.render()
