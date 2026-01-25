"""Core environment for drone racing simulations.

This module provides the shared logic for simulating drone racing environments. It defines a core
environment class that wraps our drone simulation, drone control, gate tracking, and collision
detection. The module serves as the base for both single-drone and multi-drone racing environments.

The environment is designed to be configurable, supporting:

* Different control modes (state or attitude)
* Customizable tracks with gates and obstacles
* Various randomization options for robust policy training
* Disturbance modeling for realistic flight conditions
* Vectorized execution for parallel training

This module is primarily used as a base for the higher-level environments in
:mod:`~lsy_drone_racing.envs.drone_race` and :mod:`~lsy_drone_racing.envs.multi_drone_race`,
which provide Gymnasium-compatible interfaces for reinforcement learning, MPC and other control
techniques.
"""

from __future__ import annotations

import copy as copy
import functools
import logging
from typing import TYPE_CHECKING, Callable, Literal

import flax.struct as struct
import jax
import jax.numpy as jp
import numpy as np
from crazyflow import Control, Physics
from crazyflow.sim import Sim
from crazyflow.sim.sim import sync_sim2mjx, use_box_collision
from crazyflow.utils import leaf_replace
from flax.struct import dataclass
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from ml_collections import ConfigDict

from rl_diffsim.envs.drone_env import DroneEnv, create_action_space
from rl_diffsim.envs.race_utils import (
    build_track_randomization_fn,
    gate_passed,
    load_contact_masks,
    load_track,
    rng_spec2fn,
    setup_sim,
)

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array, Device
    from mujoco.mjx import Data
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# region RaceData
@dataclass
class RaceData:
    """Struct holding the data of all auxiliary variables for the environment.

    This dataclass stores the dynamic and static state of the environment that is not directly
    part of the physics simulation. It includes information about gate progress, drone status,
    and environment boundaries. Static variables are initialized once and do not change during the
    episode.

    Args:
        contact_masks: Masks for contact detection between drones and objects
        pos_limit_low: Lower position limits for the environment
        pos_limit_high: Upper position limits for the environment
        gate_mj_ids: MuJoCo IDs for the gates
        obstacle_mj_ids: MuJoCo IDs for the obstacles
        max_episode_steps: Maximum number of steps per episode
        sensor_range: Range at which drones can detect gates and obstacles
        track: Configuration dictionary for the race track
        gates: Configuration dictionary for gates
        obstacles: Configuration dictionary for obstacles
        drone: Configuration dictionary for drone
        disturbances: Configuration dictionary for disturbances
        target_gate: Current target gate index for each drone in each environment
        gates_visited: Boolean flags indicating which gates have been visited by each drone
        obstacles_visited: Boolean flags indicating which obstacles have been detected
        last_drone_pos: Previous positions of drones, used for gate passing detection
        disabled_drones: Flags indicating which drones have crashed or are otherwise disabled
    """

    # Static variables
    sensor_range: float = struct.field(pytree_node=False)
    n_gates: int = struct.field(pytree_node=False)
    n_obstacles: int = struct.field(pytree_node=False)
    contact_masks: Array
    pos_limit_low: Array
    pos_limit_high: Array
    gate_mj_ids: Array
    obstacle_mj_ids: Array
    max_episode_steps: Array

    # Dynamic variables
    target_gate: Array
    gates_visited: Array
    obstacles_visited: Array
    last_drone_pos: Array
    disabled_drones: Array
    gate_size: float

    @classmethod
    def create(
        cls,
        n_envs: int,
        n_drones: int,
        n_gates: int,
        n_obstacles: int,
        contact_masks: Array,
        gate_mj_ids: Array,
        obstacle_mj_ids: Array,
        max_episode_steps: int,
        sensor_range: float,
        pos_limit_low: Array,
        pos_limit_high: Array,
        device: Device,
        gate_size: float,
    ) -> RaceData:
        """Create a new environment data struct with default values."""
        return cls(
            target_gate=jp.zeros((n_envs, n_drones), dtype=int, device=device),
            gates_visited=jp.zeros((n_envs, n_drones, n_gates), dtype=bool, device=device),
            obstacles_visited=jp.zeros((n_envs, n_drones, n_obstacles), dtype=bool, device=device),
            last_drone_pos=jp.zeros((n_envs, n_drones, 3), dtype=np.float32, device=device),
            disabled_drones=jp.zeros((n_envs, n_drones), dtype=bool, device=device),
            contact_masks=jp.array(contact_masks, dtype=bool, device=device),
            pos_limit_low=jp.array(pos_limit_low, dtype=np.float32, device=device),
            pos_limit_high=jp.array(pos_limit_high, dtype=np.float32, device=device),
            gate_mj_ids=jp.array(gate_mj_ids, dtype=int, device=device),
            obstacle_mj_ids=jp.array(obstacle_mj_ids, dtype=int, device=device),
            max_episode_steps=jp.array([max_episode_steps], dtype=int, device=device),
            sensor_range=sensor_range,
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            gate_size=gate_size,
        )


def create_observation_space(n_gates: int, n_obstacles: int) -> spaces.Dict:
    """Create the observation space for the environment.

    The observation space is a dictionary containing the drone state, gate information,
    and obstacle information.

    Args:
        n_gates: Number of gates in the environment.
        n_obstacles: Number of obstacles in the environment.
    """
    obs_spec = {
        "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "quat": spaces.Box(low=-1, high=1, shape=(4,)),
        "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        "target_gate": spaces.Discrete(n_gates, start=-1),
        "gates_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_gates, 3)),
        "gates_quat": spaces.Box(low=-1, high=1, shape=(n_gates, 4)),
        "gates_visited": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
        "obstacles_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 3)),
        "obstacles_visited": spaces.Box(low=0, high=1, shape=(n_obstacles,), dtype=bool),
    }
    return spaces.Dict(obs_spec)


# region DroneRaceEnv
class DroneRaceEnv(DroneEnv):
    """Drone Racing Environment.

    This class defines a subclass of DroneEnv that contains environment data and jittable functions,
    allowing for efficient execution with JAX's JIT compilation. Pass env as an argument to jitted functions.

    This environment simulates a drone racing scenario where a single drone navigates through a
    series of gates in a predefined track. It supports various configuration options for
    randomization, disturbances, and physics models.
    """

    # Constant parameters

    # Racing data
    mjx_data: Data = struct.field(pytree_node=True)
    race_data: RaceData = struct.field(pytree_node=True)

    # Non-jittable functions
    def render(self, world: int = 0) -> None:
        """Override base class render to show racing trajectory."""
        self.sim.data = self.data
        self.sim.mjx_data = self.mjx_data
        self.sim.render(world=world)

    # region Create
    @classmethod
    def create(
        cls,
        num_envs: int = 1,
        num_drones: int = 1,
        max_episode_time: float = 10.0,
        physics: Literal["so_rpy_rotor_drag", "first_principles"]
        | Physics = Physics.first_principles,
        control: Control | str = Control.default,
        drone_model: str = "cf21B_500",
        freq: int = 500,
        sim_freq: int = 500,
        state_freq: int = 100,
        attitude_freq: int = 500,
        force_torque_freq: int = 500,
        device: Literal["cpu", "gpu"] = "cpu",
        reset_rotor: bool = False,
        sensor_range: float = 0.7,
        track: ConfigDict | None = None,
        disturbances: ConfigDict | None = None,
        randomizations: ConfigDict | None = None,
        check_contacts: bool = False,
        end_on_gate_bypass: bool = False,
    ) -> DroneRaceEnv:
        """Create a drone racing environment.

        Args:
            num_envs: The number of environments to run in parallel.
            num_drones: The number of drones per environment.
            max_episode_time: The time horizon after which episodes are truncated (s).
            physics: The crazyflow physics simulation model.
            control: Control interface to use.
            drone_model: Drone model of the environment.
            freq: The frequency at which the environment is run.
            sim_freq: Simulation frequency.
            state_freq: The frequency of the state controller.
            attitude_freq: The frequency of the attitude controller.
            force_torque_freq: The frequency of the force/torque controller.
            device: The device used for the environment and the simulation.
            reset_rotor: Whether to reset rotor velocities on episode reset.
            sensor_range: Sensor range for gate and obstacle detection (m).
            track: Track configuration dictionary.
            disturbances: Disturbance configuration dictionary.
            randomizations: Randomization configuration dictionary.
            check_contacts: Whether to disable drones when contacts occur.
            end_on_gate_bypass: Whether to disable drones that fail to pass a gate.

        Returns:
            An instance of DroneRaceEnv with jittable functions and data.
        """
        # region Init
        # Initialize the simulation
        jax_device = jax.devices(device)[0]
        sim = Sim(
            n_worlds=num_envs,
            n_drones=num_drones,
            drone_model=drone_model,
            device=device,
            physics=physics,
            freq=sim_freq,
            state_freq=state_freq,
            attitude_freq=attitude_freq,
            force_torque_freq=force_torque_freq,
        )
        use_box_collision(sim, check_contacts)
        n_substeps = sim.freq // freq

        # Modify the step pipeline if needed
        if control == "rotor_vel":
            sim.step_pipeline = sim.step_pipeline[2:]  # remove all controllers
            sim.build_step_fn()

        if check_contacts is False:
            sim.step_pipeline = sim.step_pipeline[:-1]  # remove clip_floor_pos
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
                rotor_vel = 15900.0 * jp.ones(
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
        sim.reset_pipeline += (reset_rotor_randomization,)

        # Env settings
        gates, obstacles, drone = load_track(track)
        specs = {} if disturbances is None else disturbances
        disturbances = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}
        specs = {} if randomizations is None else randomizations
        randomizations = {mode: rng_spec2fn(spec) for mode, spec in specs.items()}

        # Load the track into the simulation and compile the reset and step functions with hooks
        setup_sim(sim, gates, obstacles, drone, disturbances, randomizations)

        # Create the environment data struct.
        n_gates, n_obstacles = len(track.gates), len(track.obstacles)
        contact_masks = load_contact_masks(sim)
        m = sim.mj_model
        gate_ids = [int(m.body(f"gate:{i}").mocapid.squeeze()) for i in range(n_gates)]
        obstacle_ids = [int(m.body(f"obstacle:{i}").mocapid.squeeze()) for i in range(n_obstacles)]
        race_data = RaceData.create(
            n_envs=num_envs,
            n_drones=num_drones,
            n_gates=n_gates,
            n_obstacles=n_obstacles,
            contact_masks=contact_masks,
            gate_mj_ids=gate_ids,
            obstacle_mj_ids=obstacle_ids,
            max_episode_steps=max_episode_time * freq,
            sensor_range=sensor_range,
            pos_limit_low=[-3, -3, -1],
            pos_limit_high=[3, 3, 2.5],
            device=jax_device,
            gate_size=0.45,
        )
        _randomize_track: Callable = build_track_randomization_fn(
            randomizations, gate_ids, obstacle_ids
        )

        # Create action/observation space
        single_action_space = create_action_space(control, sim.drone_model)
        action_space = batch_space(single_action_space, sim.n_worlds)
        single_observation_space = create_observation_space(n_gates, n_obstacles)
        observation_space = batch_space(single_observation_space, sim.n_worlds)  # SELF

        # Build jittable functions
        # region Action
        def _sanitize_action(action: Array, low: Array, high: Array) -> Array:
            action = jp.clip(action, low, high)
            return jp.array(action, device=jax_device).reshape((num_envs, num_drones, -1))

        def _sanitize_action_STE(action: Array, low: Array, high: Array) -> Array:
            action_clipped = jp.clip(action, low, high)
            action = action + jax.lax.stop_gradient(action_clipped - action)
            return jp.array(action, device=jax_device).reshape((num_envs, num_drones, -1))

        def _apply_action(
            data: SimData, action: Array, control: Control, disturbances: dict
        ) -> SimData:
            low, high = action_space.low, action_space.high
            action = _sanitize_action(action, low, high)
            if "action" in disturbances:
                key, subkey = jax.random.split(data.core.rng_key)
                action += disturbances["action"](subkey, action.shape)
                data = data.replace(core=data.core.replace(rng_key=key))
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

        _apply_action = functools.partial(_apply_action, control=control, disturbances=disturbances)

        # region Obs
        def _race_obs(
            mocap_pos: Array,
            mocap_quat: Array,
            gates_visited: Array,
            gate_mocap_ids: Array,
            nominal_gate_pos: NDArray,
            nominal_gate_quat: NDArray,
            obstacles_visited: Array,
            obstacle_mocap_ids: Array,
            nominal_obstacle_pos: NDArray,
        ) -> tuple[Array, Array]:
            """Get the nominal or real gate positions and orientations depending on the sensor range."""
            mask, real_pos = gates_visited[..., None], mocap_pos[:, gate_mocap_ids]
            real_quat = mocap_quat[:, gate_mocap_ids][..., [1, 2, 3, 0]]
            gates_pos = jp.where(mask, real_pos[:, None], nominal_gate_pos[None, None])
            gates_quat = jp.where(mask, real_quat[:, None], nominal_gate_quat[None, None])
            mask, real_pos = obstacles_visited[..., None], mocap_pos[:, obstacle_mocap_ids]
            obstacles_pos = jp.where(mask, real_pos[:, None], nominal_obstacle_pos[None, None])
            return gates_pos, gates_quat, obstacles_pos

        def _obs(data: SimData, mjx_data: Data, race_data: RaceData) -> dict[str, Array]:
            """Return the observation of the environment."""
            gates_pos, gates_quat, obstacles_pos = _race_obs(
                mjx_data.mocap_pos,
                mjx_data.mocap_quat,
                race_data.gates_visited,
                race_data.gate_mj_ids,
                gates["nominal_pos"],
                gates["nominal_quat"],
                race_data.obstacles_visited,
                race_data.obstacle_mj_ids,
                obstacles["nominal_pos"],
            )
            obs = {
                "pos": data.states.pos[:, 0, :],
                "quat": data.states.quat[:, 0, :],
                "vel": data.states.vel[:, 0, :],
                "ang_vel": data.states.ang_vel[:, 0, :],
                "target_gate": race_data.target_gate[:, 0],
                "gates_pos": gates_pos[:, 0, :],
                "gates_quat": gates_quat[:, 0, :],
                "gates_visited": race_data.gates_visited[:, 0, :],
                "obstacles_pos": obstacles_pos[:, 0, :],
                "obstacles_visited": race_data.obstacles_visited[:, 0],
            }
            return obs

        def _reward(data: Sim, race_data: RaceData) -> Array:
            return jp.zeros((num_envs,), dtype=jp.float32, device=jax_device)

        def _terminated(race_data: RaceData) -> Array:
            return jp.any(race_data.disabled_drones, axis=-1)  # any drone crashed

        def _truncated(time: Array, max_episode_time: float) -> Array:
            return time >= max_episode_time

        def _done(terminated: Array, truncated: Array) -> Array:
            return terminated | truncated

        def _info() -> dict:
            return {}

        # region Reset
        def _reset_race_data(
            race_data: RaceData, drone_pos: Array, mocap_pos: Array, mask: Array | None = None
        ) -> RaceData:
            """Reset auxiliary variables of the environment data."""
            mask = jp.ones((num_envs,), dtype=bool) if mask is None else mask
            target_gate = jp.where(mask[..., None], 0, race_data.target_gate)
            last_drone_pos = jp.where(mask[..., None, None], drone_pos, race_data.last_drone_pos)
            disabled_drones = jp.where(mask[..., None], False, race_data.disabled_drones)
            # Check which gates are in range of the drone
            gates_pos = mocap_pos[:, race_data.gate_mj_ids]
            dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
            gates_visited = jp.linalg.norm(dpos, axis=-1) < race_data.sensor_range
            gates_visited = jp.where(mask[..., None, None], gates_visited, race_data.gates_visited)
            # And which obstacles are in range
            obstacles_pos = mocap_pos[:, race_data.obstacle_mj_ids]
            dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
            obstacles_visited = jp.linalg.norm(dpos, axis=-1) < race_data.sensor_range
            obstacles_visited = jp.where(
                mask[..., None, None], obstacles_visited, race_data.obstacles_visited
            )
            return race_data.replace(
                target_gate=target_gate,
                last_drone_pos=last_drone_pos,
                disabled_drones=disabled_drones,
                gates_visited=gates_visited,
                obstacles_visited=obstacles_visited,
            )

        def _reset_data(
            data: SimData, mjx_data: Data, race_data: RaceData, mask: Array | None = None
        ) -> tuple[SimData, Data, RaceData]:
            """Reset all data and apply randomization."""
            data = sim._reset(data, sim.default_data, mask)
            key, track_key = jax.random.split(data.core.rng_key)
            data = data.replace(core=data.core.replace(rng_key=key))
            mjx_data = _randomize_track(
                mjx_data,
                mask,
                gates["nominal_pos"],
                gates["nominal_quat"],
                obstacles["nominal_pos"],
                track_key,
            )
            race_data = _reset_race_data(race_data, data.states.pos, mjx_data.mocap_pos, mask)
            return data, mjx_data, race_data

        def _reset(
            env: DroneRaceEnv, *, seed: int | None = None, options: dict | None = None
        ) -> tuple[DroneRaceEnv, tuple[dict, dict]]:
            data, mjx_data, race_data = env.data, env.mjx_data, env.race_data
            if seed is not None:
                rng_key = jax.device_put(jax.random.key(seed), jax_device)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data, mjx_data, race_data = _reset_data(data, mjx_data, race_data)
            _marked_for_reset = env._marked_for_reset.at[...].set(False)
            return env.replace(
                data=data,
                mjx_data=mjx_data,
                race_data=race_data,
                _marked_for_reset=_marked_for_reset,
            ), (_obs(data, mjx_data, race_data), {})

        # region Step
        def _contacts(data: SimData, mjx_data: Data) -> Array:
            # A pure contact detection function
            data, mjx_data = sync_sim2mjx(data, mjx_data, sim.mjx_model)
            contacts = mjx_data._impl.contact.dist < 0
            return (data, mjx_data), contacts

        def _disabled_drones(pos: Array, contacts: Array, race_data: RaceData) -> Array:
            disabled = race_data.disabled_drones | jp.any(pos < race_data.pos_limit_low, axis=-1)
            disabled = disabled | jp.any(pos > race_data.pos_limit_high, axis=-1)
            disabled = disabled | (race_data.target_gate == -1)
            if check_contacts:
                contacts = jp.any(contacts[:, None, :] & race_data.contact_masks, axis=-1)
                disabled = disabled | contacts
            return disabled

        def _step_race(
            race_data: RaceData,
            drone_pos: Array,
            mocap_pos: Array,
            mocap_quat: Array,
            contacts: Array,
        ) -> RaceData:
            """Step the environment data."""
            n_gates = len(race_data.gate_mj_ids)
            disabled_drones = _disabled_drones(drone_pos, contacts, race_data)
            gates_pos = mocap_pos[:, race_data.gate_mj_ids]
            obstacles_pos = mocap_pos[:, race_data.obstacle_mj_ids]
            # We need to convert the mocap quat from MuJoCo order to scipy order
            gates_quat = mocap_quat[:, race_data.gate_mj_ids][..., [1, 2, 3, 0]]
            # Extract the gate poses of the current target gates and check if the drones have passed
            # them between the last and current position
            gate_ids = race_data.gate_mj_ids[race_data.target_gate % n_gates]
            gate_pos = gates_pos[jp.arange(gates_pos.shape[0])[:, None], gate_ids]
            gate_quat = gates_quat[jp.arange(gates_quat.shape[0])[:, None], gate_ids]
            gate_size = (race_data.gate_size, race_data.gate_size)
            passed_plane, in_box = gate_passed(
                drone_pos, race_data.last_drone_pos, gate_pos, gate_quat, gate_size
            )
            passed = passed_plane & in_box
            if end_on_gate_bypass:
                # Disable drones that failed to pass the gate when they crossed the gate plane
                disabled_drones = disabled_drones | (~in_box & passed_plane)
            # Update the target gate index. Increment by one if drones have passed a gate
            target_gate = race_data.target_gate + passed * ~disabled_drones
            target_gate = jp.where(target_gate >= n_gates, -1, target_gate)
            # Update which gates and obstacles are or have been in range of the drone
            sensor_range = race_data.sensor_range
            dpos = drone_pos[..., None, :2] - gates_pos[:, None, :, :2]
            gates_visited = race_data.gates_visited | (jp.linalg.norm(dpos, axis=-1) < sensor_range)
            dpos = drone_pos[..., None, :2] - obstacles_pos[:, None, :, :2]
            obstacles_visited = race_data.obstacles_visited | (
                jp.linalg.norm(dpos, axis=-1) < sensor_range
            )
            race_data = race_data.replace(
                last_drone_pos=drone_pos,
                target_gate=target_gate,
                disabled_drones=disabled_drones,
                gates_visited=gates_visited,
                obstacles_visited=obstacles_visited,
            )
            return race_data

        def _step(
            env: DroneRaceEnv, action: Array
        ) -> tuple[tuple[SimData, Array], tuple[Array, Array, Array, Array, dict]]:
            data, mjx_data, race_data = env.data, env.mjx_data, env.race_data
            _marked_for_reset = env._marked_for_reset
            # 1. apply action: only attitude control
            data = _apply_action(data, action)
            # 2. step sim & race data
            data = sim._step(data, n_substeps)
            drone_pos = data.states.pos
            mocap_pos, mocap_quat = mjx_data.mocap_pos, mjx_data.mocap_quat
            (data, mjx_data), contacts = _contacts(data, mjx_data)
            race_data = _step_race(race_data, drone_pos, mocap_pos, mocap_quat, contacts)
            # 3. handle autoreset & update mask
            data, mjx_data, race_data = _reset_data(
                data, mjx_data, race_data, mask=_marked_for_reset
            )
            sim_time = data.core.steps / data.core.freq
            terminated, truncated = (
                _terminated(race_data),
                _truncated(sim_time[..., 0], max_episode_time),
            )
            _marked_for_reset = _done(terminated, truncated)
            # 4. construct obs & rewards
            steps = data.core.steps // (sim.freq // freq)
            return env.replace(
                data=data,
                mjx_data=mjx_data,
                race_data=race_data,
                steps=steps,
                _marked_for_reset=_marked_for_reset,
            ), (
                _obs(data, mjx_data, race_data),
                _reward(data, race_data),
                terminated,
                truncated,
                _info(),
            )

        # Initialize reset mask and step count
        steps = jp.zeros((num_envs, 1), dtype=jp.int32, device=jax_device)
        _marked_for_reset = jp.zeros((num_envs,), dtype=jp.bool_, device=jax_device)

        # region Return
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
            data=sim.data,
            mjx_data=sim.mjx_data,
            race_data=race_data,
            steps=steps,
            _marked_for_reset=_marked_for_reset,
            reset=jax.jit(_reset),
            step=jax.jit(_step),
        )


if __name__ == "__main__":
    import time
    from pathlib import Path

    import toml

    """Test the jittable drone environment implementation."""
    # Create the jittable environment
    config_path = Path(__file__).parents[2] / "scripts/config_race.toml"
    with open(config_path, "r") as f:
        config = ConfigDict(toml.load(f))

    env = DroneRaceEnv.create(num_envs=1024, device="gpu", **config.env)

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Initial Race Obs:")
    for k, v in obs.items():
        print(k, v.shape)

    def step_once(env: DroneRaceEnv, _) -> tuple[DroneRaceEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.data.states.vel[:, 0, :]  # (num_envs, 3)

        return env, (pos, vel)

    def rollout(env: DroneRaceEnv, num_steps: int) -> tuple[DroneRaceEnv, tuple[Array, Array]]:
        """Rollout for multiple steps using lax.scan."""
        env, (pos_traj, vel_traj) = jax.lax.scan(step_once, env, xs=None, length=num_steps)
        return env, (pos_traj, vel_traj)

    rollout_jit = jax.jit(rollout, static_argnames=("num_steps",))

    # Warm-up rollout
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 100)
    pos_traj.block_until_ready()
    end_time = time.time()
    print(f"Warm-up rollout took {end_time - start_time:.4f} seconds")

    # After jitting
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 100)
    pos_traj.block_until_ready()
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")
    start_time = time.time()
    env, (pos_traj, vel_traj) = rollout_jit(env, 100)
    pos_traj.block_until_ready()
    end_time = time.time()
    print(f"Jitted rollout took {end_time - start_time:.4f} seconds")

    print("\nPos trajectory shape:", pos_traj.shape)
    print("Vel trajectory shape:", vel_traj.shape)
