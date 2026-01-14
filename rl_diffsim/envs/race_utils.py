"""Utility functions for the drone racing environments."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import jax.numpy as jp
import mujoco
import numpy as np
import toml
from crazyflow.utils import leaf_replace
from jax.numpy import vectorize
from jax.scipy.spatial.transform import Rotation as JR
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from rl_diffsim.envs.randomize import (
    randomize_drone_inertia_fn,
    randomize_drone_mass_fn,
    randomize_drone_pos_fn,
    randomize_drone_quat_fn,
    randomize_gate_pos_fn,
    randomize_gate_rpy_fn,
    randomize_obstacle_pos_fn,
)

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from crazyflow.sim.data import SimData
    from jax import Array
    from mujoco import MjSpec
    from mujoco.mjx import Data
    from numpy.typing import NDArray


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def load_track(track: ConfigDict) -> tuple[ConfigDict, ConfigDict, ConfigDict]:
    """Load the track from a config dict.

    Gates and obstacles are loaded as a config dicts with keys `pos`, `quat`, `nominal_pos`, and
    `nominal_quat`. Drones are loaded as a config dicts with keys `pos`, `rpy`, `quat`, `vel` and
    `ang_vel`.

    Args:
        track: The track config dict.

    Returns:
        The gates, obstacles, and drones as config dicts.
    """
    assert "gates" in track, "Track must contain gates field."
    assert "obstacles" in track, "Track must contain obstacles field."
    assert "drones" in track, "Track must contain drones field."
    gate_pos = np.array([g["pos"] for g in track.gates], dtype=np.float32)
    gate_quat = (
        R.from_euler("xyz", np.array([g["rpy"] for g in track.gates])).as_quat().astype(np.float32)
    )
    gates = {
        "pos": gate_pos,
        "quat": gate_quat,
        "nominal_pos": gate_pos.copy(),
        "nominal_quat": gate_quat.copy(),
    }
    obstacle_pos = np.array([o["pos"] for o in track.obstacles], dtype=np.float32)
    obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos.copy()}
    drones = {
        k: np.array([drone.get(k) for drone in track.drones], dtype=np.float32)
        for k in track.drones[0].keys()
    }
    drones["quat"] = R.from_euler("xyz", drones["rpy"]).as_quat().astype(np.float32)
    return ConfigDict(gates), ConfigDict(obstacles), ConfigDict(drones)


@jax.jit
@partial(vectorize, signature="(3),(3),(3),(4)->()", excluded=[4])
def gate_passed(
    drone_pos: Array,
    last_drone_pos: Array,
    gate_pos: Array,
    gate_quat: Array,
    gate_size: tuple[float, float],
) -> bool:
    """Check if the drone has passed the current gate.

    We transform the position of the drone into the reference frame of the current gate. Gates have
    to be crossed in the direction of the x-Axis (pointing from -x to +x). Therefore, we check if x
    has changed from negative to positive. If so, the drone has crossed the plane spanned by the
    gate frame. We then check if the drone has passed the plane within the gate frame, i.e. the y
    and z box boundaries. First, we linearly interpolate to get the y and z coordinates of the
    intersection with the gate plane. Then we check if the intersection is within the gate box.

    Note:
        We need to recalculate the last drone position each time as the transform changes if the
        goal changes.

    Args:
        drone_pos: The position of the drone in the world frame.
        last_drone_pos: The position of the drone in the world frame at the last time step.
        gate_pos: The position of the gate in the world frame.
        gate_quat: The rotation of the gate as a wxyz quaternion.
        gate_size: The size of the gate box in meters.
    """
    # Transform last and current drone position into current gate frame.
    gate_rot = JR.from_quat(gate_quat)
    last_pos_local = gate_rot.apply(last_drone_pos - gate_pos, inverse=True)
    pos_local = gate_rot.apply(drone_pos - gate_pos, inverse=True)
    # Check the plane intersection. If passed, calculate the point of the intersection and check if
    # it is within the gate box.
    passed_plane = (last_pos_local[0] < 0) & (pos_local[0] > 0)
    alpha = -last_pos_local[0] / (pos_local[0] - last_pos_local[0])
    y_intersect = alpha * (pos_local[1]) + (1 - alpha) * last_pos_local[1]
    z_intersect = alpha * (pos_local[2]) + (1 - alpha) * last_pos_local[2]
    # Divide gate size by 2 to get the distance from the center to the edges
    in_box = (abs(y_intersect) < gate_size[0] / 2) & (abs(z_intersect) < gate_size[1] / 2)
    return passed_plane & in_box


# region Utils
def setup_sim(
    sim: Sim,
    gates: ConfigDict,
    obstacles: ConfigDict,
    drone: ConfigDict,
    disturbances: dict,
    randomizations: dict,
):
    """Setup the simulation data and build the reset and step functions with custom hooks."""
    gate_spec_path = Path(__file__).parent / "assets/gate.xml"
    obstacle_spec_path = Path(__file__).parent / "assets/obstacle.xml"
    gate_spec = mujoco.MjSpec.from_file(str(gate_spec_path))
    obstacle_spec = mujoco.MjSpec.from_file(str(obstacle_spec_path))
    load_track_into_sim(sim, gates, obstacles, gate_spec, obstacle_spec)
    # Set the initial drone states
    pos = sim.data.states.pos.at[...].set(drone["pos"])
    quat = sim.data.states.quat.at[...].set(drone["quat"])
    vel = sim.data.states.vel.at[...].set(drone["vel"])
    ang_vel = sim.data.states.ang_vel.at[...].set(drone["ang_vel"])
    states = sim.data.states.replace(pos=pos, quat=quat, vel=vel, ang_vel=ang_vel)
    sim.data = sim.data.replace(states=states)
    sim.build_default_data()
    # Build the reset randomizations and disturbances into the sim itself
    sim.reset_pipeline = sim.reset_pipeline + (build_reset_fn(randomizations),)
    sim.build_reset_fn()
    if "dynamics" in disturbances:
        disturbance_fn = build_dynamics_disturbance_fn(disturbances["dynamics"])
        sim.step_pipeline = sim.step_pipeline[:2] + (disturbance_fn,) + sim.step_pipeline[2:]
        sim.build_step_fn()


def load_track_into_sim(
    sim: Sim, gates: ConfigDict, obstacles: ConfigDict, gate_spec: MjSpec, obstacle_spec: MjSpec
) -> None:
    """Load the track into the simulation."""
    frame = sim.spec.worldbody.add_frame()
    n_gates, n_obstacles = len(gates["pos"]), len(obstacles["pos"])
    for i in range(n_gates):
        gate_body = gate_spec.body("gate")
        if gate_body is None:
            raise ValueError("Gate body not found in gate spec")
        gate = frame.attach_body(gate_body, "", f":{i}")
        gate.pos = gates["pos"][i]
        gate.quat = gates["quat"][i][[3, 0, 1, 2]]  # scipy->mujoco
        gate.mocap = True
    for i in range(n_obstacles):
        obstacle_body = obstacle_spec.body("obstacle")
        if obstacle_body is None:
            raise ValueError("Obstacle body not found in obstacle spec")
        obstacle = frame.attach_body(obstacle_body, "", f":{i}")
        obstacle.pos = obstacles["pos"][i]
        obstacle.mocap = True

    sim.build_mjx()  # Python call by object reference


def load_contact_masks(sim: Sim) -> Array:
    """Load contact masks for the simulation that zero out irrelevant contacts per drone."""
    sim.contacts()  # Trigger initial contact information computation
    contact = sim.mjx_data._impl.contact
    n_contacts = len(contact.geom1[0])
    masks = np.zeros((sim.n_drones, n_contacts), dtype=bool)
    # We only need one world to create the mask
    geom1, geom2 = (contact.geom1[0], contact.geom2[0])
    for i in range(sim.n_drones):
        geom_start = sim.mj_model.body_geomadr[sim.mj_model.body(f"drone:{i}").id]
        geom_count = sim.mj_model.body_geomnum[sim.mj_model.body(f"drone:{i}").id]
        geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
        geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)
        masks[i, :] = geom1_valid | geom2_valid
    geom_start = sim.mj_model.body_geomadr[sim.mj_model.body("world").id]
    geom_count = sim.mj_model.body_geomnum[sim.mj_model.body("world").id]
    geom1_valid = (geom1 >= geom_start) & (geom1 < geom_start + geom_count)
    geom2_valid = (geom2 >= geom_start) & (geom2 < geom_start + geom_count)

    masks = np.tile(masks[None, ...], (sim.n_worlds, 1, 1))
    return masks


# region Factories
def rng_spec2fn(fn_spec: dict) -> Callable:
    """Convert a function spec to a wrapped and scaled function from jax.random."""
    offset, scale = np.array(fn_spec.get("offset", 0)), np.array(fn_spec.get("scale", 1))
    kwargs = fn_spec.get("kwargs", {})
    if "shape" in kwargs:
        raise KeyError("Shape must not be specified for randomization functions.")
    kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
    jax_fn = partial(getattr(jax.random, fn_spec["fn"]), **kwargs)

    def random_fn(*args: Any, **kwargs: Any) -> Array:
        return jax_fn(*args, **kwargs) * scale + offset

    return random_fn


def build_reset_fn(randomizations: dict) -> Callable[[SimData, Array], SimData]:
    """Build the reset hook for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "drone_pos":
                randomization_fns += (randomize_drone_pos_fn(rng),)
            case "drone_rpy":
                randomization_fns += (randomize_drone_quat_fn(rng),)
            case "drone_mass":
                randomization_fns += (randomize_drone_mass_fn(rng),)
            case "drone_inertia":
                randomization_fns += (randomize_drone_inertia_fn(rng),)
            case "gate_pos" | "gate_rpy" | "obstacle_pos":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    def reset_fn(data: SimData, mask: Array) -> SimData:
        for randomize_fn in randomization_fns:
            data = randomize_fn(data, mask)
        return data

    return reset_fn


def build_track_randomization_fn(
    randomizations: dict, gate_mocap_ids: list[int], obstacle_mocap_ids: list[int]
) -> Callable[[Data, Array, jax.random.PRNGKey], Data]:
    """Build the track randomization function for the simulation."""
    randomization_fns = ()
    for target, rng in sorted(randomizations.items()):
        match target:
            case "gate_pos":
                randomization_fns += (randomize_gate_pos_fn(rng, gate_mocap_ids),)
            case "gate_rpy":
                randomization_fns += (randomize_gate_rpy_fn(rng, gate_mocap_ids),)
            case "obstacle_pos":
                randomization_fns += (randomize_obstacle_pos_fn(rng, obstacle_mocap_ids),)
            case "drone_pos" | "drone_rpy" | "drone_mass" | "drone_inertia":
                pass
            case _:
                raise ValueError(f"Invalid target: {target}")

    @jax.jit
    def track_randomization(
        data: Data,
        mask: Array,
        nominal_gate_pos: Array,
        nominal_gate_quat: Array,
        nominal_obstacle_pos: Array,
        key: jax.random.PRNGKey,
    ) -> Data:
        gate_quat = jp.roll(nominal_gate_quat, 1, axis=-1)  # Convert from scipy to MuJoCo order

        # Reset to default track positions first
        nominal_mocap_pos = data.mocap_pos.at[:, gate_mocap_ids].set(nominal_gate_pos)
        nominal_mocap_pos = nominal_mocap_pos.at[:, obstacle_mocap_ids].set(nominal_obstacle_pos)
        nominal_mocap_quat = data.mocap_quat.at[:, gate_mocap_ids].set(gate_quat)
        data = leaf_replace(data, mask, mocap_pos=nominal_mocap_pos, mocap_quat=nominal_mocap_quat)

        keys = jax.random.split(key, len(randomization_fns))
        for key, randomize_fn in zip(keys, randomization_fns, strict=True):
            data = randomize_fn(data, mask, key)
        return data

    return track_randomization


def build_dynamics_disturbance_fn(
    fn: Callable[[jax.random.PRNGKey, tuple[int]], jax.Array],
) -> Callable[[SimData], SimData]:
    """Build the dynamics disturbance function for the simulation."""

    def dynamics_disturbance(data: SimData) -> SimData:
        key, subkey = jax.random.split(data.core.rng_key)
        states = data.states
        states = states.replace(force=fn(subkey, states.force.shape))  # World frame
        return data.replace(states=states, core=data.core.replace(rng_key=key))

    return dynamics_disturbance


def generate_random_track(
    track: ConfigDict,
    key: jax.random.PRNGKey,
    border_safety_margin: float = 0.5,
    start_pos_min_r: float = 1.0,
    gates_min_r: float = 1.0,
    obstacle_min_r: float = 1.0,
    corridor_width_gates: float = 0.4,
    corridor_width_obstacles: float = 0.4,
    yaw_offset_randomization: float = 0.75,
    grid_size: tuple = (40, 40),
    jitter: bool = True,
) -> ConfigDict:
    """Fully JAX-jittable random track generator.

    Args:
        track: default track layout (n_gates, n_obs, start pos etc)
        key: for randomization
        border_safety_margin: min distance [m] of all objects fom the border
        start_pos_min_r: exclusion radius around inital drone position
        gates_min_r: exclusion radius around gates
        obstacle_min_r: minimum distance of obstacles from gates
        corridor_width_gates: width of corridor between gates
        corridor_width_obstacles: width of corridor between obstacles
        yaw_offset_randomization: amount of randomization for yaw
        grid_size: tuple(H, W) grid resolution
        jitter: whether to jitter gate inside grid cell

    Returns:
        New track layout with randomized tracks
    """
    # Get infos from track
    xmin, ymin = jnp.array(track.safety_limits["pos_limit_low"][:2]) + border_safety_margin
    xmax, ymax = jnp.array(track.safety_limits["pos_limit_high"][:2]) - border_safety_margin
    start_pos = jax.random.uniform(
        key,
        (2,),
        minval=jnp.array([xmin - border_safety_margin, ymin - border_safety_margin]),
        maxval=jnp.array([xmax + border_safety_margin, ymax + border_safety_margin]),
    )

    N_gates, N_obstacles = len(track.gates), len(track.obstacles)

    H, W = grid_size
    xs = jnp.linspace(xmin, xmax, W)
    ys = jnp.linspace(ymin, ymax, H)
    grid_x, grid_y = jnp.meshgrid(xs, ys)
    coords = jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
    coords_flat = coords.reshape(-1, 2)
    cell_w = (xmax - xmin) / W
    cell_h = (ymax - ymin) / H

    # Initial mask: everything allowed except around start_pos
    mask = jnp.ones((H, W), dtype=jnp.float32)
    dist2 = jnp.sum((coords - start_pos) ** 2, axis=-1)
    start_pos_mask = dist2 > (start_pos_min_r**2)
    mask = mask * start_pos_mask

    # Preallocate arrays
    assert N_gates == N_obstacles
    gates = jnp.full((N_gates, 3), jnp.nan, dtype=jnp.float32)
    obstacles = jnp.full((N_obstacles, 2), jnp.nan, dtype=jnp.float32)
    gate_distance_mask = jnp.ones((H, W), dtype=jnp.float32)
    gate_distance_mask_obstacles = jnp.ones((H, W), dtype=jnp.float32)
    obstacle_distance_mask = jnp.ones((H, W), dtype=jnp.float32)
    corridor_mask = jnp.ones((H, W), dtype=jnp.bool)

    # PRNG keys
    keys = jax.random.split(key, 2 * N_gates + 1)
    key_pos, key_yaw = keys[::2], keys[1::2]
    # --- Sample obstacles ---
    keys_obs = jax.random.split(keys[-1], N_obstacles)

    # --- Helper: yaw adjustment ---
    def adjust_yaw(i: int, yaw: jnp.floating, gates: Array, candidate: Array) -> jnp.floating:
        prev_pos = jax.lax.cond(
            i == 0, lambda _: start_pos, lambda _: gates[i - 1, :2], operand=None
        )
        travel_dir = candidate - prev_pos
        yaw += jnp.arctan2(travel_dir[1], travel_dir[0])
        return yaw % (2 * jnp.pi)

    # --- Scan body for gate placement ---
    def body(carry: tuple, i: int) -> tuple:
        (
            mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask_obstacles,
        ) = carry
        sub_pos, sub_yaw = key_pos[i], key_yaw[i]
        sub_obs = keys_obs[i]

        flat_mask = mask.reshape(-1)
        total = flat_mask.sum()
        p = jnp.where(total > 0, flat_mask / total, jnp.ones_like(flat_mask) / flat_mask.size)
        idx = jax.random.choice(sub_pos, flat_mask.shape[0], p=p)
        chosen_center = coords_flat[idx]

        # optional jitter
        if jitter:
            sub_pos, subk1, subk2 = jax.random.split(sub_pos, 3)
            off_x = (jax.random.uniform(subk1, ()) - 0.5) * cell_w
            off_y = (jax.random.uniform(subk2, ()) - 0.5) * cell_h
            candidate = chosen_center + jnp.array([off_x, off_y])
        else:
            candidate = chosen_center

        # sample yaw and adjust
        yaw = jax.random.uniform(
            sub_yaw, (), minval=-yaw_offset_randomization, maxval=yaw_offset_randomization
        )
        # yaw = adjust_yaw(i, yaw, gates, candidate)
        yaw = adjust_yaw(i, yaw, gates, candidate)
        # yaw = adjust_yaw(i, gates, candidate)

        gates = gates.at[i].set(jnp.array([candidate[0], candidate[1], yaw]))

        # mask out circular region around gate
        dist2 = jnp.sum((coords - candidate) ** 2, axis=-1)
        gate_distance_mask = gate_distance_mask * (dist2 > (gates_min_r**2))
        gate_distance_mask_obstacles = gate_distance_mask_obstacles * (dist2 > (obstacle_min_r**2))

        # mask out corridor from prev gate or start
        prev_pos = jax.lax.cond(i == 0, lambda _: start_pos, lambda _: gates[i - 1, :2], None)
        v = candidate - prev_pos
        v_norm = jnp.linalg.norm(v) + 1e-8
        u = v / v_norm
        p_to_line = coords - prev_pos
        proj = jnp.sum(p_to_line * u, axis=-1)

        proj_exp = proj[..., None]  # shape (H, W, 1)
        closest = prev_pos + proj_exp * u  # shape (H, W, 2)
        perp_dist = jnp.linalg.norm(coords - closest, axis=-1)  # shape (H, W)

        on_segment = (proj >= 0) & (proj <= v_norm)
        corridor_mask_gates = (perp_dist < corridor_width_gates) & on_segment
        corridor_mask_obstacles = (perp_dist < corridor_width_obstacles) & on_segment

        new_mask = mask * gate_distance_mask
        new_mask = new_mask * (1.0 - corridor_mask_gates)

        # mask_corridors = jnp.maximum(mask_corridors, corridor_mask.astype(jnp.float32))
        mask_corridors = (
            corridor_mask_obstacles
            * gate_distance_mask_obstacles
            * obstacle_distance_mask
            * start_pos_mask
        )

        # sample obstacle pos
        flat_mask = mask_corridors.reshape(-1)
        total = flat_mask.sum()
        p = jnp.where(total > 0, flat_mask / total, jnp.ones_like(flat_mask) / flat_mask.size)
        idx = jax.random.choice(sub_obs, flat_mask.shape[0], p=p)
        chosen_center = coords_flat[idx]

        candidate = chosen_center

        obstacles = obstacles.at[i].set(jnp.array([candidate[0], candidate[1]]))

        # mask out circular region around obstacle
        dist2 = jnp.sum((coords - candidate) ** 2, axis=-1)
        obstacle_distance_mask = obstacle_distance_mask * (dist2 > (obstacle_min_r**2))

        return (
            new_mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask_obstacles,
            # jnp.bool(mask_corridors),
        ), None

    (
        (
            mask_final,
            gates_final,
            obstacles_final,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask,
        ),
        _,
    ) = jax.lax.scan(
        body,
        (
            mask,
            gates,
            obstacles,
            gate_distance_mask,
            gate_distance_mask_obstacles,
            obstacle_distance_mask,
            corridor_mask,
        ),
        jnp.arange(N_gates),
    )

    # Write random track
    for i, d in enumerate(track.drones):
        d["pos"][:2] = start_pos
        # TODO multi drones?

    for i, g in enumerate(track.gates):
        g["pos"][:2] = gates_final[i, :2].tolist()
        g["rpy"][2] = gates_final[i, 2].tolist()

    for i, o in enumerate(track.obstacles):
        o["pos"][:2] = obstacles_final[i, :2].tolist()

    return track


# checks
def check_race_track(
    gates_pos: NDArray,
    nominal_gates_pos: NDArray,
    gates_quat: NDArray,
    nominal_gates_quat: NDArray,
    obstacles_pos: NDArray,
    nominal_obstacles_pos: NDArray,
    rng_config: ConfigDict,
):
    """Check if the race track's gates and obstacles are within tolerances.

    Args:
        gates_pos: The positions of the gates.
        nominal_gates_pos: The nominal positions of the gates.
        gates_quat: The orientations of the gates as quaternions.
        nominal_gates_quat: The nominal orientations of the gates as quaternions.
        obstacles_pos: The positions of the obstacles.
        nominal_obstacles_pos: The nominal positions of the obstacles.
        rng_config: Environment randomization config.
    """
    assert rng_config.gate_pos.fn == "uniform", "Race track checks expect uniform distributions"
    assert rng_config.obstacle_pos.fn == "uniform", "Race track checks expect uniform distributions"
    low, high = rng_config.gate_pos.kwargs.minval, rng_config.gate_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(gates_pos, nominal_gates_pos)):
        check_bounds(f"gate{i + 1}", pos, nominal_pos, np.array(low), np.array(high))

    high_tol = np.array(rng_config.gate_rpy.kwargs.maxval)
    low_tol = np.array(rng_config.gate_rpy.kwargs.minval)
    for i, (quat, nominal_quat) in enumerate(zip(gates_quat, nominal_gates_quat)):
        gate_rot = R.from_quat(quat)
        nominal_rot = R.from_quat(nominal_quat)
        check_rotation(f"gate{i + 1}", gate_rot, nominal_rot, low=low_tol, high=high_tol)

    low, high = rng_config.obstacle_pos.kwargs.minval, rng_config.obstacle_pos.kwargs.maxval
    for i, (pos, nominal_pos) in enumerate(zip(obstacles_pos, nominal_obstacles_pos)):
        check_bounds(
            f"obstacle{i + 1}", pos[:2], nominal_pos[:2], np.array(low[:2]), np.array(high[:2])
        )


def check_drone_start_pos(
    nominal_pos: NDArray, real_pos: NDArray, rng_config: ConfigDict, drone_name: str
):
    """Check if the real drone start position matches the settings.

    Args:
        nominal_pos: Nominal drone position.
        real_pos: Current drone position.
        rng_config: Environment randomization config.
        drone_name: Name of the drone (e.g. cf10).
    """
    assert rng_config.drone_pos.fn == "uniform", (
        "Drone start position check expects uniform distributions"
    )
    tol_min, tol_max = rng_config.drone_pos.kwargs.minval, rng_config.drone_pos.kwargs.maxval
    check_bounds(
        drone_name, real_pos[:2], nominal_pos[:2], np.array(tol_min[:2]), np.array(tol_max[:2])
    )


def check_bounds(name: str, actual: NDArray, desired: NDArray, low: NDArray, high: NDArray):
    """Check if the actual value is within the specified bounds of the desired value.

    Args:
        name: Name of the object being checked.
        actual: Values to check.
        desired: Reference values.
        low: Lower bound. Minimum permissible value of (actual - desired).
        high: Upper bound. Maximum value of (actual - desired).

    Raises:
            RuntimeError: The values are not in the permissible interval.
    """
    if np.any(actual - desired < low):
        raise RuntimeError(
            f"{name} exceeds lower tolerances ({low}). Position is: {actual}, should be: {desired}"
        )
    if np.any(actual - desired > high):
        raise RuntimeError(
            f"{name} exceeds upper tolerances ({high}). Position is: {actual}, should be: {desired}"
        )


def check_rotation(name: str, actual_rot: R, desired_rot: R, low: NDArray, high: NDArray):
    """Compare gate orientations in world-frame Euler xyz.

    Warning:
        Comparing Euler angles is tricky. While we try to sanitize the comparison as best as we
        can, edge cases may still cause failures.

    Todo:
        Switch to a more sane rotation check method.

    Args:
        name: Name of the object being checked.
        actual_rot: R object describing rotation of the real object.
        desired_rot:  R object describing rotation of the nominal object.
        low: Array designating the per axis rotation lower limit
        high: Array designating the per axis rotation higher limit

    """
    actual = actual_rot.as_euler("xyz", degrees=False)
    desired = desired_rot.as_euler("xyz", degrees=False)
    diff = (actual - desired + np.pi) % (2 * np.pi) - np.pi
    if np.any(diff < low):
        raise RuntimeError(
            f"{name} exceeds lower rotation tolerances ({low}).\n"
            f"Rotation is: {actual}, should be: {desired}"
        )
    elif np.any(diff > high):
        raise RuntimeError(
            f"{name} exceeds higher rotation tolerances ({high}).\n"
            f"Rotation is: {actual}, should be: {desired}"
        )
