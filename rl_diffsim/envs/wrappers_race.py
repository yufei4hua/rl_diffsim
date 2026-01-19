"""wrappers for struct.PyTreeNode-style environments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import flax.struct as struct
import jax
import jax.numpy as jp
import matplotlib
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from rl_diffsim.envs.race_utils import compute_objects_contact_masks
from rl_diffsim.envs.wrappers import Wrapper

matplotlib.use("Agg")  # render to raster images
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from mujoco.mjx import Data

    from rl_diffsim.envs.drone_race_env import DroneRaceEnv, RaceData


# region RaceWrapper
@struct.dataclass
class RaceWrapper(Wrapper):
    """Constructs race-specific observations and rewards.

    RL racing observations:
    - Drone basic states: pos, quat, vel, ang_vel (13,)
    - One-hot encoding of target gate (n_gates,)
    - Relative position to target gate (3,)
    - Normal vector of target gate (3,)
    - Relative velocity to target gate (3,)
    - Relative xy-position to all obstacles (n_obstacles, 2)

    """

    base: DroneRaceEnv = struct.field(pytree_node=True)

    progress_coef: float = struct.field(pytree_node=True)
    last_target_gate: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @property
    def single_observation_space(self) -> spaces.Space:
        """Single observation space."""
        n_gates = len(self.base.unwrapped.race_data.track.gates)
        n_obstacles = len(self.base.unwrapped.race_data.track.obstacles)
        obs_spec = {
            "drone_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_quat": spaces.Box(low=-1, high=1, shape=(4,)),
            "drone_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "drone_ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "gate_onehot": spaces.Box(low=0, high=1, shape=(n_gates,), dtype=bool),
            "gate_rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "gate_normal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            "obst_rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(n_obstacles, 2)),
        }
        return spaces.Dict(obs_spec)

    @property
    def observation_space(self) -> spaces.Space:
        """Batched observation space matching the wrapper's num_envs."""
        return batch_space(self.single_observation_space, self.num_envs)

    @classmethod
    def create(
        cls,
        base: DroneRaceEnv,
        gate_pos_coef: float | tuple[float, float] = 1.0,
        gate_vel_coef: float | tuple[float, float] = 1.0,
        gate_pass_coef: float | tuple[float, float] = 5.0,
        min_vel: float = 0.5,
        max_vel: float = 2.0,
        cont_floor_safe_dist: float = 0.1,
        cont_gate_safe_dist: float = 0.1,
        cont_obst_safe_dist: float = 0.1,
        contact_coef: float | tuple[float, float] = 10.0,
        gate_size: float | tuple[float, float] = 0.45,
        total_timesteps: int = 0,
    ) -> "RaceWrapper":
        """Create a RaceWrapper around `base`.

        Parameters:
            base: The jittable base environment to wrap.

        Returns:
            RaceWrapper: Configured wrapper instance.
        """
        n_envs = base.unwrapped.num_envs
        n_gates = base.unwrapped.race_data.n_gates
        floor_contact_masks = compute_objects_contact_masks(
            base.unwrapped.sim, base.unwrapped.race_data.contact_masks, "world"
        )
        gate_contact_masks = compute_objects_contact_masks(
            base.unwrapped.sim, base.unwrapped.race_data.contact_masks, "gate"
        )
        obstacle_contact_masks = compute_objects_contact_masks(
            base.unwrapped.sim, base.unwrapped.race_data.contact_masks, "obstacle"
        )

        # region Obs
        def _basic_obs(obs: dict[str, Array]) -> dict[str, Array]:
            """Extract the basic observation [pos, quat, vel, ang_vel]."""
            # add drone_ prefix because flax will sort keys alphabetically
            return {
                "drone_pos": obs["pos"],  # (num_envs, 3)
                "drone_quat": obs["quat"],  # (num_envs, 4)
                "drone_vel": obs["vel"],  # (num_envs, 3)
                "drone_ang_vel": obs["ang_vel"],  # (num_envs, 3)
            }

        def _race_obs(obs: dict[str, Array]) -> dict[str, Array]:
            """Compute race observation for rl agent."""
            pos = obs["pos"]  # (num_envs, 3)
            target_gate = obs["target_gate"]  # (num_envs,)

            # One-hot encoding of target gate
            gate_onehot = (
                jp.zeros((n_envs, n_gates)).at[jp.arange(n_envs), target_gate].set(1)
            )  # (num_envs, n_gates)

            # Get target gate position for each env
            gate_pos = obs["gates_pos"][jp.arange(n_envs), target_gate]  # (num_envs, 3)

            # Relative position to target gate
            gate_rel_pos = gate_pos - pos  # (num_envs, 3)

            # Normal vector of target gate
            gate_quat = obs["gates_quat"][jp.arange(n_envs), target_gate]  # (num_envs, 4)
            gate_normal = R.from_quat(gate_quat).apply(jp.array([1.0, 0.0, 0.0]))  # (num_envs, 3)

            # Relative xy-position to all obstacles
            obstacles_pos = obs["obstacles_pos"]  # (num_envs, n_obstacles, 2)
            obst_rel_pos = obstacles_pos[:, :, :2] - pos[:, None, :2]  # (num_envs, n_obstacles, 2)

            race_obs = {
                "gate_onehot": gate_onehot,
                "gate_rel_pos": gate_rel_pos,
                "gate_normal": gate_normal,
                "obst_rel_pos": obst_rel_pos,
            }

            return race_obs

        # region Rewards
        def _schedule_coef(coef: float | tuple[float, float], progress: float) -> float:
            if isinstance(coef, float):
                return coef
            elif isinstance(coef, tuple) and len(coef) == 2:
                start, end = coef
                return start + (end - start) * progress
            else:
                raise ValueError("coef must be a float or a tuple of two floats.")

        def _contacts_reward(mjx_data: Data) -> Array:
            # A pure contact reward calculation
            dists = mjx_data._impl.contact.dist  # (num_envs, num_contacts)
            floor_dists = jp.where(floor_contact_masks[:, 0], dists - cont_floor_safe_dist, 0.0)
            gate_dists = jp.where(gate_contact_masks[:, 0], dists - cont_gate_safe_dist, 0.0)
            obstacle_dists = jp.where(
                obstacle_contact_masks[:, 0], dists - cont_obst_safe_dist, 0.0
            )
            r_floor = jp.sum(jp.where(floor_dists < 0.0, -floor_dists * floor_dists, 0.0), axis=-1)
            r_gate = jp.sum(jp.where(gate_dists < 0.0, -gate_dists * gate_dists, 0.0), axis=-1)
            r_obstacle = jp.sum(
                jp.where(obstacle_dists < 0.0, -obstacle_dists * obstacle_dists, 0.0), axis=-1
            )
            return r_floor + r_gate + r_obstacle

        def _race_reward(
            env: RaceWrapper,
            data: SimData,
            mjx_data: Data,
            race_data: RaceData,
            obs: dict[str, Array],
        ) -> Array:
            """Compute race reward for rl training.

            Rewards terms:
            Differentiable:
            - Approaching gate: distance towards gate
            - Approaching gate: vel towards gate
            - Avoiding obstacles: distance towards obstacles
            Non-differentiable:
            - Passing through gate
            """
            pos = data.states.pos[:, 0, :]  # (num_envs, 3)
            vel = data.states.vel[:, 0, :]  # (num_envs, 3)
            # Relative position to target gate
            gate_rel_pos = obs["gate_rel_pos"]  # (num_envs, 3)
            gate_dist = jp.linalg.norm(gate_rel_pos, axis=-1)  # (num_envs,)
            r_gate_pos = jp.exp(-2.0 * gate_dist)  # (num_envs,)
            # r_gate_pos = gate_pos_coef * jp.exp(
            #     -1.0 * jp.sum(gate_rel_pos * gate_rel_pos, axis=-1)
            # )  # (num_envs,)

            # Relative velocity (velocity projected onto gate_rel_pos)
            gate_norm = obs["gate_normal"]  # (num_envs, 3)
            ref_vel = gate_rel_pos - (1 - 1 / (gate_dist[:, None] + 1e-8)) * jp.sum(gate_rel_pos * gate_norm, axis=-1, keepdims=True) * gate_norm
            ref_vel_unit = ref_vel / (
                jp.linalg.norm(ref_vel, axis=-1, keepdims=True) + 1e-8
            )
            gate_rel_vel_norm = jp.sum(vel * ref_vel_unit, axis=-1)  # (num_envs,)
            r_gate_vel = jp.tanh((gate_rel_vel_norm - min_vel) / (max_vel / 2.0))  # (num_envs,)

            # Penalty for collisions
            r_collision = _contacts_reward(mjx_data)  # (num_envs,)

            # Reward for passing through gate
            passed = (
                jp.logical_or(
                    race_data.target_gate > env.last_target_gate, race_data.target_gate == -1
                )
                .astype(jp.float32)
                .squeeze(-1)
            )  # (num_envs,)
            r_pass_gate = passed  # (num_envs,)
            env = env.replace(last_target_gate=race_data.target_gate)

            # construct total reward
            rewards = jp.zeros((pos.shape[0],))
            k_gate_pos = _schedule_coef(gate_pos_coef, env.progress_coef)
            k_gate_vel = _schedule_coef(gate_vel_coef, env.progress_coef)
            k_contact = _schedule_coef(contact_coef, env.progress_coef)
            k_gate_pass = _schedule_coef(gate_pass_coef, env.progress_coef)
            rewards += k_gate_pos * r_gate_pos
            rewards += k_gate_vel * r_gate_vel
            rewards += k_contact * r_collision
            rewards += k_gate_pass * r_pass_gate
            # jax.debug.print("r_pos: {r_gate_pos}, r_vel: {r_gate_vel}, r_coll: {r_collision}, r_pass: {r_pass_gate}", 
            #                 r_gate_pos=k_gate_pos * r_gate_pos, r_gate_vel=k_gate_vel * r_gate_vel, r_collision=k_contact * r_collision, r_pass_gate=k_gate_pass * r_pass_gate)
            return env, rewards

        # region Reset & Step
        def _reset(
            env: "RaceWrapper", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["RaceWrapper", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            basic_obs = _basic_obs(obs)
            race_obs = _race_obs(obs)
            obs = {**basic_obs, **race_obs}
            env = env.replace(
                base=base_env,
                progress_coef=0.0,
                last_target_gate=jp.zeros((n_envs, 1), dtype=jp.int32),
            )
            return env, (obs, info)

        def _step(env: "RaceWrapper", action: Array) -> tuple["RaceWrapper", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            basic_obs = _basic_obs(obs)  # (num_envs, 13)
            race_obs = _race_obs(obs)  # (num_envs, 15)
            obs = {**basic_obs, **race_obs}  # (num_envs, 28)
            env, reward = _race_reward(
                env, base_env.data, base_env.mjx_data, base_env.race_data, obs
            )
            if total_timesteps > 0:
                progress_coef = env.progress_coef + env.num_envs / total_timesteps
            else:
                progress_coef = 1.0
            # scheduled gate_size
            g_size = _schedule_coef(gate_size, progress_coef)
            base_env = base_env.replace(race_data=base_env.race_data.replace(gate_size=g_size))
            # enable check_contacts after certain progress
            base_env = jax.lax.cond(
                progress_coef > 0.66,
                lambda env: env.replace(race_data=env.race_data.replace(check_contacts=True)),
                lambda env: env,
                base_env
            )

            env = env.replace(base=base_env, progress_coef=progress_coef)
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            last_target_gate=jp.zeros((n_envs, 1), dtype=jp.int32),
            progress_coef=0.0,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )


if __name__ == "__main__":
    import time
    from pathlib import Path

    import toml
    from ml_collections import ConfigDict

    from rl_diffsim.envs.drone_race_env import DroneRaceEnv

    """Test the jittable drone environment implementation."""
    # Create the jittable environment
    config_path = Path(__file__).parents[2] / "scripts/config_race.toml"
    with open(config_path, "r") as f:
        config = ConfigDict(toml.load(f))

    env = DroneRaceEnv.create(num_envs=1024, device="gpu", **config.env)
    env = RaceWrapper.create(env)

    # Reset the environment
    env, (obs, info) = env.reset(env, seed=42)
    print("Initial Race RL Obs:")
    for k, v in obs.items():
        print(k, v[0])

    def step_once(env: DroneRaceEnv, _) -> tuple[DroneRaceEnv, tuple[Array, Array]]:
        """Single env step for lax.scan."""
        base_action = jp.array([0.0, 0.0, 0.0, 0.4], dtype=jp.float32)
        action = jp.broadcast_to(base_action, env.action_space.shape)  # (num_envs, act_dim)

        env, (next_obs, reward, terminated, truncated, info) = env.step(env, action)

        pos = env.unwrapped.data.states.pos[:, 0, :]  # (num_envs, 3)
        vel = env.unwrapped.data.states.vel[:, 0, :]  # (num_envs, 3)

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


# region RecordRaceData
@struct.dataclass
class RecordRaceData(Wrapper):
    """Wrapper that records debugging data."""

    base: struct.PyTreeNode = struct.field(pytree_node=True)

    _record_act: Array = struct.field(pytree_node=True)
    _record_pos: Array = struct.field(pytree_node=True)
    _record_rpy: Array = struct.field(pytree_node=True)

    step: Callable = struct.field(pytree_node=False)
    reset: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(cls, base: struct.PyTreeNode) -> "RecordRaceData":
        """Create a RecordRaceData wrapper around `base`.

        Parameters:
            base: The jittable environment to wrap.

        Returns:
            RecordRaceData: Configured wrapper instance.
        """
        assert hasattr(base.unwrapped, "max_episode_time"), (
            "Base env must have max_episode_time attribute"
        )
        max_T = int(base.unwrapped.max_episode_time * base.unwrapped.freq)
        num_envs = int(base.num_envs)
        act_dim = int(base.action_space.shape[-1])
        pos_dim = 3

        # initialize buffers # TODO: this takes a lot of memory
        empty_act = jp.zeros((max_T, num_envs, act_dim), dtype=jp.float32)
        empty_pos = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)
        empty_rpy = jp.zeros((max_T, num_envs, pos_dim), dtype=jp.float32)

        def _extract_data(base: "RecordRaceData") -> dict[str, Array]:
            """Extract recorded data as a dict of arrays."""
            raw = base.unwrapped
            pos = raw.data.states.pos[:, 0, :]
            rpy = R.from_quat(raw.data.states.quat[:, 0, :]).as_euler("xyz")
            return pos, rpy

        def _reset(
            env: "RecordRaceData", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["RecordRaceData", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            pos, rpy = _extract_data(base_env)
            env = env.replace(
                base=base_env,
                _record_act=empty_act,
                _record_pos=empty_pos.at[0, ...].set(pos),
                _record_rpy=empty_rpy.at[0, ...].set(rpy),
            )
            return env, (obs, info)

        def _step(env: "RecordRaceData", action: Array) -> tuple["RecordRaceData", tuple[Any, ...]]:
            # step the wrapped environment
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)

            act = action  # shape: (num_envs, act_dim)
            pos, rpy = _extract_data(base_env)

            # record data
            new_act = env._record_act.at[env.steps, ...].set(act)
            new_pos = env._record_pos.at[env.steps, ...].set(pos)
            new_rpy = env._record_rpy.at[env.steps, ...].set(rpy)

            env = env.replace(
                base=base_env, _record_act=new_act, _record_pos=new_pos, _record_rpy=new_rpy
            )
            return env, (obs, reward, terminated, truncated, info)

        return cls(
            base=base,
            _record_act=empty_act,
            _record_pos=empty_pos,
            _record_rpy=empty_rpy,
            step=jax.jit(_step),
            reset=jax.jit(_reset),
        )

    def calc_rmse(self) -> float:
        """Compute RMSE between recorded position and goal (return in meters)."""
        pos = np.array(self._record_pos)  # shape: (T, num_envs, 3)
        goal = np.array(self._record_goal)  # shape: (T, num_envs, 3)
        pos_err = np.linalg.norm(pos - goal, axis=-1)  # shape: (T, num_envs)
        rmse = np.sqrt(np.mean(pos_err**2))
        return rmse

    def plot_eval(self, save_path: str = "race_eval_plot.png") -> plt.Figure:
        """Plot recorded traces and save to `save_path`."""
        episode_length = self.steps[0, 0]
        actions = np.array(self._record_act)[:episode_length]
        pos = np.array(self._record_pos)[:episode_length]
        rpy = np.array(self._record_rpy)[:episode_length]

        fig = plt.figure(figsize=(18, 12), constrained_layout=True)
        gs = GridSpec(nrows=3, ncols=4, figure=fig, hspace=0.05, wspace=0.05)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1:3, 0:2]),
            fig.add_subplot(gs[1:3, 2:4]),
        ]
        raw = self.unwrapped

        # Actions
        if raw.control == "attitude":
            action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
        else:
            raise ValueError(f"Unsupported control type: {raw.sim.control}")
        action_sim_low = np.array(raw.single_action_space.low)
        action_sim_high = np.array(raw.single_action_space.high)
        scale = (action_sim_high - action_sim_low) / 2.0
        mean = (action_sim_high + action_sim_low) / 2.0
        actions = actions * scale + mean  # rescale to sim action range
        for i in range(4):
            if i < 3:
                axes[i].plot(rpy[:, 0, i], label="Actual")
            axes[i].plot(actions[:, 0, i], linestyle="--", color="orange", label="Command")
            axes[i].set_title(f"{action_labels[i]}")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Angle (rad)")
            axes[i].legend()
            axes[i].grid(True)

        # Race trajectory plot
        gates = np.array(raw.mjx_data.mocap_pos[0, raw.race_data.gate_mj_ids])
        obstacles = np.array(raw.mjx_data.mocap_pos[0, raw.race_data.obstacle_mj_ids])

        axes[4].plot(pos[:, 0, 0], pos[:, 0, 1])
        axes[4].scatter(
            gates[:, 0], gates[:, 1], c="green", s=80, marker="o", label="Gates", zorder=5
        )
        axes[4].scatter(
            obstacles[:, 0], obstacles[:, 1], c="red", s=80, marker="x", label="Obstacles", zorder=5
        )
        axes[4].set_title("Race Trajectory XY Plane")
        axes[4].set_xlabel("X Position (m)")
        axes[4].set_ylabel("Y Position (m)")
        axes[4].grid(True)
        axes[4].axis("equal")

        axes[5].plot(pos[:, 0, 0], pos[:, 0, 2])
        axes[5].scatter(
            gates[:, 0], gates[:, 2], c="green", s=80, marker="o", label="Gates", zorder=5
        )
        axes[5].set_title("Race Trajectory XZ Plane")
        axes[5].set_xlabel("X Position (m)")
        axes[5].set_ylabel("Z Position (m)")
        axes[5].grid(True)
        axes[5].axis("equal")

        fig.savefig(Path(__file__).parents[2] / "saves" / save_path, bbox_inches="tight")

        return fig
