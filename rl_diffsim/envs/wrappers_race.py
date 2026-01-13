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

matplotlib.use("Agg")  # render to raster images

from rl_diffsim.envs.wrappers import Wrapper

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from mujoco.mjx import Data

    from rl_diffsim.envs.drone_race_env import RaceData


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

    base: struct.PyTreeNode = struct.field(pytree_node=True)

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
        base: struct.PyTreeNode,
        gate_pos_coef: float = 1.0,
        gate_vel_coef: float = 1.0,
        max_vel: float = 2.0,
        contact_safe_dist: float = 0.1,
        contact_coef: float = 50.0,
    ) -> "RaceWrapper":
        """Create a RaceWrapper around `base`.

        Parameters:
            base: The jittable base environment to wrap.

        Returns:
            RaceWrapper: Configured wrapper instance.
        """
        n_envs = base.unwrapped.num_envs
        n_gates = base.unwrapped.race_data.n_gates

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
        def _contacts_reward(mjx_data: Data) -> Array:
            # A pure contact reward calculation
            dists = mjx_data._impl.contact.dist - contact_safe_dist  # (num_envs, num_contacts)
            rewards = jp.sum(jp.where(dists < 0.0, -dists * dists, 0.0), axis=-1)
            return rewards

        def _race_reward(
            data: SimData, mjx_data: Data, race_data: RaceData, obs: dict[str, Array]
        ) -> Array:
            """Compute race reward for rl training.

            Rewards terms:
            Differentiable:
            - Approaching gate: distance towards gate
            - Approaching gate: vel towards gate
            - Avoiding obstacles: distance towards obstacles
            """
            pos = data.states.pos[:, 0, :]  # (num_envs, 3)
            vel = data.states.vel[:, 0, :]  # (num_envs, 3)
            # Relative position to target gate
            gate_rel_pos = obs["gate_rel_pos"]  # (num_envs, 3)
            # r_gate_pos = gate_pos_coef * jp.exp(
            #     -2.0 * jp.linalg.norm(gate_rel_pos, axis=-1)
            # )  # (num_envs,)
            r_gate_pos = gate_pos_coef * jp.exp(
                -1.0 * jp.sum(gate_rel_pos * gate_rel_pos, axis=-1)
            )  # (num_envs,)

            # Relative velocity (velocity projected onto gate_rel_pos)
            gate_rel_pos_unit = gate_rel_pos / (
                jp.linalg.norm(gate_rel_pos, axis=-1, keepdims=True) + 1e-8
            )
            gate_rel_vel_norm = jp.sum(vel * gate_rel_pos_unit, axis=-1)  # (num_envs,)
            r_gate_vel = gate_vel_coef * jp.tanh(gate_rel_vel_norm / max_vel)  # (num_envs,)

            # Penalty for collisions
            r_collision = contact_coef * _contacts_reward(mjx_data)  # (num_envs,)

            rewards = jp.zeros((pos.shape[0],))
            rewards += r_gate_pos
            rewards += r_gate_vel
            rewards += r_collision
            return rewards

        def _reset(
            env: "RaceWrapper", *, seed: int | None = None, options: dict | None = None
        ) -> tuple["RaceWrapper", tuple[Any, Any]]:
            base_env, (obs, info) = env.base.reset(env.base, seed=seed, options=options)
            basic_obs = _basic_obs(obs)
            race_obs = _race_obs(obs)
            obs = {**basic_obs, **race_obs}
            env = env.replace(base=base_env)
            return env, (obs, info)

        def _step(env: "RaceWrapper", action: Array) -> tuple["RaceWrapper", tuple[Any, ...]]:
            base_env, (obs, reward, terminated, truncated, info) = env.base.step(env.base, action)
            basic_obs = _basic_obs(obs)  # (num_envs, 13)
            race_obs = _race_obs(obs)  # (num_envs, 15)
            obs = {**basic_obs, **race_obs}  # (num_envs, 28)
            reward = _race_reward(
                env.base.unwrapped.data,
                env.base.unwrapped.mjx_data,
                env.base.unwrapped.race_data,
                obs,
            )

            env = env.replace(base=base_env)
            return env, (obs, reward, terminated, truncated, info)

        return cls(base=base, step=jax.jit(_step), reset=jax.jit(_reset))


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
