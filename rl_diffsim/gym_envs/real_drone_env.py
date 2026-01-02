"""Real-world drone environments.

This module contains the environments for controlling a single or multiple drones in a real-world
setting. It uses data from real-world observations from motion capture systems and sends actions to
the real drones via ROS2.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cflib
import jax
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_models.core import load_params
from drone_models.transform import force2pwm
from gymnasium import Env
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rl_diffsim.control.controller import Controller

logger = logging.getLogger(__name__)


# region CoreEnv
class RealDroneCoreEnv:
    """Deployable version of the (multi-agent) drone environments.

    This class acts as a generic core implementation of the environment logic that can be reused for
    both single-agent and multi-agent deployments.
    """

    POS_UPDATE_FREQ = 30  # Frequency of position updates to the drone estimator in Hz

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        pos_limit_low: list[float] | None = None,
        pos_limit_high: list[float] | None = None,
        control_mode: Literal["state", "attitude", "rotor_vel"] = "state",
    ):
        """Create a deployable version of the drone environment.

        Args:
            drones: List of all drones, including their channel and id.
            rank: Rank of the drone that is controlled by this environment.
            freq: Environment step frequency.
            pos_limit_low: Lower position limits for safety [x, y, z].
            pos_limit_high: Upper position limits for safety [x, y, z].
            control_mode: Control mode of the drone.
        """
        assert rclpy.ok(), "ROS2 is not running. Please start ROS2 before creating a deploy env."
        # Static env data
        self.n_drones = len(drones)
        self.pos_limit_low = np.array(pos_limit_low if pos_limit_low else [-5.0, -5.0, 0.0])
        self.pos_limit_high = np.array(pos_limit_high if pos_limit_high else [5.0, 5.0, 3.0])
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        self.takeoff_pos = [drone.get("takeoff_pos", [0.0, 0.0, 0.05]) for drone in drones]
        self.channel = drones[rank]["channel"]
        self.rank = rank
        self.freq = freq
        self.device = jax.devices("cpu")[0]
        assert control_mode in ["state", "attitude", "force_torque", "rotor_vel", "rl_state"], (
            f"Invalid control mode {control_mode}"
        )
        self.control_mode = control_mode
        self.drone_parameters = load_params("first_principles", drones[rank]["drone_model"])
        self.rotor_vel_min, self.rotor_vel_max = (
            np.sqrt(self.drone_parameters["thrust_min"] / self.drone_parameters["rpm2thrust"][2]),
            np.sqrt(self.drone_parameters["thrust_max"] / self.drone_parameters["rpm2thrust"][2]),
        )
        # Establish drone connection
        self._drone_healthy = mp.Event()
        self._drone_healthy.set()
        self.drone = self._connect_to_drone(
            radio_id=rank, radio_channel=drones[rank]["channel"], drone_id=drones[rank]["id"]
        )
        self._last_drone_pos_update = 0  # Last time a position was sent to the drone estimator

        self._ros_connector = ROSConnector(
            estimator_names=self.drone_names,
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )

    def _reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        options = {} if options is None else options

        self._reset_drone()

        if self.control_mode == "attitude":
            # Unlock thrust mode protection by sending a zero thrust command
            self.drone.commander.send_setpoint(0, 0, 0, 0)

        return self.obs(), self.info()

    def _step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment."""
        # Note: We do not send the action to the drone here.
        self.send_action(action)

        drone_pos = np.stack([self._ros_connector.pos[drone] for drone in self.drone_names])
        assert drone_pos.dtype == np.float32, "Drone position must be of type float32"
        drone_quat = np.stack([self._ros_connector.quat[drone] for drone in self.drone_names])
        assert drone_quat.dtype == np.float32, "Drone quaternion must be of type float32"

        # Send vicon position updates to the drone at a fixed frequency irrespective of the env freq
        # Sending too many updates may deteriorate the performance of the drone, hence the limiter
        if (t := time.perf_counter()) - self._last_drone_pos_update > 1 / self.POS_UPDATE_FREQ:
            self.drone.extpos.send_extpose(*drone_pos[self.rank], *drone_quat[self.rank])
            self._last_drone_pos_update = t
        return self.obs(), self.reward(), self.terminated(), self.truncated(), self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the observation of the environment."""
        drone_pos = np.stack([self._ros_connector.pos[drone] for drone in self.drone_names])
        drone_quat = np.stack([self._ros_connector.quat[drone] for drone in self.drone_names])
        drone_vel = np.stack([self._ros_connector.vel[drone] for drone in self.drone_names])
        drone_ang_vel = np.stack([self._ros_connector.ang_vel[drone] for drone in self.drone_names])
        obs = {"pos": drone_pos, "quat": drone_quat, "vel": drone_vel, "ang_vel": drone_ang_vel}
        return obs

    def reward(self) -> float:
        """Compute the reward for the current state.

        Note:
            This is a placeholder reward function. If you want to use reinforcement learning,
            you will need to define your own reward function based on your specific task.

        Returns:
            Reward for the current state.
        """
        return np.stack([0.0 for drone in self.drone_names])  # Placeholder reward

    def terminated(self) -> NDArray:
        """Check if the episode is terminated."""
        terminated = np.zeros(self.n_drones, dtype=bool)
        terminated[self.rank] |= not self._drone_healthy.is_set()

        drone_pos = np.stack([self._ros_connector.pos[drone] for drone in self.drone_names])
        terminated |= np.any((self.pos_limit_low > drone_pos) | (drone_pos > self.pos_limit_high))

        return terminated

    def truncated(self) -> NDArray:
        """Check if the episode is truncated."""
        return np.zeros(self.n_drones, dtype=bool)

    def info(self) -> dict:
        """Return an info dictionary containing additional information about the environment."""
        return {}

    def send_action(self, action: NDArray):
        """Send the action to the drone."""
        if self.control_mode == "attitude":
            pwm = force2pwm(
                action[3], self.drone_parameters["thrust_max"] * 4, self.drone_parameters["pwm_max"]
            )
            pwm = np.clip(pwm, self.drone_parameters["pwm_min"], self.drone_parameters["pwm_max"])
            action = (*np.rad2deg(action[:3]), int(pwm))
            self.drone.commander.send_setpoint(*action)
            self._ros_connector.publish_cmd(action)
        elif self.control_mode == "rotor_vel":
            # convert rotor velocity (RPM) to PWM
            rotor_vel = action
            forces = (
                self.drone_parameters["rpm2thrust"][..., 0]
                + self.drone_parameters["rpm2thrust"][..., 1] * rotor_vel
                + self.drone_parameters["rpm2thrust"][..., 2] * rotor_vel**2
            )
            pwms_normalized = forces / self.drone_parameters["thrust_max"]  # [0.0, 1.0]
            pwms = np.clip(pwms_normalized, 0.0, 1.0)
            # use position setpoint UI for PWM command
            self.drone.commander.send_position_setpoint(*pwms)
        elif self.control_mode == "force_torque":
            # thrust in N -> thrust in PWM
            thrust_pwm = force2pwm(
                action[0], self.drone_parameters["thrust_max"] * 4, self.drone_parameters["pwm_max"]
            )
            thrust_pwm = np.clip(thrust_pwm, self.drone_parameters["pwm_min"], self.drone_parameters["pwm_max"])
            # torque in Nm -> torque in PWM
            L = self.drone_parameters["L"]
            thrust2torque = self.drone_parameters["thrust2torque"]
            tau_x, tau_y, tau_z = action[1], action[2], action[3]
            roll_force = tau_x / L * 2.0
            pitch_force = tau_y / L * 2.0
            yaw_force = tau_z / thrust2torque
            torque_force = np.array([roll_force, pitch_force, yaw_force])
            torque_pwm = force2pwm(torque_force, self.drone_parameters["thrust_max"] * 4, self.drone_parameters["pwm_max"])
            # use attitude setpoint UI for force_torque command
            self.drone.commander.send_position_setpoint(*torque_pwm, thrust_pwm)
        elif self.control_mode == "rl_state":
            pos, vel= action[:3], action[3:6]
            acc = np.zeros(3) # currently we don't have acc command for RL policy
            quat = np.array([0., 0., 0., 1.]) # currently no orientation command for RL policy
            rollrate, pitchrate, yawrate = 0., 0., 0. # currently no rate command for RL policy
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )
        else:
            pos, vel, acc = action[:3], action[3:6], action[6:9]
            # TODO: We currently limit ourselves to yaw rotation only because the simulation is
            # based on the old crazyswarm full_state command definition. Once the simulation does
            # support the real full_state command, we can remove this limitation and use full
            # quaternions as inputs
            quat = R.from_euler("z", action[9]).as_quat()
            rollrate, pitchrate, yawrate = action[10:]
            self.drone.commander.send_full_state_setpoint(
                pos, vel, acc, quat, rollrate, pitchrate, yawrate
            )
            # TODO: The estimators can't handle state commands, so we simply don't send anything
            # Make sure to use the legacy estimator with the state interface

    def _connect_to_drone(self, radio_id: int, radio_channel: int, drone_id: int) -> Crazyflie:
        cflib.crtp.init_drivers()
        uri = f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7" + f"{drone_id:02x}".upper()

        power_switch = PowerSwitch(uri)
        power_switch.stm_power_cycle()
        time.sleep(2)

        drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
        event = mp.Event()

        def connect_callback(_: str):
            event.set()

        drone.fully_connected.add_callback(connect_callback)
        drone.disconnected.add_callback(lambda _: self._drone_healthy.clear())
        drone.connection_failed.add_callback(
            lambda _, msg: logger.warning(f"Connection failed: {msg}")
        )
        drone.connection_lost.add_callback(lambda _, msg: logger.warning(f"Connection lost: {msg}"))
        drone.open_link(uri)

        logger.info(f"Waiting for drone {drone_id} to connect...")
        connected = event.wait(10.0)
        if not connected:
            raise TimeoutError("Timed out while waiting for the drone.")
        logger.info(f"Drone {drone_id} connected to {uri}")

        return drone

    def _reset_drone(self):
        # Arm the drone
        self.drone.platform.send_arming_request(True)
        self._apply_drone_settings()
        pos = self._ros_connector.pos[self.drone_name]
        # Reset Kalman filter values
        self.drone.param.set_value("kalman.initialX", pos[0])
        self.drone.param.set_value("kalman.initialY", pos[1])
        self.drone.param.set_value("kalman.initialZ", pos[2])
        quat = self._ros_connector.quat[self.drone_name]
        yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
        self.drone.param.set_value("kalman.initialYaw", yaw)
        self.drone.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.drone.param.set_value("kalman.resetEstimation", "0")

    def _apply_drone_settings(self):
        """Apply firmware settings to the drone.

        Note:
            These settings are also required to make the high-level drone commander work properly.
        """
        # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
        self.drone.param.set_value("stabilizer.estimator", 2)
        time.sleep(0.1)  # TODO: Maybe remove
        # enable/disable tumble control. Required 0 for agressive maneuvers
        self.drone.param.set_value("supervisor.tmblChckEn", 1)
        # Choose controller: 1: PID 2:Mellinger 6: Rotor Velocity 7: Force Torque 8: RL State
        self.drone.param.set_value("stabilizer.controller", 8)
        # rate: 0, angle: 1
        self.drone.param.set_value("flightmode.stabModeRoll", 1)
        self.drone.param.set_value("flightmode.stabModePitch", 1)
        self.drone.param.set_value("flightmode.stabModeYaw", 1)
        time.sleep(0.1)  # Wait for settings to be applied

    def _move_to_start(self, start_controller: Controller):
        # use attitude interface to fly to start
        save_control_mode = self.control_mode  # TODO: this is ugly
        self.control_mode = "attitude"
        self.drone.commander.send_setpoint(0, 0, 0, 0)
        # Move the drone to the start position
        pos = self._ros_connector.pos[self.drone_name]
        FREQ = 50  # Hz
        START_HEIGHT = max(self.takeoff_pos[self.rank][2], 0.2)  # m
        TAKEOFF_DURATION = max(START_HEIGHT / 0.5, 0.5)  # s
        MOVE_DURATION = max(
            np.linalg.norm(self.takeoff_pos[self.rank][:2] - pos[:2]) / 1.0, 1.0
        )  # s
        HOVER_DURATION = 1.0  # get ready for episode
        p0 = np.asarray(pos, dtype=np.float32)
        p1 = np.array([pos[0], pos[1], START_HEIGHT], dtype=np.float32)
        p2 = np.asarray(self.takeoff_pos[self.rank], dtype=np.float32)

        takeoff_steps = int(np.ceil(TAKEOFF_DURATION * FREQ))
        move_steps = int(np.ceil(MOVE_DURATION * FREQ))
        hover_steps = int(np.ceil(HOVER_DURATION * FREQ))

        seg_takeoff = np.linspace(p0, p1, num=takeoff_steps, endpoint=True)
        seg_move = np.linspace(p1, p2, num=move_steps, endpoint=True)
        seg_hover = np.broadcast_to(p2, (hover_steps, 3))

        traj = np.concatenate([seg_takeoff, seg_move, seg_hover], axis=0)  # (T, 3)
        start_controller.trajectory = traj

        while rclpy.ok():
            t_loop = time.perf_counter()
            obs = self.obs()
            obs = {k: v[0] for k, v in obs.items()}
            action = start_controller.compute_control(obs, None)
            next_obs, _, terminated, truncated, _ = self._step(action)
            next_obs, terminated, truncated = (
                {k: v[0, ...] for k, v in next_obs.items()},
                terminated[0],
                truncated[0],
            )
            controller_finished = start_controller.step_callback(None, None, None, None, None, None)
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / FREQ):
                time.sleep(1 / FREQ - dt)
            else:
                exc = dt - 1 / FREQ
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")

        self.control_mode = save_control_mode
        if self.control_mode == "rotor_vel":
            self.drone.param.set_value("stabilizer.controller", 6)
        elif self.control_mode == "force_torque":
            self.drone.param.set_value("stabilizer.controller", 7)
        elif self.control_mode == "rl_state":
            self.drone.param.set_value("stabilizer.controller", 8)
            print("Running RL state controller")

    def _return_to_start(self):
        # Enable high-level functions of the drone and disable low-level control access
        self.drone.commander.send_stop_setpoint()
        self.drone.commander.send_notify_setpoint_stop()
        self.drone.param.set_value("commander.enHighLevel", "1")
        self.drone.param.set_value("stabilizer.controller", 2) # switch back to mellinger for return
        self.drone.platform.send_arming_request(True)
        # Fly back to the start position
        pos = self._ros_connector.pos[self.drone_name]
        vel = self._ros_connector.vel[self.drone_name]
        RETURN_HEIGHT = 1.75  # m
        BREAKING_DISTANCE = 1.0  # m
        BREAKING_DURATION = 3.0  # s
        RETURN_DURATION = max(
            np.linalg.norm(self.takeoff_pos[self.rank][:2] - pos[:2]) / 1.0, 2.0
        )  # s
        LAND_DURATION = 3.0  # s

        def wait_for_action(dt: float):
            tstart = time.perf_counter()
            # Wait for the action to be completed and send the current position to the drone
            while time.perf_counter() - tstart < dt:
                if not rclpy.ok():
                    raise RuntimeError("ROS has already stopped")
                if not self._drone_healthy.is_set():
                    raise RuntimeError("Drone connection lost")
                obs = self.obs()
                self.drone.extpos.send_extpose(*obs["pos"][self.rank], *obs["quat"][self.rank])
                time.sleep(0.05)

        # This quick check prevents us from engaging the return controller if we havent even started yet.
        if pos[2] < 0.2:
            return
        break_pos = pos + vel / np.linalg.norm(vel) * BREAKING_DISTANCE
        break_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*break_pos, 0, BREAKING_DURATION)
        wait_for_action(BREAKING_DURATION)
        return_pos = self.takeoff_pos[self.rank]
        return_pos[2] = RETURN_HEIGHT
        self.drone.high_level_commander.go_to(*return_pos, 0, RETURN_DURATION)
        wait_for_action(RETURN_DURATION)
        return_pos[2] = 0.05
        self.drone.high_level_commander.go_to(*return_pos, 0, LAND_DURATION)
        wait_for_action(LAND_DURATION)

    def close(self):
        """Close the environment.

        If the drone has finished the track, it will try to return to the start position.
        Irrespective of succeeding or not, the drone will be stopped immediately afterwards or in
        case of errors, and close the connections to the ROSConnector.
        """
        try:
            self._return_to_start()
        finally:
            try:
                # Kill the drone
                pk = CRTPPacket()
                pk.port = CRTPPort.LOCALIZATION
                pk.channel = Localization.GENERIC_CH
                pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
                self.drone.send_packet(pk)
                self.drone.close_link()
            finally:
                # Close all ROS connections
                self._ros_connector.close()


# region Single Drone Env
class RealDroneEnv(RealDroneCoreEnv, Env):
    """A Gymnasium environment for controlling a real Crazyflie drone.

    This environment provides a standardized interface for deploying drone algorithms on
    physical hardware. It handles communication with the drone through the cflib library and tracks
    the drone's position using a motion capture system via ROS2.

    The environment provides basic drone control functionality with support for both state-based
    and attitude-based control modes.

    Features:
    - Interfaces with physical Crazyflie drones through radio communication
    - Tracks drone position and orientation using motion capture data via ROS2
    - Supports both state-based and attitude-based control modes
    - Handles automatic return-to-home behavior when the environment is closed

    Note:
        This environment is designed for single-drone control. For multi-drone scenarios, use the
        :class:`~.RealMultiDroneEnv` class instead.
    """

    def __init__(
        self,
        drones: list[dict[str, int]],
        freq: int,
        pos_limit_low: list[float] | None = None,
        pos_limit_high: list[float] | None = None,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the drone environment.

        Action space:
            The action space is a single action vector for the drone with the environment rank.
            See :class:`~.RealDroneCoreEnv` for more information. Depending on the control mode, it
            is either a 13D desired drone state setpoint, or a 4D desired attitude and collective
            thrust setpoint.

        Observation space:
            The observation space is a dictionary containing the state of the drone.
            It contains position, quaternion, velocity, and angular velocity.

        Note:
            rclpy must be initialized before creating this environment.

        Args:
            drones: List of all drones, including their channel and id.
            freq: Environment step frequency.
            pos_limit_low: Lower position limits for safety [x, y, z].
            pos_limit_high: Upper position limits for safety [x, y, z].
            control_mode: Control mode of the drone.
        """
        super().__init__(
            drones=drones,
            rank=0,
            freq=freq,
            pos_limit_low=pos_limit_low,
            pos_limit_high=pos_limit_high,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        obs, info = self._reset(seed=seed, options=options)
        return {k: v[0, ...] for k, v in obs.items()}, info

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment.

        Args:
            action: Action to be taken by the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        return {k: v[0, ...] for k, v in obs.items()}, reward[0], terminated[0], truncated[0], info


# region Multi Drone Env
class RealMultiDroneEnv(RealDroneCoreEnv, Env):
    """A Gymnasium environment for controlling a specific drone in a multi-drone physical environment.

    This environment extends the functionality of `RealDroneCoreEnv` to support multi-drone
    scenarios. Each instance of this environment controls a single drone identified by its rank, but
    maintains awareness of all drones in the environment. This allows for coordinated multi-drone
    deployments where each drone runs in a separate process with its own controller.

    The environment handles communication with the specific drone through cflib and tracks all
    drones' positions using a motion capture system via ROS2. It provides observations that include
    the state of all drones, allowing controllers to implement collision avoidance or cooperative
    strategies.

    Features:
    - Controls a specific drone in a multi-drone environment based on its rank
    - Tracks all drones' positions and states via ROS2
    - Supports both state-based and attitude-based control modes
    - Handles automatic return-to-home behavior when the environment is closed

    Action space:
        The action space is a **single** action vector for the drone with the environment rank.
        See :class:`~.RealDroneCoreEnv` for more information.

    Warning:
        The action space differs from the action space of the simulated counterpart. This deviation
        is necessary to run different controller types at different frequencies that asynchronously
        publish their commands to the drone.

    Observation space:
        The observation space is a dictionary containing the state of all drones in the environment.

    Note:
        Each instance of this environment controls only one drone (specified by rank), but provides
        observations for all drones in the environment. This allows us to run controllers at different
        frequencies for different drones. Consequently the step method applies actions only to the
        drone with the specified rank.
    """

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        pos_limit_low: list[float] | None = None,
        pos_limit_high: list[float] | None = None,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the multi-drone environment.

        Args:
            drones: List of all drones, including their channel and id.
            rank: Rank of the drone that is controlled by this environment.
            freq: Environment step frequency.
            pos_limit_low: Lower position limits for safety [x, y, z].
            pos_limit_high: Upper position limits for safety [x, y, z].
            control_mode: Control mode of the drone.
        """
        super().__init__(
            drones=drones,
            rank=rank,
            freq=freq,
            pos_limit_low=pos_limit_low,
            pos_limit_high=pos_limit_high,
            control_mode=control_mode,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and return the initial observation and info."""
        return self._reset(seed=seed, options=options)

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a step in the environment.

        Note:
            The action is applied only to the drone with the environment rank!

        Args:
            action: Action to be taken by the drone.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, terminated, truncated, info = self._step(action)
        return obs, reward[self.rank], terminated[self.rank], truncated[self.rank], info
