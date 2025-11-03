"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>

"""

from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import os
import struct
import time
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import cflib
import numpy as np
import rclpy
from cflib.crazyflie import Crazyflie, Localization
from cflib.crazyflie.log import LogConfig
from cflib.crtp import init_drivers
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.utils.power_switch import PowerSwitch
from drone_estimators.ros_nodes.ros2_connector import ROSConnector

os.environ["SCIPY_ARRAY_API"] = "1"
from scipy.spatial.transform import Rotation as R

from rl_diffsim.control.attitude_rl import AttitudeRL
from scripts.utils import constants

if TYPE_CHECKING:
    from numpy.typing import NDArray

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    controller = AttitudeRL(initial_obs, initial_info)
    _last_drone_pos_update = time.perf_counter()

    try:
        while rclpy.ok():
            t_loop = time.perf_counter()

            obs = get_obs()
            if obs["is_outdated"]:
                logger.info("Measurements outdated. Emergency stop.")
                break

            action = controller.compute_control(obs)
            send_action(action)

            controller_finished = controller.step_callback()

            if controller_finished:
                logger.info("Controller terminated. Returning home.")
                return_to_start()
                break

            # Send vicon position updates to the drone at a fixed frequency irrespective of the env freq
            # This is important to remove drift of the onboard estimators!
            # Sending too many updates may deteriorate the performance of the drone, hence the limiter
            if (t := time.perf_counter()) - _last_drone_pos_update > 1 / initial_info[
                "pos_update_freq"
            ]:
                drone.extpos.send_extpose(*get_obs()["pos"], *get_obs()["quat"])
                _last_drone_pos_update = t

            if (dt := (time.perf_counter() - t_loop)) < (1 / initial_info["env_freq"]):
                time.sleep(1 / initial_info["env_freq"] - dt)
            else:
                exc = dt - 1 / initial_info["env_freq"]
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")

    finally:  # Kill the drone
        try:
            pk = CRTPPacket()
            pk.port = CRTPPort.LOCALIZATION
            pk.channel = Localization.GENERIC_CH
            pk.data = struct.pack("<B", Localization.EMERGENCY_STOP)
            drone.send_packet(pk)
            drone.close_link()
        finally:
            # Close all ROS connections
            ros_connector.close()


def get_obs():
    obs = {
        "pos": ros_connector.pos[drone_name],
        "quat": ros_connector.quat[drone_name],
        "vel": ros_connector.vel[drone_name],
        "ang_vel": ros_connector.ang_vel[drone_name],
        "is_outdated": ros_connector.is_outdated[drone_name],
    }
    obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
    return obs


def thrust2pwm(thrust: float) -> float:
    pwm = thrust / constants["THRUST_MAX"] / 4 * constants["PWM_MAX"]
    return np.clip(pwm, constants["PWM_MIN"], constants["PWM_MAX"])


def send_action(action: NDArray):
    """Send the action to the drone."""
    action = (*np.rad2deg(action[:3]), int(thrust2pwm(action[3])))
    drone.commander.send_setpoint(*action)
    ros_connector.publish_cmd(action)


def connect_to_drone(radio_id: int, radio_channel: int, drone_id: int) -> Crazyflie:
    cflib.crtp.init_drivers()
    uri = f"radio://{radio_id}/{radio_channel}/2M/E7E7E7E7" + f"{drone_id:02x}".upper()

    power_switch = PowerSwitch(uri)
    power_switch.stm_power_cycle()
    time.sleep(2)

    drone = Crazyflie(rw_cache=str(Path(__file__).parent / ".cache"))
    connected_event = mp.Event()

    drone.fully_connected.add_callback(lambda _: connected_event.set())
    drone.disconnected.add_callback(lambda _: drone_healthy.clear())
    drone.connection_failed.add_callback(lambda _, msg: logger.warning(f"Connection failed: {msg}"))
    drone.connection_lost.add_callback(lambda _, msg: logger.warning(f"Connection lost: {msg}"))
    # drone.console.receivedChar.add_callback(
    #     lambda msg: logger.info(f"drone: {msg.strip().replace('\n', '').replace('\r', '')}")
    # )
    drone.open_link(uri)

    logger.info(f"Waiting for drone {drone_id} to connect...")
    connected = connected_event.wait(5.0)
    assert connected, f"Drone {drone_id} failed to connect to {uri}"
    logger.info(f"Drone {drone_id} connected to {uri}")

    return drone


def reset_drone():
    init_drivers()
    drone.platform.send_arming_request(True)
    apply_drone_settings()
    pos = ros_connector.pos[drone_name]
    # Reset Kalman filter values
    drone.param.set_value("kalman.initialX", pos[0])
    drone.param.set_value("kalman.initialY", pos[1])
    drone.param.set_value("kalman.initialZ", pos[2])
    quat = ros_connector.quat[drone_name]
    yaw = R.from_quat(quat).as_euler("xyz", degrees=False)[2]
    drone.param.set_value("kalman.initialYaw", yaw)
    drone.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    drone.param.set_value("kalman.resetEstimation", "0")


def apply_drone_settings():
    """Apply firmware settings to the drone.

    Note:
        These settings are also required to make the high-level drone commander work properly.
    """
    # Estimator setting;  1: complementary, 2: kalman -> Manual test: kalman significantly better!
    drone.param.set_value("stabilizer.estimator", 2)
    time.sleep(0.1)  # TODO: Maybe remove
    # enable/disable tumble control. Required 0 for agressive maneuvers
    drone.param.set_value("supervisor.tmblChckEn", 1)
    # Choose controller: 1: PID; 2:Mellinger
    drone.param.set_value("stabilizer.controller", 2)
    # rate: 0, angle: 1
    drone.param.set_value("flightmode.stabModeRoll", 1)
    drone.param.set_value("flightmode.stabModePitch", 1)
    drone.param.set_value("flightmode.stabModeYaw", 1)
    time.sleep(0.1)  # Wait for settings to be applied


def return_to_start():
    # Enable high-level functions of the drone and disable low-level control access
    drone.commander.send_stop_setpoint()
    drone.commander.send_notify_setpoint_stop()  # Tell drone to ignore low level setpoints
    drone.param.set_value("commander.enHighLevel", 1)
    drone.platform.send_arming_request(True)
    # drone.commander.send_setpoint(0, 0, 0, 0)
    # Fly back to the start position
    RETURN_HEIGHT = 1.75  # m
    BREAKING_DISTANCE = 1.0  # m
    BREAKING_DURATION = 3.0  # s
    RETURN_DURATION = 5.0  # s
    LAND_DURATION = 3.0  # s

    def wait_for_action(dt: float):
        tstart = time.perf_counter()
        # Wait for the action to be completed and send the current position to the drone
        while time.perf_counter() - tstart < dt:
            obs = get_obs()
            drone.extpos.send_extpose(*obs["pos"], *obs["quat"])
            time.sleep(1 / initial_info["pos_update_freq"])
            if not drone_healthy.is_set():
                raise RuntimeError("Drone connection lost")

    pos = ros_connector.pos[drone_name]
    vel = ros_connector.vel[drone_name]
    vel_norm = np.linalg.norm(vel)
    break_pos = pos + vel / vel_norm * np.clip(vel_norm, 0.1, BREAKING_DISTANCE)
    break_pos[2] = RETURN_HEIGHT
    drone.high_level_commander.go_to(*break_pos, 0, BREAKING_DURATION)
    wait_for_action(BREAKING_DURATION)
    return_pos = copy(initial_info["pos_landing"])
    return_pos[2] = RETURN_HEIGHT
    drone.high_level_commander.go_to(*return_pos, 0, RETURN_DURATION)
    wait_for_action(RETURN_DURATION)
    return_pos[2] = initial_info["pos_landing"][2]
    drone.high_level_commander.go_to(*return_pos, 0, LAND_DURATION)
    wait_for_action(LAND_DURATION)

def plot_eval_trajectory(actions: NDArray, pos: NDArray, goal: NDArray, rpy: NDArray, save_path: str = "plot.png"):
    import matplotlib
    matplotlib.use("Agg")  # render to raster images
    import matplotlib.pyplot as plt

    # Plot the actions over time
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    action_labels = ["Roll", "Pitch", "Yaw", "Thrust"]
    for i in range(4):
        axes[i].plot(actions[:, i])
        axes[i].set_title(f"{action_labels[i]} Command")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Action Value")
        axes[i].grid(True)

    # Plot position components
    position_labels = ["X Position", "Y Position", "Z Position"]
    for i in range(3):
        axes[4 + i].plot(pos[:, i])
        axes[4 + i].set_title(position_labels[i])
        axes[4 + i].set_xlabel("Time Step")
        axes[4 + i].set_ylabel("Position (m)")
        axes[4 + i].grid(True)
    # Plot goal position components in same plots
    for i in range(3):
        axes[4 + i].plot(goal[:, i], linestyle="--")
        axes[4 + i].legend(["Position", "Goal"])
    # Plot error in position
    pos_err = np.linalg.norm(pos - goal, axis=1)
    axes[7].plot(pos_err)
    axes[7].set_title("Position Error")
    axes[7].set_xlabel("Time Step")
    axes[7].set_ylabel("Error (m)")
    axes[7].grid(True)

    # Plot angle components (roll, pitch, yaw)
    rpy_labels = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        axes[8 + i].plot(rpy[:, i])
        axes[8 + i].set_title(f"{rpy_labels[i]} Angle")
        axes[8 + i].set_xlabel("Time Step")
        axes[8 + i].set_ylabel("Angle (rad)")
        axes[8 + i].grid(True)

    # compute RMSE for position
    rmse_pos = np.sqrt(np.mean(pos_err**2))
    axes[11].text(0.1, 0.5, f"Position RMSE: {rmse_pos*1000:.3f} mm", fontsize=14)
    axes[11].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)

    return fig, axes, rmse_pos

if __name__ == "__main__":
    drone_id = 52
    drone_name = f"cf{drone_id}"

    rclpy.init()
    ros_connector = ROSConnector(
        estimator_names=[drone_name], cmd_topic=f"/drones/{drone_name}/command", timeout=10.0
    )

    drone_healthy = mp.Event()
    drone_healthy.set()
    drone = connect_to_drone(0, 80, drone_id)
    time.sleep(0.5)
    reset_drone()
    drone.commander.send_setpoint(0, 0, 0, 0)  # unlock drone

    initial_obs = get_obs()
    initial_info = {
        "drone_mass": constants["MASS"],
        "pos_landing": [-1.5, 0.5, 0.05],
        "env_freq": 200,  # = control frequency!
        "low_level_ctrl_freq": 500,
        "pos_update_freq": 10,
    }

    # Settings for data collection:
    # initial_info["pos_hover"] = [-0.5, 0.0, 1.5]  # center of trajectory
    initial_info["pos_hover"] = [0.0, 0.0, 1.0]  # center of trajectory
    initial_info["time_hover"] = 2.0  # s, hovering time between trajectories
    initial_info["time_connection"] = 4.0  # s, time for connecting trajectories
    # initial_info["scaling"] = [5.5, 3.5, 2.0]  # size of trajectory in xyz
    initial_info["scaling"] = [2.0, 2.0, 1.0]  # size of trajectory in xyz
    initial_info["axis_order"] = "xyz"
    # Type of trajectory, can be one of valid_traj_type of generator
    # If set to sysid_train or sysid_valid, a set of trajectories get flown
    # TODO write down which exactly
    # Alternatively, a list of tuples of the form (traj_type, cycles, cycle_time, scaling, axis_order)
    # Hints on timings and repetitions:
    # melon 4.5 (12 cycles), figure8 5.5, figure8_bent 9, flower1 10 [4.5, 2.5, 1.5]!,
    # flower2 15, pringle 7.0, spiral 3.0 (6 cycles), lissajous 18.0 (1 cycle)
    # chirp_1d 10.0 3 [1.0, 1.0, 1.0], spiral 1.5 15 [1.5, 1.5, 0], figure8 4.5 (6.5 for yx) [5.0, 3.0, 1.0]
    initial_info["traj_type"] = "figure8"  # sysid_train, sysid_valid, _thrust, lol_sysid
    initial_info["planner_cycle_time"] = 10.0  # time per cycle
    initial_info["planner_cycles"] = 1  # number of cycles

    main()
