#!/usr/bin/env python3
import os
import argparse

_parser = argparse.ArgumentParser(description="YAM ROS Bridge")
_parser.add_argument("--config", "-c", required=True, metavar="PATH",
                     help="config.yaml 路径")
_args = _parser.parse_args()
os.environ["YAM_ROS_CONFIG"] = os.path.abspath(_args.config)

import time
import threading
import numpy as np
import roslibpy

from config import (
    CAN_CHANNEL, GRIPPER_TYPE, SITE_NAME,
    ROSBRIDGE_HOST, ROSBRIDGE_PORT,
    SAFE_JOINT_POS, SAFE_MOVE_TIME,
)
from shared_state import SharedState
from ros_bridge import RosSubscribers, publisher_thread
from control_loop import control_loop

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


def _parse_gripper_type(s: str) -> GripperType:
    mapping = {
        "linear_4310":        GripperType.LINEAR_4310,
        "linear_3507":        GripperType.LINEAR_3507,
        "crank_4310":         GripperType.CRANK_4310,
        "no_gripper":         GripperType.NO_GRIPPER,
        "yam_teaching_handle": GripperType.YAM_TEACHING_HANDLE,
    }
    key = s.lower()
    if key not in mapping:
        raise ValueError(f"未知 gripper_type: {s}，可选: {list(mapping)}")
    return mapping[key]


def init_robot(gripper_type: GripperType):
    print("[Init] 初始化 YAM 机械臂...")
    robot = get_yam_robot(
        channel=CAN_CHANNEL,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
    )

    has_gripper = gripper_type not in (GripperType.NO_GRIPPER, GripperType.YAM_TEACHING_HANDLE)
    safe_q = np.array(SAFE_JOINT_POS, dtype=float)
    if has_gripper:
        safe_q = np.append(safe_q, 1.0)  # 初始夹爪全开

    print("[Init] 移动到安全位置...")
    robot.move_joints(safe_q, time_interval_s=SAFE_MOVE_TIME)
    print("[Init] ✓ 已到达安全位置")
    return robot, has_gripper


def init_kinematics() -> Kinematics:
    # 运动学模型只需要 arm，不含夹爪
    xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    return Kinematics(xml_path, SITE_NAME)


def init_state(robot, kin: Kinematics) -> SharedState:
    obs = robot.get_observations()
    arm_pos  = obs["joint_pos"]
    arm_vel  = obs["joint_vel"]
    grip_pos = float(obs.get("gripper_pos", np.array([1.0]))[0])
    grip_vel = float(obs.get("gripper_vel", np.array([0.0]))[0])

    T = kin.fk(arm_pos)

    state = SharedState(n_arm_joints=len(arm_pos))
    state.reset_target_to(T[:3, 3], T[:3, :3])
    state.last_valid_joint_pos = arm_pos.copy()
    state.set_robot_state(arm_pos, arm_vel, T[:3, 3], T[:3, :3],
                          gripper_pos=grip_pos, gripper_vel=grip_vel)
    print(f"[Init] 初始末端位置: {[f'{v:.3f}' for v in T[:3, 3]]}")
    return state


def main():
    print("=" * 60)
    print("YAM ROS Bridge 节点")
    print("=" * 60)

    gripper_type = _parse_gripper_type(GRIPPER_TYPE)
    robot, has_gripper = init_robot(gripper_type)
    kin   = init_kinematics()
    state = init_state(robot, kin)

    # 连接 rosbridge
    print(f"\n[ROS] 连接 {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}...")
    ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    ros.run()
    if not ros.is_connected:
        raise RuntimeError("rosbridge 连接失败")
    print("[ROS] ✓ 已连接")

    subscribers = RosSubscribers(ros, state)
    stop_event  = threading.Event()

    threads = [
        threading.Thread(
            target=publisher_thread,
            args=(ros, state, stop_event),
            daemon=True, name="StatePublisher",
        ),
        threading.Thread(
            target=control_loop,
            args=(robot, kin, state, stop_event, has_gripper),
            daemon=True, name="ControlLoop",
        ),
    ]
    for t in threads:
        t.start()

    print("\n[Main] 运行中，Ctrl+C 退出\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] 退出信号...")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        subscribers.unsubscribe_all()
        ros.terminate()
        print("[Main] 返回安全位置...")
        safe_q = np.array(SAFE_JOINT_POS, dtype=float)
        if has_gripper:
            safe_q = np.append(safe_q, 1.0)
        robot.move_joints(safe_q, time_interval_s=SAFE_MOVE_TIME)
        robot.close()
        print("[Main] 完成")


if __name__ == "__main__":
    main()
