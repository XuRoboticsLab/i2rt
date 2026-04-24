#!/usr/bin/env python3
"""
机械臂控制端 - 双臂版 (运行在机械臂 conda 环境, i2rt YAM)

双臂控制:
  右手 Grip按住:    右臂跟随右手柄
  左手 Grip按住:    左臂跟随左手柄
  各自独立校准、独立回零、独立夹爪控制
  左右臂可以同时或单独操作
  Y键:              紧急停止 (双臂同时)
  双击 Grip:        对应臂回到安全位置

运行: conda activate robot_env && python i2rt_xr_cartesian_pos_control_2arm.py
先启动 xr_server_2arm.py, 再启动本程序。
"""

import time
import argparse
import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

from xr_robot_shm import XRSharedMemReader


# ======================== 命令行参数 ========================

parser = argparse.ArgumentParser(description="i2rt YAM 双臂 VR 遥操作")
parser.add_argument("--can-right",     default="can_right",    help="右臂 CAN 通道 (默认: can_right)")
parser.add_argument("--can-left",      default="can_left",     help="左臂 CAN 通道 (默认: can_left)")
parser.add_argument("--gripper-right", default="linear_4310",
                    choices=["linear_4310", "linear_3507", "crank_4310", "no_gripper"],
                    help="右臂夹爪类型 (默认: linear_4310)")
parser.add_argument("--gripper-left",  default="linear_4310",
                    choices=["linear_4310", "linear_3507", "crank_4310", "no_gripper"],
                    help="左臂夹爪类型 (默认: linear_4310)")
parser.add_argument("--site",          default="grasp_site",   help="末端 site 名称 (默认: grasp_site)")
args = parser.parse_args()

_GRIPPER_MAP = {
    "linear_4310": GripperType.LINEAR_4310,
    "linear_3507":  GripperType.LINEAR_3507,
    "crank_4310":   GripperType.CRANK_4310,
    "no_gripper":   GripperType.NO_GRIPPER,
}

SAFE_JOINT_POS = np.zeros(6)
SAFE_MOVE_TIME = 3.0


# ======================== 坐标系映射 ========================

XR_TO_ROBOT_POS = np.array([
    [+0.0, +0.0, -1.0],   # 机械臂 X ← XR -Z
    [-1.0, +0.0, +0.0],   # 机械臂 Y ← XR -X
    [+0.0, +1.0, +0.0],   # 机械臂 Z ←  XR Y
])

XR_TO_ROBOT_ROT = np.array([
    [+0.0, +0.0, -1.0],   # 机械臂 X ← XR -Z
    [-1.0, +0.0, +0.0],   # 机械臂 Y ← XR -X
    [+0.0, +1.0, +0.0],   # 机械臂 Z ←  XR Y
])

POS_SCALE       = 1.0
MAX_DELTA_POS   = 0.3
GRIP_THRESHOLD  = 0.5
DOUBLE_TAP_WINDOW = 0.4


# ======================== 工具函数 ========================

def quat_to_rotation_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)],
    ])


def xr_pose_to_pos_rot(pose_7d):
    pos = np.array(pose_7d[:3], dtype=np.float64)
    rot = quat_to_rotation_matrix(pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6])
    return pos, rot


# ======================== 单臂控制器 ========================

class ArmController:
    """单条机械臂的遥操作控制器 (左右臂各一个实例)"""

    def __init__(self, robot, kin: Kinematics, has_gripper: bool, name: str = "arm"):
        self.robot       = robot
        self.kin         = kin
        self.has_gripper = has_gripper
        self.name        = name

        self.is_calibrated = False
        self.is_resetting  = False

        self.xr_init_pos    = None
        self.xr_init_rot    = None
        self.robot_init_pos = None
        self.robot_init_rot = None

        obs = robot.get_observations()
        self.last_valid_arm_q = obs["joint_pos"].copy()
        self.gripper_pos      = float(obs.get("gripper_pos", np.array([1.0]))[0])

        self.ik_fail_count  = 0
        self.ik_total_count = 0

        self.max_joint_delta_per_step = 0.05   # rad/step

        # Grip 边沿检测 (机械臂端独立做，与 xr_server 端对称)
        self.prev_grip_pressed   = False
        self.last_grip_press_time = 0.0

    def calibrate(self, xr_pose_7d):
        print(f"\n--- [{self.name}] 校准 ---")
        self.xr_init_pos, self.xr_init_rot = xr_pose_to_pos_rot(xr_pose_7d)

        obs   = self.robot.get_observations()
        arm_q = obs["joint_pos"]
        T     = self.kin.fk(arm_q)
        self.robot_init_pos = T[:3, 3].copy()
        self.robot_init_rot = T[:3, :3].copy()

        self.last_valid_arm_q = arm_q.copy()
        self.is_calibrated    = True
        self.ik_fail_count    = 0
        self.ik_total_count   = 0

        print(f"  手柄: {self.xr_init_pos.round(4)}")
        print(f"  机械臂: {self.robot_init_pos.round(4)}")

    def compute_target(self, xr_pose_7d):
        xr_pos, xr_rot = xr_pose_to_pos_rot(xr_pose_7d)

        delta_pos_xr    = xr_pos - self.xr_init_pos
        delta_pos_robot = XR_TO_ROBOT_POS @ delta_pos_xr * POS_SCALE
        if np.linalg.norm(delta_pos_robot) > MAX_DELTA_POS:
            delta_pos_robot = np.zeros(3)
        target_pos = self.robot_init_pos + delta_pos_robot

        delta_rot_xr    = xr_rot @ self.xr_init_rot.T
        delta_rot_robot = XR_TO_ROBOT_ROT @ delta_rot_xr @ XR_TO_ROBOT_ROT.T
        target_rot      = delta_rot_robot @ self.robot_init_rot

        return target_pos, target_rot

    def step(self, xr_pose_7d, trigger_value):
        target_pos, target_rot = self.compute_target(xr_pose_7d)

        target_T = np.eye(4)
        target_T[:3, :3] = target_rot
        target_T[:3,  3] = target_pos

        self.ik_total_count += 1
        success, q_sol = self.kin.ik(
            target_T, args.site,
            init_q=self.last_valid_arm_q,
        )

        if success:
            delta = q_sol - self.last_valid_arm_q
            max_d = np.max(np.abs(delta))
            if max_d > self.max_joint_delta_per_step:
                q_sol = self.last_valid_arm_q + delta * (self.max_joint_delta_per_step / max_d)
            self.last_valid_arm_q = q_sol.copy()
        else:
            self.ik_fail_count += 1

        # trigger=1 → 夹爪闭合(0.0)，trigger=0 → 夹爪张开(1.0)
        self.gripper_pos = 1.0 - trigger_value
        self._send_control()

        return target_pos, success

    def hold_position(self):
        self._send_control()

    def move_to_zero(self):
        """每控制周期调用一次，非阻塞地逐步回到安全位置。返回 True 表示仍在运动。"""
        delta    = SAFE_JOINT_POS - self.last_valid_arm_q
        max_step = 0.002   # rad/step

        if np.max(np.abs(delta)) < 0.01:
            self.last_valid_arm_q = SAFE_JOINT_POS.copy()
            self._send_control()
            return False

        self.last_valid_arm_q = self.last_valid_arm_q + np.clip(delta, -max_step, max_step)
        self._send_control()
        return True

    def handle_grip(self, grip_value, xr_pose_7d, trigger_value):
        """处理单条臂的完整 Grip 逻辑，返回 (target_pos | None, ik_ok)"""
        grip_pressed = grip_value > GRIP_THRESHOLD

        # 上升沿
        if grip_pressed and not self.prev_grip_pressed:
            now = time.time()
            if now - self.last_grip_press_time < DOUBLE_TAP_WINDOW:
                self.is_resetting  = True
                self.is_calibrated = False
                print(f"\n[{self.name}] 双击 Grip → 回到安全位置!")
            else:
                self.is_resetting = False
                self.calibrate(xr_pose_7d)
            self.last_grip_press_time = now

        # 下降沿
        if not grip_pressed and self.prev_grip_pressed:
            print(f"\n[{self.name}] Grip 松开 → 停止")

        self.prev_grip_pressed = grip_pressed

        # 回零中
        if self.is_resetting:
            still_moving = self.move_to_zero()
            if not still_moving:
                self.is_resetting = False
                print(f"\n[{self.name}] 已到达安全位置!")
            return None, True

        # Grip 按住且已校准 → 跟踪
        if grip_pressed and self.is_calibrated:
            return self.step(xr_pose_7d, trigger_value)

        # 未激活 → 保持位置
        self.hold_position()
        return None, True

    def _send_control(self):
        if self.has_gripper:
            full_q = np.append(self.last_valid_arm_q, self.gripper_pos)
        else:
            full_q = self.last_valid_arm_q
        self.robot.command_joint_pos(full_q)

    @property
    def ik_success_rate(self):
        if self.ik_total_count == 0:
            return 100.0
        return (1 - self.ik_fail_count / self.ik_total_count) * 100


# ======================== 主函数 ========================

def main():
    print("=" * 60)
    print("i2rt YAM 机械臂控制端 (双臂)")
    print("=" * 60)

    # ---- 初始化双臂 ----
    gripper_type_R = _GRIPPER_MAP[args.gripper_right]
    gripper_type_L = _GRIPPER_MAP[args.gripper_left]
    has_gripper_R  = gripper_type_R != GripperType.NO_GRIPPER
    has_gripper_L  = gripper_type_L != GripperType.NO_GRIPPER

    print(f"\n初始化右臂 (CAN: {args.can_right}, 夹爪: {args.gripper_right})...")
    robot_R = get_yam_robot(channel=args.can_right, gripper_type=gripper_type_R, zero_gravity_mode=True)
    print("[OK] 右臂初始化完成")

    print(f"\n初始化左臂 (CAN: {args.can_left}, 夹爪: {args.gripper_left})...")
    robot_L = get_yam_robot(channel=args.can_left,  gripper_type=gripper_type_L, zero_gravity_mode=True)
    print("[OK] 左臂初始化完成")

    # ---- 初始化运动学 (arm-only XML，左右臂共用同一模型) ----
    xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
    kin_R = Kinematics(xml_path, site_name=args.site)
    kin_L = Kinematics(xml_path, site_name=args.site)

    # ---- 移动到安全位置 ----
    safe_q_R = np.append(SAFE_JOINT_POS, 1.0) if has_gripper_R else SAFE_JOINT_POS.copy()
    safe_q_L = np.append(SAFE_JOINT_POS, 1.0) if has_gripper_L else SAFE_JOINT_POS.copy()

    print("\n移动双臂到安全位置...")
    robot_R.move_joints(safe_q_R, time_interval_s=SAFE_MOVE_TIME)
    robot_L.move_joints(safe_q_L, time_interval_s=SAFE_MOVE_TIME)
    print("[OK] 已到达安全位置\n")

    # ---- 连接共享内存 ----
    print("连接共享内存 (请确保 xr_server_2arm.py 已启动)...")
    reader = XRSharedMemReader()

    # ---- 创建控制器 ----
    ctrl_R = ArmController(robot_R, kin_R, has_gripper_R, name="右臂")
    ctrl_L = ArmController(robot_L, kin_L, has_gripper_L, name="左臂")

    print("\n" + "-" * 60)
    print("准备就绪! 双臂遥操作")
    print("  右手 Grip 按住: 控制右臂")
    print("  左手 Grip 按住: 控制左臂")
    print("  可同时操作，也可单独操作")
    print("  连按两次 Grip:  对应臂回到安全位置")
    print("-" * 60 + "\n")

    loop_count = 0

    try:
        while True:
            data = reader.read_xr_data()

            # ---- 数据超时 ----
            if not data['is_fresh']:
                for ctrl in (ctrl_R, ctrl_L):
                    if ctrl.is_resetting:
                        if not ctrl.move_to_zero():
                            ctrl.is_resetting = False
                            print(f"\n[{ctrl.name}] 已到达安全位置!")
                    else:
                        ctrl.hold_position()
                if loop_count % 500 == 0:
                    print(f"\r[等待] XR 数据超时...    ", end="", flush=True)
                loop_count += 1
                time.sleep(0.001)
                continue

            # ---- 紧急停止 ----
            if data['emergency_stop']:
                for ctrl in (ctrl_R, ctrl_L):
                    ctrl.is_resetting = False
                    ctrl.hold_position()
                if loop_count % 500 == 0:
                    print(f"\r[急停] 双臂已停止! 按 Grip 恢复    ", end="", flush=True)
                loop_count += 1
                time.sleep(0.001)
                continue

            # ---- 右臂控制 ----
            target_pos_R, ik_ok_R = ctrl_R.handle_grip(
                data['right_grip'], data['right_pose'], data['right_trigger']
            )

            # ---- 左臂控制 ----
            target_pos_L, ik_ok_L = ctrl_L.handle_grip(
                data['left_grip'], data['left_pose'], data['left_trigger']
            )

            # ---- 写入反馈 ----
            ee_r = target_pos_R if target_pos_R is not None else np.zeros(3)
            ee_l = target_pos_L if target_pos_L is not None else np.zeros(3)

            total_cnt  = ctrl_R.ik_total_count + ctrl_L.ik_total_count
            total_fail = ctrl_R.ik_fail_count  + ctrl_L.ik_fail_count
            combined_ik_rate = (1 - total_fail / max(total_cnt, 1)) * 100

            reader.write_robot_feedback(
                ee_pos_right=ee_r,
                ee_pos_left=ee_l,
                ik_success_rate=combined_ik_rate,
                connected=True,
                ik_ok=ik_ok_R and ik_ok_L,
                tracking=(data['right_grip'] > GRIP_THRESHOLD or
                          data['left_grip']  > GRIP_THRESHOLD),
            )

            # ---- 打印状态 (20 Hz) ----
            if loop_count % 50 == 0:
                r_active = data['right_grip'] > GRIP_THRESHOLD
                l_active = data['left_grip']  > GRIP_THRESHOLD

                r_status = "回零" if ctrl_R.is_resetting else ("控制" if r_active and ctrl_R.is_calibrated else "待机")
                l_status = "回零" if ctrl_L.is_resetting else ("控制" if l_active and ctrl_L.is_calibrated else "待机")

                parts = [f"右:{r_status}"]
                if r_status == "控制":
                    obs = robot_R.get_observations()
                    T   = kin_R.fk(obs["joint_pos"])
                    p   = T[:3, 3]
                    parts.append(f"[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] IK:{ctrl_R.ik_success_rate:.0f}%")

                parts.append(f" 左:{l_status}")
                if l_status == "控制":
                    obs = robot_L.get_observations()
                    T   = kin_L.fk(obs["joint_pos"])
                    p   = T[:3, 3]
                    parts.append(f"[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] IK:{ctrl_L.ik_success_rate:.0f}%")

                print(f"\r{''.join(parts)}    ", end="", flush=True)

            loop_count += 1
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n双臂控制端退出")
        if ctrl_R.ik_total_count > 0:
            print(f"  右臂 IK: {ctrl_R.ik_total_count} 次, 成功率 {ctrl_R.ik_success_rate:.1f}%")
        if ctrl_L.ik_total_count > 0:
            print(f"  左臂 IK: {ctrl_L.ik_total_count} 次, 成功率 {ctrl_L.ik_success_rate:.1f}%")
    finally:
        reader.close()
        print("返回安全位置...")
        robot_R.move_joints(safe_q_R, time_interval_s=SAFE_MOVE_TIME)
        robot_L.move_joints(safe_q_L, time_interval_s=SAFE_MOVE_TIME)
        robot_R.close()
        robot_L.close()
        print("完成")


if __name__ == "__main__":
    main()
