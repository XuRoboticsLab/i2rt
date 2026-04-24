import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation

from config import TRANSLATION_SCALE, ROTATION_SCALE, WATCHDOG_TIMEOUT, CONTROL_RATE, TRACKING_GAIN_HZ, DAMPING_RATIO, MAX_LINEAR_VEL, MAX_ANGULAR_VEL


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class SharedState:
    def __init__(self, n_arm_joints: int = 6):
        self._lock = threading.Lock()
        self._n = n_arm_joints

        # 每控制周期的速度上限（饱和裁剪用）
        self._max_linear_step  = MAX_LINEAR_VEL  / CONTROL_RATE
        self._max_angular_step = MAX_ANGULAR_VEL / CONTROL_RATE
        # PD 增益（per-cycle）
        self._kp = TRACKING_GAIN_HZ / CONTROL_RATE
        self._kd = DAMPING_RATIO * self._kp   # D项：error变化率加权，误差收缩时产生制动力

        # 上一周期误差，用于计算 D 项
        self._prev_linear_error  = np.zeros(3)
        self._prev_angular_error = np.zeros(3)  # rotvec 表示

        # 目标末端位姿（hard target，由 Twist 偏移量直接设置）
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3, dtype=float)

        # smooth target：每控制周期向 hard target 限速步进，IK 使用此值
        self._smooth_position = self.target_position.copy()
        self._smooth_rotation = self.target_rotation.copy()

        # 校准基准：Grip 单击时记录的机械臂末端位姿
        self.calibration_position: np.ndarray | None = None
        self.calibration_rotation: np.ndarray | None = None

        # 待处理的 Twist 偏移量（last-wins，取最新值）
        self._pending_twist = None

        # 夹爪目标（归一化 0=closed 1=open）
        self.target_gripper_pos: float = 1.0
        # 复位
        self.reset_requested = False

        # Watchdog
        self.last_cmd_time = 0.0

        # 机器人当前状态（供发布线程读取）
        self.joint_positions  = [0.0] * n_arm_joints
        self.joint_velocities = [0.0] * n_arm_joints
        self.gripper_position = 1.0
        self.gripper_velocity = 0.0
        self.fk_position      = [0.0, 0.0, 0.0]
        self.fk_rotation      = np.eye(3)
        self.stamp            = {"secs": 0, "nsecs": 0}

        # 上一次有效 IK 结果（arm joints only）
        self.last_valid_joint_pos = np.zeros(n_arm_joints)

    # ── Calibration ────────────────────────────
    def set_calibration_pose(self, pos, rot):
        """Grip 单击时由 ROS 回调调用，记录当前末端位姿为位置控制基准。"""
        with self._lock:
            self.calibration_position = np.array(pos, dtype=float)
            self.calibration_rotation = np.array(rot, dtype=float)
            # 同步 hard/smooth target，避免校准后出现初始跳变
            self.target_position    = self.calibration_position.copy()
            self.target_rotation    = self.calibration_rotation.copy()
            self._smooth_position   = self.calibration_position.copy()
            self._smooth_rotation   = self.calibration_rotation.copy()
            self._prev_linear_error  = np.zeros(3)
            self._prev_angular_error = np.zeros(3)
            self._pending_twist      = None
        print("[State] 校准基准已更新")

    # ── Twist ──────────────────────────────────
    def push_twist(self, linear_xyz, angular_xyz):
        with self._lock:
            # 位置控制：Pico 发送的是绝对偏移量，取最新值即可
            self._pending_twist = (np.array(linear_xyz), np.array(angular_xyz))
            self.last_cmd_time = time.time()

    def pop_twist(self):
        with self._lock:
            twist = self._pending_twist
            self._pending_twist = None
        return twist

    def set_target_from_offset(self, linear_xyz, angular_xyz):
        """位置控制：target = 校准基准 + Pico 偏移量（直接设置，不累加）。"""
        if self.calibration_position is None:
            return
        pos_offset = np.array(linear_xyz) * TRANSLATION_SCALE
        rot_offset = Rotation.from_rotvec(np.array(angular_xyz) * ROTATION_SCALE).as_matrix()
        with self._lock:
            self.target_position = self.calibration_position + pos_offset
            self.target_rotation = rot_offset @ self.calibration_rotation

    def step_smooth_target(self):
        """将 smooth target 向 hard target 步进一个控制周期，返回 (pos, rot) 供 IK 使用。

        线性部分：若误差超过最大步长则按方向裁剪；
        旋转部分：用 SLERP 在误差角度方向上步进最多 max_angular_step。
        """
        with self._lock:
            # ── 线性 PD ───────────────────────────────
            err  = self.target_position - self._smooth_position
            step = self._kp * err + self._kd * (err - self._prev_linear_error)
            mag  = np.linalg.norm(step)
            if mag > self._max_linear_step:
                step = step / mag * self._max_linear_step
            self._prev_linear_error = err.copy()
            self._smooth_position  += step

            # ── 旋转 PD（误差用 rotvec 表示）────────────
            r_curr  = Rotation.from_matrix(self._smooth_rotation)
            r_tgt   = Rotation.from_matrix(self.target_rotation)
            err_rv  = (r_tgt * r_curr.inv()).as_rotvec()
            omega   = self._kp * err_rv + self._kd * (err_rv - self._prev_angular_error)
            ang_mag = np.linalg.norm(omega)
            if ang_mag > self._max_angular_step:
                omega = omega / ang_mag * self._max_angular_step
            self._prev_angular_error = err_rv.copy()
            self._smooth_rotation    = (Rotation.from_rotvec(omega) * r_curr).as_matrix()

            return self._smooth_position.copy(), self._smooth_rotation.copy()

    def get_target(self):
        with self._lock:
            return self.target_position.copy(), self.target_rotation.copy()

    def reset_target_to(self, position, rotation):
        """复位时同步重置 smooth target 和 PD 历史，避免复位后产生跳变。"""
        with self._lock:
            self.target_position     = np.array(position)
            self.target_rotation     = np.array(rotation, dtype=float)
            self._smooth_position    = self.target_position.copy()
            self._smooth_rotation    = self.target_rotation.copy()
            self._prev_linear_error  = np.zeros(3)
            self._prev_angular_error = np.zeros(3)

    # ── Watchdog ───────────────────────────────
    def is_watchdog_ok(self):
        with self._lock:
            if self.last_cmd_time == 0.0:
                return False
            return (time.time() - self.last_cmd_time) < WATCHDOG_TIMEOUT

    # ── Robot state ────────────────────────────
    def set_robot_state(self, joint_pos, joint_vel, fk_pos, fk_rot,
                        gripper_pos: float = 1.0, gripper_vel: float = 0.0):
        t_ns = time.time_ns()
        with self._lock:
            self.joint_positions  = list(joint_pos)
            self.joint_velocities = list(joint_vel)
            self.gripper_position = gripper_pos
            self.gripper_velocity = gripper_vel
            self.fk_position      = list(fk_pos)
            self.fk_rotation      = fk_rot.copy()
            self.stamp            = {
                "secs":  t_ns // 1_000_000_000,
                "nsecs": t_ns  % 1_000_000_000,
            }

    def get_robot_state(self):
        with self._lock:
            return (
                list(self.joint_positions),
                list(self.joint_velocities),
                self.gripper_position,
                self.gripper_velocity,
                list(self.fk_position),
                self.fk_rotation.copy(),
                self.stamp.copy(),
            )
