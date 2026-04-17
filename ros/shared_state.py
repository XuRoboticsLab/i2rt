import time
import threading
import numpy as np

from config import TRANSLATION_SCALE, ROTATION_SCALE, WATCHDOG_TIMEOUT


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

        # 目标末端位姿（由 Twist 累积更新）
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3, dtype=float)

        # 待处理的 Twist 增量（累加，不丢帧）
        self._pending_twist = None          # (linear_xyz, angular_xyz)

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

    # ── Twist ──────────────────────────────────
    def push_twist(self, linear_xyz, angular_xyz):
        with self._lock:
            if self._pending_twist is None:
                self._pending_twist = (np.array(linear_xyz), np.array(angular_xyz))
            else:
                self._pending_twist = (
                    self._pending_twist[0] + np.array(linear_xyz),
                    self._pending_twist[1] + np.array(angular_xyz),
                )
            self.last_cmd_time = time.time()

    def pop_twist(self):
        with self._lock:
            twist = self._pending_twist
            self._pending_twist = None
        return twist

    def apply_twist_to_target(self, linear_xyz, angular_xyz):
        lx, ly, lz = np.array(linear_xyz) * TRANSLATION_SCALE
        rx, ry, rz = np.array(angular_xyz) * ROTATION_SCALE
        with self._lock:
            self.target_position += np.array([lx, ly, lz])
            if rx: self.target_rotation = self.target_rotation @ _rx(rx)
            if ry: self.target_rotation = self.target_rotation @ _ry(ry)
            if rz: self.target_rotation = self.target_rotation @ _rz(rz)

    def get_target(self):
        with self._lock:
            return self.target_position.copy(), self.target_rotation.copy()

    def reset_target_to(self, position, rotation):
        with self._lock:
            self.target_position = np.array(position)
            self.target_rotation = np.array(rotation, dtype=float)

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
