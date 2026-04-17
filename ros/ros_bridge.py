import time
import threading
import numpy as np
import roslibpy
from scipy.spatial.transform import Rotation

from config import (
    PUBLISH_RATE,
    JOINT_NAMES,
    TOPIC_CMD, TOPIC_GRIPPER, TOPIC_RESET,
    TOPIC_JOINTS, TOPIC_EE,
)
from shared_state import SharedState


def _rot_to_quat(R) -> list:
    """3x3 旋转矩阵 → [x, y, z, w]"""
    return Rotation.from_matrix(R).as_quat().tolist()


# ── 订阅回调 ────────────────────────────────
def make_cmd_callback(state: SharedState):
    def callback(msg):
        l = msg["linear"]
        a = msg["angular"]
        state.push_twist([l["x"], l["y"], l["z"]],
                         [a["x"], a["y"], a["z"]])
    return callback


def make_gripper_callback(state: SharedState):
    """接收 std_msgs/Int8：1=open，-1=close，0=idle"""
    def callback(msg):
        cmd = int(msg["data"])
        with state._lock:
            if cmd == 1:
                state.target_gripper_pos = 1.0
            elif cmd == -1:
                state.target_gripper_pos = 0.0
    return callback


def make_reset_callback(state: SharedState):
    def callback(msg):
        if msg["data"]:
            with state._lock:
                state.reset_requested = True
    return callback


# ── 订阅管理 ────────────────────────────────
class RosSubscribers:
    def __init__(self, ros: roslibpy.Ros, state: SharedState):
        self._subs = [
            roslibpy.Topic(ros, TOPIC_CMD,     "geometry_msgs/Twist"),
            roslibpy.Topic(ros, TOPIC_GRIPPER, "std_msgs/Int8"),
            roslibpy.Topic(ros, TOPIC_RESET,   "std_msgs/Bool"),
        ]
        self._subs[0].subscribe(make_cmd_callback(state))
        self._subs[1].subscribe(make_gripper_callback(state))
        self._subs[2].subscribe(make_reset_callback(state))
        print(f"[ROS] 已订阅 {TOPIC_CMD}, {TOPIC_GRIPPER}, {TOPIC_RESET}")

    def unsubscribe_all(self):
        for sub in self._subs:
            sub.unsubscribe()


# ── State 发布线程 ──────────────────────────
def publisher_thread(ros: roslibpy.Ros, state: SharedState,
                     stop_event: threading.Event):
    joint_pub = roslibpy.Topic(ros, TOPIC_JOINTS, "sensor_msgs/JointState")
    ee_pub    = roslibpy.Topic(ros, TOPIC_EE,     "geometry_msgs/PoseStamped")
    interval  = 1.0 / PUBLISH_RATE

    while not stop_event.is_set():
        t0 = time.time()

        if ros.is_connected:
            joint_pos, joint_vel, grip_pos, grip_vel, fk_pos, fk_rot, stamp = \
                state.get_robot_state()

            all_names = JOINT_NAMES + ["gripper"]
            all_pos   = joint_pos + [grip_pos]
            all_vel   = joint_vel + [grip_vel]

            joint_pub.publish(roslibpy.Message({
                "header":   {"stamp": stamp, "frame_id": "base_link"},
                "name":     all_names,
                "position": all_pos,
                "velocity": all_vel,
                "effort":   [0.0] * len(all_names),
            }))

            q = _rot_to_quat(fk_rot)
            ee_pub.publish(roslibpy.Message({
                "header": {"stamp": stamp, "frame_id": "base_link"},
                "pose": {
                    "position":    {"x": fk_pos[0], "y": fk_pos[1], "z": fk_pos[2]},
                    "orientation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
                },
            }))

        time.sleep(max(0.0, interval - (time.time() - t0)))
