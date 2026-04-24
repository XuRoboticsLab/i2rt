#!/usr/bin/env python3
"""
Single-arm leader-follower teleoperation with bilateral force feedback.
Single-process version: both arms connected to the same machine.

The leader arm runs in low/zero stiffness so the operator can drag it freely.
The follower arm tracks the leader's joint positions.
Contact torques on the follower are reflected back to the leader so the
operator physically feels what the follower encounters.

Force feedback law:
  τ_diff    = τ_follower_measured - τ_leader_measured
  ff_leader = -force_scale * τ_diff
  When both arms are in free space their torques are similar and τ_diff ≈ 0.
  When follower hits an obstacle τ_follower rises, so τ_diff captures the contact.

Gripper control:
  The gripper is toggled open/closed by a foot pedal via ROS2.
  Each message on --pedal_topic flips the gripper state.
  rosbridge_server must be running (ros2 launch rosbridge_server rosbridge_websocket_launch.xml).

Usage:
  python lead_follower.py --leader_can can_leader --follower_can can_follower
  python lead_follower.py --leader_can can_leader --follower_can can_follower \\
      --force_scale 0.2 --leader_kp 0.0 --leader_kd 0.5
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import roslibpy
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


@dataclass
class Args:
    leader_can: str = "can_left"
    """CAN interface for the leader robot arm."""
    follower_can: str = "can_right"
    """CAN interface for the follower robot arm."""
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "no_gripper"] = "linear_4310"
    """Gripper type — same for both arms."""
    force_scale: float = 0.15
    """
    Torque reflection gain applied to (τ_follower - τ_leader).
    When follower hits an obstacle its torque rises relative to the leader's,
    so the difference isolates contact force regardless of gravity.
    0.0 = no force feedback. Flip sign if feedback direction feels inverted.
    """
    leader_kp: float = 0.0
    """
    Leader arm position stiffness. 0.0 = fully compliant (gravity comp only,
    operator drags freely). Increase slightly if the arm feels too loose.
    """
    leader_kd: float = 0.5
    """Leader arm damping — prevents oscillation in free-float mode."""
    ee_mass: Optional[float] = None
    """Override end-effector mass (kg) for gravity compensation on both arms."""
    rosbridge_host: str = "localhost"
    """Hostname of the rosbridge_server WebSocket."""
    rosbridge_port: int = 9090
    """Port of the rosbridge_server WebSocket."""
    pedal_topic: str = "/foot_pedal/press"
    """ROS2 topic published by the foot pedal. Any incoming message toggles the gripper."""
    gripper_open: bool = True
    """Initial gripper state: True = open, False = closed."""


def main(args: Args) -> None:
    gripper_type = GripperType.from_string_name(args.gripper)
    has_gripper = (gripper_type != GripperType.NO_GRIPPER)

    # --- Foot-pedal gripper toggle via rosbridge ---
    gripper_open = args.gripper_open
    gripper_lock = threading.Lock()

    ros_client: Optional[roslibpy.Ros] = None
    if has_gripper:
        ros_client = roslibpy.Ros(host=args.rosbridge_host, port=args.rosbridge_port)

        def _on_pedal(msg):
            nonlocal gripper_open
            with gripper_lock:
                gripper_open = not gripper_open
                state = "open" if gripper_open else "closed"
            print(f"\n[pedal] gripper → {state}")

        def _on_ready():
            print(f"[rosbridge] connected to {args.rosbridge_host}:{args.rosbridge_port}")
            pedal_sub = roslibpy.Topic(ros_client, args.pedal_topic, "std_msgs/Empty")
            pedal_sub.subscribe(_on_pedal)
            print(f"[rosbridge] subscribed to {args.pedal_topic}")

        def _on_error(error):
            print(f"[rosbridge] error: {error}")

        def _on_close(error):
            print(f"[rosbridge] connection closed: {error}")

        ros_client.on_ready(_on_ready)
        ros_client.on("error", _on_error)
        ros_client.on("close", _on_close)
        ros_client.run()  # starts websocket loop in a background thread

    # --- Robot init ---
    print("Initializing leader...")
    leader = get_yam_robot(channel=args.leader_can, gripper_type=gripper_type, ee_mass=args.ee_mass)
    print("Initializing follower...")
    follower = get_yam_robot(channel=args.follower_can, gripper_type=gripper_type, ee_mass=args.ee_mass)

    n_dofs = leader.num_dofs()
    n_arm = n_dofs - 1 if has_gripper else n_dofs

    # Put leader in low/zero stiffness — operator drags it freely.
    # Gravity compensation remains active internally.
    leader.update_kp_kd(
        kp=np.full(n_dofs, args.leader_kp),
        kd=np.full(n_dofs, args.leader_kd),
    )

    # Sync: move follower to leader's current position before starting.
    q_leader = leader.get_joint_pos()
    q_follower = follower.get_joint_pos()
    print(f"Leader  joints: {np.round(q_leader, 3)}")
    print(f"Follower joints: {np.round(q_follower, 3)}")
    print("Moving follower to leader position...")

    steps = 100
    for i in range(1, steps + 1):
        follower.command_joint_pos(q_follower + (q_leader - q_follower) * (i / steps))
        time.sleep(0.03)

    print(f"Ready. force_scale={args.force_scale}  leader_kp={args.leader_kp}  leader_kd={args.leader_kd}")
    if has_gripper:
        print(f"Gripper initial state: {'open' if gripper_open else 'closed'}. Press foot pedal to toggle.")
    print("Move the leader arm — follower will track with force feedback. Ctrl+C to stop.")

    try:
        while True:
            # get_joint_pos() returns all DOFs (arm + gripper) — used for position commands
            q_leader_full = leader.get_joint_pos()   # (n_dofs,)

            # get_observations()["joint_eff"] returns arm joints only — used for torque diff
            τ_leader   = leader.get_observations()["joint_eff"]    # (n_arm,)
            τ_follower = follower.get_observations()["joint_eff"]  # (n_arm,)

            # Torque difference: near zero in free space, nonzero on contact.
            # Gravity cancels naturally since both arms track the same configuration.
            τ_diff = τ_follower - τ_leader           # (n_arm,)

            # Build separate commands for leader and follower.
            q_leader_cmd = q_leader_full.copy()
            q_follower_cmd = q_leader_full.copy()
            if has_gripper:
                # Follower gripper follows foot-pedal toggle.
                with gripper_lock:
                    q_follower_cmd[-1] = 1.0 if gripper_open else 0.0
                # Leader gripper is always closed, independent of follower.
                q_leader_cmd[-1] = 0.0

            follower.command_joint_pos(q_follower_cmd)

            ff = np.zeros(n_dofs)
            ff[:n_arm] = -args.force_scale * τ_diff
            leader.command_joint_pos(q_leader_cmd, feedforward_torques=ff)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        leader.close()
        follower.close()
        if ros_client is not None:
            ros_client.terminate()


if __name__ == "__main__":
    main(tyro.cli(Args))
