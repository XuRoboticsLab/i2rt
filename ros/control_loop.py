import time
import threading
import numpy as np

from config import CONTROL_RATE, SAFE_JOINT_POS, SAFE_MOVE_TIME
from shared_state import SharedState


def _read_arm_state(robot, kin, state: SharedState):
    """读取 FK + 关节状态 + 夹爪并写入共享状态。"""
    obs = robot.get_observations()
    arm_pos = obs["joint_pos"]        # (n_arm,) rad
    arm_vel = obs["joint_vel"]        # (n_arm,) rad/s
    grip_pos = float(obs.get("gripper_pos", np.array([1.0]))[0])
    grip_vel = float(obs.get("gripper_vel", np.array([0.0]))[0])

    T = kin.fk(arm_pos)               # (4, 4)
    fk_pos = T[:3, 3]
    fk_rot = T[:3, :3]

    state.set_robot_state(arm_pos, arm_vel, fk_pos, fk_rot,
                          gripper_pos=grip_pos, gripper_vel=grip_vel)


def _hold(robot, state: SharedState):
    """保持上一次有效关节位置（gravity comp 由 MotorChainRobot 内部处理）。"""
    with state._lock:
        grip = state.target_gripper_pos
    full_q = np.append(state.last_valid_joint_pos, grip)
    robot.command_joint_pos(full_q)


def control_loop(robot, kin, state: SharedState, stop_event: threading.Event,
                 has_gripper: bool = True):
    """
    50 Hz 控制循环。

    逻辑与 Panthera 版本完全一致：
      1. 复位（最高优先级）
      2. 消费 Twist 增量，更新目标位姿
      3. Watchdog 超时 → hold
      4. IK → command_joint_pos
      5. 更新共享状态供发布线程读取
    """
    interval = 1.0 / CONTROL_RATE
    n_arm = len(SAFE_JOINT_POS)
    site_name = kin._site_name
    print(f"[Control] 控制循环启动，频率 {CONTROL_RATE:.0f} Hz")

    while not stop_event.is_set():
        t0 = time.time()

        # ── 1. 复位 ───────────────────────────────
        with state._lock:
            do_reset = state.reset_requested
            state.reset_requested = False

        if do_reset:
            print("[Control] 复位中...")
            safe_q = np.array(SAFE_JOINT_POS, dtype=float)
            if has_gripper:
                safe_q = np.append(safe_q, state.target_gripper_pos)
            robot.move_joints(safe_q, time_interval_s=SAFE_MOVE_TIME)

            obs = robot.get_observations()
            arm_pos = obs["joint_pos"]
            T = kin.fk(arm_pos)
            state.reset_target_to(T[:3, 3], T[:3, :3])
            state.last_valid_joint_pos = arm_pos.copy()
            _read_arm_state(robot, kin, state)
            print("[Control] 复位完成")
            time.sleep(max(0.0, interval - (time.time() - t0)))
            continue

        # ── 2. 消费 Twist 增量 ────────────────────
        twist = state.pop_twist()
        if twist is not None:
            state.apply_twist_to_target(twist[0], twist[1])

        # ── 3. Watchdog ───────────────────────────
        if not state.is_watchdog_ok():
            _hold(robot, state)
            _read_arm_state(robot, kin, state)
            time.sleep(max(0.0, interval - (time.time() - t0)))
            continue

        # ── 4. IK + 发送指令 ──────────────────────
        target_pos, target_rot = state.get_target()
        target_T = np.eye(4)
        target_T[:3, :3] = target_rot
        target_T[:3, 3]  = target_pos

        success, q_sol = kin.ik(
            target_T,
            site_name,
            init_q=state.last_valid_joint_pos,
        )

        if success:
            with state._lock:
                grip = state.target_gripper_pos
            full_q = np.append(q_sol[:n_arm], grip) if has_gripper else q_sol[:n_arm]
            robot.command_joint_pos(full_q)
            state.last_valid_joint_pos = q_sol[:n_arm].copy()
        else:
            _hold(robot, state)

        # ── 5. 更新发布状态 ───────────────────────
        _read_arm_state(robot, kin, state)

        time.sleep(max(0.0, interval - (time.time() - t0)))
