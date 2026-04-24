import time
import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310)
xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
kin = Kinematics(xml_path, site_name="grasp_site")

MOVE_TIME = 2.0   # 每步运动时间（秒）
STEPS = [
    ("z", np.array([0.0, 0.0, 0.1])),
    ("x", np.array([0.1, 0.0, 0.0])),
    ("y", np.array([0.0, 0.1, 0.0])),
]

q = robot.get_joint_pos()[:6]

for axis, delta in STEPS:
    T = kin.fk(q)
    T_target = T.copy()
    T_target[:3, 3] += delta

    success, q_sol = kin.ik(T_target, "grasp_site", init_q=q)
    if not success:
        print(f"[{axis}] IK 未收敛，跳过")
        continue

    print(f"[{axis}] 目标末端位置: {T_target[:3, 3].round(4)}")
    full_q = np.append(q_sol[:6], robot.get_joint_pos()[6])  # 保持当前夹爪
    robot.move_joints(full_q, time_interval_s=MOVE_TIME)
    time.sleep(0.5)
    q = robot.get_joint_pos()[:6]
    print(f"[{axis}] 实际末端位置: {kin.fk(q)[:3, 3].round(4)}")

robot.close()
