# Kinematics API Reference

`Kinematics` 提供基于 MuJoCo + [mink](https://github.com/kevinzakka/mink) 的正运动学（FK）和微分逆运动学（IK），与 `MotorChainRobot` 解耦，需单独实例化使用。

**文件路径:** `i2rt/robots/kinematics.py`

---

## 依赖

```bash
uv pip install -e .   # mujoco 和 mink 已包含在项目依赖中
```

---

## 构造函数

```python
Kinematics(xml_path: str, site_name: Optional[str])
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `xml_path` | `str` | MuJoCo XML 模型文件路径。推荐通过 `combine_arm_and_gripper_xml()` 生成组合模型路径 |
| `site_name` | `str \| None` | 默认末端执行器 site 名称，调用 `fk()`/`ik()` 时若不传 `site_name` 则使用此值 |

### 获取 xml_path

```python
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
```

| `ArmType` | 说明 |
|-----------|------|
| `ArmType.YAM` | YAM 机械臂 |

| `GripperType` | 说明 |
|---------------|------|
| `GripperType.NO_GRIPPER` | 无夹爪（纯机械臂） |
| `GripperType.LINEAR_4310` | 直线夹爪（DM4310 电机） |
| `GripperType.LINEAR_3507` | 直线夹爪（DM3507 电机） |
| `GripperType.CRANK_4310` | 曲柄夹爪（DM4310 电机） |
| `GripperType.YAM_TEACHING_HANDLE` | 示教手柄（Leader 臂） |

### 常用 site 名称

| site 名称 | 说明 |
|-----------|------|
| `"grasp_site"` | 抓取点（绝大多数夹爪配置均有此 site） |
| `"tcp_site"` | 工具中心点（示教手柄使用，无 `grasp_site`） |

---

## 方法

### `fk(q, site_name=None) -> np.ndarray`

计算正运动学，返回指定 site 在世界坐标系下的位姿。

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | `np.ndarray` | 关节角度，形状 `(n_joints,)`，单位 rad |
| `site_name` | `str \| None` | 目标 site 名称；省略时使用构造时传入的默认值 |
| **返回** | `np.ndarray (4, 4)` | 末端位姿的齐次变换矩阵（SE3），`[:3, :3]` 为旋转矩阵，`[:3, 3]` 为平移向量（m） |

```python
q = np.zeros(6)
T = kin.fk(q)          # shape: (4, 4)
R = T[:3, :3]          # 旋转矩阵
t = T[:3, 3]           # 位置 (x, y, z) in meters
```

零位（`q = zeros(6)`）的参考位姿（YAM + NO_GRIPPER，`grasp_site`）：

```
R = [[0, 0, 1],
     [0, 1, 0],
     [-1, 0, 0]]
t = [0.113, 0.004, 0.172]  (m)
```

---

### `ik(target_pose, site_name, init_q=None, ...) -> Tuple[bool, np.ndarray]`

微分逆运动学求解，基于 mink 的 QP 优化迭代收敛。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_pose` | `np.ndarray (4, 4)` | — | 目标末端位姿，齐次变换矩阵 |
| `site_name` | `str` | — | 目标 site 名称 |
| `init_q` | `np.ndarray \| None` | `None` | 迭代初始关节构型；省略时使用上次 `fk()`/`ik()` 更新后的内部状态 |
| `limits` | `List[mink.Limit] \| None` | `None` | 额外约束列表（如关节限位、速度限位） |
| `dt` | `float` | `0.01` | 积分步长（s） |
| `solver` | `str` | `"quadprog"` | QP 求解器名称 |
| `pos_threshold` | `float` | `1e-4` | 位置收敛阈值（m） |
| `ori_threshold` | `float` | `1e-4` | 姿态收敛阈值（rad） |
| `damping` | `float` | `1e-4` | Levenberg-Marquardt 阻尼系数 |
| `max_iters` | `int` | `200` | 最大迭代次数 |
| `verbose` | `bool` | `False` | 打印迭代信息 |

| 返回 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | `True` 表示在 `max_iters` 内收敛到阈值内 |
| `q` | `np.ndarray (n_joints,)` | 收敛（或最终）的关节角度（rad） |

> **注意：** 即使 `success=False`，也会返回迭代结束时最接近目标的关节构型，可视情况决定是否使用。

```python
target_T = kin.fk(np.zeros(6))
target_T[0, 3] -= 0.1   # x 方向移动 -10 cm

success, q_sol = kin.ik(target_T, "grasp_site", init_q=np.zeros(6))
if success:
    robot.command_joint_pos(q_sol)
else:
    print("IK 未收敛，请检查目标位姿是否可达")
```

---

## 完整示例

```python
import numpy as np
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

# 1. 构建模型路径（arm + gripper 的组合 XML）
xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
kin = Kinematics(xml_path, site_name="grasp_site")

# 2. 正运动学
q_current = np.zeros(6)
T_current = kin.fk(q_current)   # (4, 4)
print("当前末端位置:", T_current[:3, 3])

# 3. 构造目标位姿（在当前基础上平移）
T_target = T_current.copy()
T_target[0, 3] -= 0.1   # x 方向后退 10 cm
T_target[2, 3] += 0.1   # z 方向上升 10 cm

# 4. 逆运动学
success, q_sol = kin.ik(T_target, "grasp_site", init_q=q_current, verbose=True)

# 5. 验证精度（FK 往返检验）
if success:
    T_check = kin.fk(q_sol)
    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    print(f"位置误差: {pos_err*1000:.3f} mm")
```

---

## 与 MotorChainRobot 结合使用

`Kinematics` 与 `MotorChainRobot` 完全解耦，典型的笛卡尔空间控制流程：

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
import numpy as np

robot = get_yam_robot(channel="can0", gripper_type="linear_4310")
xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
kin = Kinematics(xml_path, site_name="grasp_site")

# 读取当前关节角
q_now = robot.get_joint_pos()[:6]   # 去掉夹爪

# 正解得到当前末端位姿
T_now = kin.fk(q_now)

# 修改目标位姿
T_target = T_now.copy()
T_target[0, 3] += 0.05  # 向前移动 5 cm

# 逆解
success, q_target = kin.ik(T_target, "grasp_site", init_q=q_now)
if success:
    # 追加夹爪通道（保持当前夹爪状态）
    gripper_pos = robot.get_joint_pos()[6:]
    robot.move_joints(np.concatenate([q_target, gripper_pos]), time_interval_s=2.0)
```

---

## 注意事项

- `Kinematics` 内部维护一个 `mink.Configuration` 状态，**非线程安全**，多线程场景下需自行加锁或为每个线程创建独立实例。
- IK 求解结果受 `init_q` 影响较大，建议始终传入当前关节角作为初始值以提高收敛率和结果合理性。
- `combine_arm_and_gripper_xml` 使用 `NO_GRIPPER` 时，模型只包含 6 个关节，IK 求解的 `q` 形状为 `(6,)`；运动学模型中不含夹爪，夹爪需单独管理。
