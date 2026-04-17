# MotorChainRobot API Reference

`MotorChainRobot` 是 i2rt 的核心机器人控制类，封装了 CAN 总线电机链，提供重力补偿、关节位置控制、夹爪管理等功能。

**文件路径:** `i2rt/robots/motor_chain_robot.py`

---

## 构造函数

```python
MotorChainRobot(
    motor_chain: MotorChain,
    xml_path: Optional[str] = None,
    use_gravity_comp: bool = True,
    gravity: Optional[np.ndarray] = None,
    gravity_comp_factor: Optional[np.ndarray] = None,
    gripper_index: Optional[int] = None,
    kp: Union[float, List[float]] = 10.0,
    kd: Union[float, List[float]] = 1.0,
    joint_limits: Optional[np.ndarray] = None,
    gripper_limits: Optional[np.ndarray] = None,
    limit_gripper_force: float = -1,
    clip_motor_torque: float = np.inf,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    arm_type: ArmType = ArmType.YAM,
    temp_record_flag: bool = False,
    enable_gripper_calibration: bool = False,
    zero_gravity_mode: bool = True,
    ...
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `motor_chain` | `MotorChain` | CAN 总线电机链实例 |
| `xml_path` | `str \| None` | MuJoCo XML 模型路径，用于重力补偿；为 `None` 时必须将 `use_gravity_comp` 设为 `False` |
| `use_gravity_comp` | `bool` | 是否启用重力补偿（默认 `True`） |
| `gravity` | `np.ndarray \| None` | 自定义重力向量（默认使用 MuJoCo 模型中的值） |
| `gravity_comp_factor` | `np.ndarray \| None` | 各关节重力补偿缩放系数，形状 `(n_joints,)`，默认全为 1 |
| `gripper_index` | `int \| None` | 夹爪对应的关节索引（必须是最后一个关节），无夹爪时为 `None` |
| `kp` | `float \| List[float]` | 位置环增益，可为标量或每个关节独立指定（默认 `10.0`） |
| `kd` | `float \| List[float]` | 速度环增益，可为标量或每个关节独立指定（默认 `1.0`） |
| `joint_limits` | `np.ndarray \| None` | 关节角度限位，形状 `(n_arm_joints, 2)`，`[:, 0]` 为下限，`[:, 1]` 为上限（rad）；设置后覆盖 XML 中的限位 |
| `gripper_limits` | `np.ndarray \| None` | 夹爪电机位置限位 `[closed, open]`（rad）；启用夹爪时必须提供，或开启自动标定 |
| `limit_gripper_force` | `float` | 夹爪力矩上限（N）；`-1` 表示不限制 |
| `clip_motor_torque` | `float` | 重力补偿叠加力矩的裁剪上限（Nm）；默认不裁剪 |
| `gripper_type` | `GripperType` | 夹爪类型，影响力矩换算 |
| `arm_type` | `ArmType` | 机械臂类型 |
| `temp_record_flag` | `bool` | 是否在 observations 中记录电机温度 |
| `enable_gripper_calibration` | `bool` | 是否自动检测夹爪限位（上电时运行标定程序） |
| `zero_gravity_mode` | `bool` | `True`：上电后进入零力矩模式（可手动拖动）；`False`：上电后保持当前位置 |

---

## 数据类

### `JointStates`

```python
@dataclass
class JointStates:
    names: List[str]        # 关节名称列表
    pos: np.ndarray         # 关节位置 (rad)
    vel: np.ndarray         # 关节速度 (rad/s)
    eff: np.ndarray         # 关节力矩 (Nm)
    temp_mos: np.ndarray    # 电机 MOS 温度 (°C)
    temp_rotor: np.ndarray  # 电机转子温度 (°C)
    timestamp: float        # 时间戳 (s)
```

### `JointCommands`

```python
@dataclass
class JointCommands:
    torques: np.ndarray     # 额外前馈力矩 (Nm)
    pos: np.ndarray         # 目标位置 (rad)
    vel: np.ndarray         # 目标速度 (rad/s)
    kp: np.ndarray          # 位置增益
    kd: np.ndarray          # 速度增益
    indices: Optional[List[int]]  # 可选，指定控制的关节索引
```

---

## 公开方法

### 读取状态

#### `num_dofs() -> int`

返回机器人自由度总数（含夹爪）。

```python
n = robot.num_dofs()  # e.g. 7 (6 arm + 1 gripper)
```

---

#### `get_joint_pos() -> np.ndarray`

读取当前所有关节位置（含夹爪，归一化到 `[0, 1]`）。

| | 说明 |
|---|---|
| **返回** | `np.ndarray`，形状 `(n_dofs,)`，单位 rad（夹爪为归一化值） |

```python
q = robot.get_joint_pos()  # shape: (7,)
```

---

#### `get_observations() -> Dict[str, np.ndarray]`

获取完整的机器人观测值，适合机器学习 pipeline 直接使用。

| 键 | 形状 | 说明 |
|----|------|------|
| `joint_pos` | `(n_arm_dofs,)` | 机械臂关节位置 (rad) |
| `joint_vel` | `(n_arm_dofs,)` | 机械臂关节速度 (rad/s) |
| `joint_eff` | `(n_arm_dofs,)` | 机械臂关节力矩 (Nm) |
| `gripper_pos` | `(1,)` | 夹爪位置，归一化 `[0, 1]`（有夹爪时） |
| `gripper_vel` | `(1,)` | 夹爪速度（有夹爪时） |
| `gripper_eff` | `(1,)` | 夹爪力矩（有夹爪时） |
| `temp_mos` | `(n_dofs,)` | MOS 温度，仅当 `temp_record_flag=True` |
| `temp_rotor` | `(n_dofs,)` | 转子温度，仅当 `temp_record_flag=True` |

```python
obs = robot.get_observations()
arm_q = obs["joint_pos"]       # shape: (6,)
gripper = obs["gripper_pos"]   # shape: (1,), 0=closed, 1=open
```

---

#### `get_robot_info() -> Dict[str, Any]`

返回机器人配置参数。

| 键 | 类型 | 说明 |
|----|------|------|
| `kp` | `np.ndarray` | 各关节位置增益 |
| `kd` | `np.ndarray` | 各关节速度增益 |
| `joint_limits` | `np.ndarray \| None` | 关节限位，形状 `(n_arm_joints, 2)` |
| `gripper_limits` | `np.ndarray \| None` | 夹爪电机限位 `[closed, open]` |
| `gravity_comp_factor` | `np.ndarray` | 重力补偿缩放系数 |
| `limit_gripper_effort` | `float` | 夹爪力矩限制值 |
| `gripper_index` | `int \| None` | 夹爪关节索引 |

---

#### `get_motor_torques() -> Optional[np.ndarray]`

返回上一控制周期发送给电机的实际力矩（重力补偿 + 指令前馈）。

| | 说明 |
|---|---|
| **返回** | `np.ndarray`，形状 `(n_dofs,)`，单位 Nm；初始化前为 `None` |

---

### 发送指令

#### `command_joint_pos(joint_pos: np.ndarray) -> None`

命令机械臂运动到目标关节角度（含夹爪）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `joint_pos` | `np.ndarray` | 目标关节角度，形状 `(n_dofs,)`，单位 rad；夹爪为归一化值 `[0=closed, 1=open]` |

> 超出 `joint_limits` 的指令会被自动裁剪。

```python
robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
#                                  -------- arm (rad) ----------  ^ gripper open
```

---

#### `command_joint_state(joint_state: Dict[str, np.ndarray]) -> None`

同时指定目标位置和速度（以及可选 kp/kd）。

| 键 | 类型 | 说明 |
|----|------|------|
| `pos` | `np.ndarray` | 目标位置，形状 `(n_dofs,)`，单位 rad |
| `vel` | `np.ndarray` | 目标速度，形状 `(n_dofs,)`，单位 rad/s |
| `kp` | `np.ndarray`（可选） | 位置增益，省略时使用构造时的默认值 |
| `kd` | `np.ndarray`（可选） | 速度增益，省略时使用构造时的默认值 |

```python
robot.command_joint_state({
    "pos": np.zeros(7),
    "vel": np.zeros(7),
})
```

---

#### `move_joints(target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None`

从当前位置线性插值运动到目标位置（阻塞调用）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_joint_positions` | `np.ndarray` | 目标关节角度，形状 `(n_dofs,)` |
| `time_interval_s` | `float` | 运动总时长（秒），默认 `2.0`；在 50 步内匀速插值 |

```python
robot.move_joints(np.zeros(7), time_interval_s=3.0)
```

---

#### `zero_torque_mode() -> None`

将所有 kp/kd 设为零，使机器人进入零力矩模式（可自由拖动）。

> 注意：此操作会永久修改当前 `_kp`/`_kd`，如需恢复请调用 `update_kp_kd()`。

---

#### `update_kp_kd(kp: np.ndarray, kd: np.ndarray) -> None`

在线更新 PD 增益。

| 参数 | 类型 | 说明 |
|------|------|------|
| `kp` | `np.ndarray` | 新的位置增益，形状必须与当前 `_kp` 相同 |
| `kd` | `np.ndarray` | 新的速度增益，形状必须与当前 `_kd` 相同 |

---

### 录制

#### `start_recording(save_dir: str) -> bool`

开始异步录制关节状态数据到指定目录。需要构造时传入 `joint_state_saver_factory`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `save_dir` | `str` | 数据保存目录路径 |
| **返回** | `bool` | 成功返回 `True` |

---

#### `stop_recording(prefix: str = "") -> Tuple[bool, str]`

停止录制。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prefix` | `str` | 保存文件名前缀（可选） |
| **返回** | `Tuple[bool, str]` | `(成功标志, 消息字符串)` |

---

### 生命周期

#### `close() -> None`

安全关闭机器人：停止控制线程，向电机链发送零力矩，释放资源。

```python
robot.close()
```

---

## 快速上手

```python
from i2rt.robots.get_robot import get_yam_robot
import numpy as np

# 初始化（默认零力矩模式，可手动拖动）
robot = get_yam_robot(channel="can0", gripper_type="linear_4310")

# 读取当前状态
q = robot.get_joint_pos()          # shape: (7,)
obs = robot.get_observations()     # dict with joint_pos, joint_vel, gripper_pos ...

# 命令目标位置（arm 6 dof + gripper 1 dof）
robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

# 平滑运动到零位
robot.move_joints(np.zeros(7), time_interval_s=3.0)

# 关闭
robot.close()
```

---

## 控制架构说明

`MotorChainRobot` 启动后台线程（`start_server`）以约 **250 Hz** 循环执行 `update()`：

1. 读取当前电机状态 → 更新 `_joint_state`
2. 通过 MuJoCo KDL 计算重力补偿力矩
3. 合并指令力矩并裁剪
4. 通过 CAN 总线发送给电机链

用户调用 `command_joint_pos` 等方法时仅写入带锁的指令缓冲区，不会阻塞控制循环。
