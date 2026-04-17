import os
import yaml

_YAML_PATH = os.environ.get("YAM_ROS_CONFIG")
if not _YAML_PATH:
    raise RuntimeError("未指定 config 路径，请通过 --config 传入 config.yaml 路径")

with open(_YAML_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# ── 硬件 ──────────────────────────────────────
CAN_CHANNEL  = str(_cfg["can_channel"])
GRIPPER_TYPE = str(_cfg["gripper_type"])
SITE_NAME    = str(_cfg["site_name"])

# ── ROS Bridge ────────────────────────────────
ROSBRIDGE_HOST = _cfg["rosbridge"]["host"]
ROSBRIDGE_PORT = _cfg["rosbridge"]["port"]

# ── 频率 ──────────────────────────────────────
CONTROL_RATE     = float(_cfg["rates"]["control_hz"])
PUBLISH_RATE     = float(_cfg["rates"]["publish_hz"])
WATCHDOG_TIMEOUT = float(_cfg["rates"]["watchdog_timeout_s"])

# ── 安全位置 ──────────────────────────────────
SAFE_JOINT_POS = list(_cfg["safe_position"]["joint_pos"])
SAFE_MOVE_TIME = float(_cfg["safe_position"]["move_time_s"])

# ── Twist 缩放 ────────────────────────────────
TRANSLATION_SCALE = float(_cfg["twist_scale"]["translation_m"])
ROTATION_SCALE    = float(_cfg["twist_scale"]["rotation_rad"])

# ── 关节名称 ──────────────────────────────────
JOINT_NAMES = list(_cfg["joints"]["names"])

# ── Topics ────────────────────────────────────
TOPIC_CMD     = _cfg["topics"]["subscribe"]["cmd"]
TOPIC_GRIPPER = _cfg["topics"]["subscribe"]["gripper"]
TOPIC_RESET   = _cfg["topics"]["subscribe"]["reset"]
TOPIC_JOINTS  = _cfg["topics"]["publish"]["joints"]
TOPIC_EE      = _cfg["topics"]["publish"]["ee"]
