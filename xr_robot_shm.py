"""
XR-机械臂 共享内存通信协议

用于同一主机上两个 conda 环境之间的超低延迟数据交换。
使用 Python 标准库 multiprocessing.shared_memory，零额外依赖。

数据布局 (共享内存 buffer):
═══════════════════════════════════════════════════════════════
  偏移    字节数    内容
───────────────────────────────────────────────────────────────
  XR → 机械臂方向:
  0       8        timestamp (float64, 秒)
  8       4        control_flags (uint32):
                     bit 0: is_active (XR端是否激活)
                     bit 1: request_init (请求重新校准)
                     bit 2: emergency_stop (紧急停止)
  12      56       right_controller_pose (7 x float64: x,y,z,qx,qy,qz,qw)
  68      56       left_controller_pose  (7 x float64: x,y,z,qx,qy,qz,qw)
  124     8        right_trigger (float64, 0~1)
  132     8        left_trigger  (float64, 0~1)
  140     8        right_grip    (float64, 0~1)
  148     8        left_grip     (float64, 0~1)
  156     4        buttons_packed (uint32, 每个bit一个按钮)
  ── 小计: 160 字节 ──

  机械臂 → XR 方向 (反馈, 可选):
  160     8        robot_timestamp (float64)
  168     4        robot_status (uint32):
                     bit 0: robot_connected
                     bit 1: ik_ok
                     bit 2: is_tracking
  172     24       robot_ee_pos_right (3 x float64: x,y,z)
  196     24       robot_ee_pos_left  (3 x float64: x,y,z)
  220     8        ik_success_rate (float64, 0~100)
  ── 小计: 68 字节 ──

  总计: 228 字节
═══════════════════════════════════════════════════════════════

使用方法:
  XR 端:     writer = XRSharedMemWriter()
  机械臂端:  reader = XRSharedMemReader()
"""

import struct
import time
import numpy as np
from multiprocessing import shared_memory


# ======================== 常量 ========================

SHM_NAME = "xr_robot_bridge"       # 共享内存名称
SHM_SIZE = 256                      # 总大小 (略大于实际需要, 留余量)

# XR → Robot 偏移量
OFFSET_TIMESTAMP      = 0          # float64
OFFSET_CONTROL_FLAGS  = 8          # uint32
OFFSET_RIGHT_POSE     = 12         # 7 x float64
OFFSET_LEFT_POSE      = 68         # 7 x float64
OFFSET_RIGHT_TRIGGER  = 124        # float64
OFFSET_LEFT_TRIGGER   = 132        # float64
OFFSET_RIGHT_GRIP     = 140        # float64
OFFSET_LEFT_GRIP      = 148        # float64
OFFSET_BUTTONS        = 156        # uint32

# Robot → XR 偏移量 (反馈)
OFFSET_ROBOT_TIMESTAMP    = 160    # float64
OFFSET_ROBOT_STATUS       = 168    # uint32
OFFSET_ROBOT_EE_POS_R     = 172    # 3 x float64
OFFSET_ROBOT_EE_POS_L     = 196    # 3 x float64
OFFSET_IK_SUCCESS_RATE    = 220    # float64

# Control flags 位定义
FLAG_IS_ACTIVE       = 0x01
FLAG_REQUEST_INIT    = 0x02
FLAG_EMERGENCY_STOP  = 0x04
FLAG_RESET_TO_ZERO   = 0x08

# Robot status 位定义
STATUS_CONNECTED     = 0x01
STATUS_IK_OK         = 0x02
STATUS_TRACKING      = 0x04

# 按钮位定义
BTN_A             = 0x01
BTN_B             = 0x02
BTN_X             = 0x04
BTN_Y             = 0x08
BTN_LEFT_MENU     = 0x10
BTN_RIGHT_MENU    = 0x20
BTN_LEFT_STICK    = 0x40
BTN_RIGHT_STICK   = 0x80

# 超时检测
TIMEOUT_THRESHOLD = 0.5  # 秒, 超过此时间未更新认为连接断开


# ======================== XR 端 (写入者) ========================

class XRSharedMemWriter:
    """XR 端: 写入手柄数据到共享内存"""
    
    def __init__(self):
        """创建共享内存 (如果已存在则先清理再创建)"""
        # 先尝试清理旧的
        try:
            old_shm = shared_memory.SharedMemory(name=SHM_NAME)
            old_shm.close()
            old_shm.unlink()
            print(f"[XR Writer] 清理旧的共享内存: {SHM_NAME}")
        except FileNotFoundError:
            pass
        
        self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        # 清零
        self.shm.buf[:SHM_SIZE] = bytearray(SHM_SIZE)
        print(f"[XR Writer] 共享内存已创建: {SHM_NAME} ({SHM_SIZE} bytes)")
    
    def write_xr_data(self, right_pose, left_pose, 
                      right_trigger, left_trigger,
                      right_grip, left_grip,
                      buttons_dict, 
                      is_active=True, request_init=False, emergency_stop=False,
                      reset_to_zero=False):
        """写入一帧 XR 数据
        
        Args:
            right_pose: 右手柄位姿 [x,y,z,qx,qy,qz,qw] (7,)
            left_pose:  左手柄位姿 [x,y,z,qx,qy,qz,qw] (7,)
            right_trigger: 右trigger值 [0-1]
            left_trigger:  左trigger值 [0-1]
            right_grip:    右grip值 [0-1]
            left_grip:     左grip值 [0-1]
            buttons_dict:  按钮字典 {'A': bool, 'B': bool, ...}
            is_active:     是否激活控制
            request_init:  请求重新校准
            emergency_stop: 紧急停止
            reset_to_zero: 回零位
        """
        buf = self.shm.buf
        
        # 时间戳
        struct.pack_into('d', buf, OFFSET_TIMESTAMP, time.time())
        
        # 控制标志
        flags = 0
        if is_active:       flags |= FLAG_IS_ACTIVE
        if request_init:    flags |= FLAG_REQUEST_INIT
        if emergency_stop:  flags |= FLAG_EMERGENCY_STOP
        if reset_to_zero:   flags |= FLAG_RESET_TO_ZERO
        struct.pack_into('I', buf, OFFSET_CONTROL_FLAGS, flags)
        
        # 手柄位姿
        struct.pack_into('7d', buf, OFFSET_RIGHT_POSE, *right_pose[:7])
        struct.pack_into('7d', buf, OFFSET_LEFT_POSE, *left_pose[:7])
        
        # Trigger 和 Grip
        struct.pack_into('d', buf, OFFSET_RIGHT_TRIGGER, float(right_trigger))
        struct.pack_into('d', buf, OFFSET_LEFT_TRIGGER, float(left_trigger))
        struct.pack_into('d', buf, OFFSET_RIGHT_GRIP, float(right_grip))
        struct.pack_into('d', buf, OFFSET_LEFT_GRIP, float(left_grip))
        
        # 按钮打包
        btn = 0
        if buttons_dict.get('A', False):            btn |= BTN_A
        if buttons_dict.get('B', False):            btn |= BTN_B
        if buttons_dict.get('X', False):            btn |= BTN_X
        if buttons_dict.get('Y', False):            btn |= BTN_Y
        if buttons_dict.get('left_menu', False):    btn |= BTN_LEFT_MENU
        if buttons_dict.get('right_menu', False):   btn |= BTN_RIGHT_MENU
        if buttons_dict.get('left_stick', False):   btn |= BTN_LEFT_STICK
        if buttons_dict.get('right_stick', False):  btn |= BTN_RIGHT_STICK
        struct.pack_into('I', buf, OFFSET_BUTTONS, btn)
    
    def read_robot_feedback(self):
        """读取机械臂反馈 (可选)
        
        Returns:
            dict: 包含 robot_connected, ik_ok, is_tracking, 
                  ee_pos_right, ee_pos_left, ik_success_rate
                  如果机械臂端未运行, 返回 None
        """
        buf = self.shm.buf
        
        robot_ts = struct.unpack_from('d', buf, OFFSET_ROBOT_TIMESTAMP)[0]
        if robot_ts == 0:
            return None
        
        # 超时检查
        if time.time() - robot_ts > TIMEOUT_THRESHOLD:
            return None
        
        status = struct.unpack_from('I', buf, OFFSET_ROBOT_STATUS)[0]
        ee_r = struct.unpack_from('3d', buf, OFFSET_ROBOT_EE_POS_R)
        ee_l = struct.unpack_from('3d', buf, OFFSET_ROBOT_EE_POS_L)
        ik_rate = struct.unpack_from('d', buf, OFFSET_IK_SUCCESS_RATE)[0]
        
        return {
            'robot_connected': bool(status & STATUS_CONNECTED),
            'ik_ok': bool(status & STATUS_IK_OK),
            'is_tracking': bool(status & STATUS_TRACKING),
            'ee_pos_right': np.array(ee_r),
            'ee_pos_left': np.array(ee_l),
            'ik_success_rate': ik_rate,
        }
    
    def close(self):
        """关闭并清理共享内存"""
        try:
            self.shm.close()
            self.shm.unlink()
            print(f"[XR Writer] 共享内存已释放: {SHM_NAME}")
        except Exception as e:
            print(f"[XR Writer] 清理共享内存时出错: {e}")
    
    def __del__(self):
        self.close()


# ======================== 机械臂端 (读取者) ========================

class XRSharedMemReader:
    """机械臂端: 从共享内存读取 XR 数据"""
    
    def __init__(self, timeout_retries=50, retry_interval=0.1):
        """连接到共享内存
        
        Args:
            timeout_retries: 最大重试次数
            retry_interval: 重试间隔 (秒)
        """
        self.shm = None
        
        for i in range(timeout_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=SHM_NAME)
                print(f"[Robot Reader] 已连接到共享内存: {SHM_NAME}")
                return
            except FileNotFoundError:
                if i % 10 == 0:
                    print(f"[Robot Reader] 等待 XR 端启动... ({i}/{timeout_retries})")
                time.sleep(retry_interval)
        
        raise RuntimeError(f"[Robot Reader] 超时: 无法连接到共享内存 '{SHM_NAME}', 请先启动 XR 端")
    
    def read_xr_data(self):
        """读取一帧 XR 数据
        
        Returns:
            dict: {
                'timestamp': float,
                'is_active': bool,
                'request_init': bool,
                'emergency_stop': bool,
                'right_pose': np.ndarray (7,),
                'left_pose': np.ndarray (7,),
                'right_trigger': float,
                'left_trigger': float,
                'right_grip': float,
                'left_grip': float,
                'buttons': dict,
                'is_fresh': bool,   # 数据是否新鲜 (未超时)
            }
        """
        buf = self.shm.buf
        
        # 时间戳
        ts = struct.unpack_from('d', buf, OFFSET_TIMESTAMP)[0]
        is_fresh = (time.time() - ts) < TIMEOUT_THRESHOLD if ts > 0 else False
        
        # 控制标志
        flags = struct.unpack_from('I', buf, OFFSET_CONTROL_FLAGS)[0]
        
        # 手柄位姿
        right_pose = np.array(struct.unpack_from('7d', buf, OFFSET_RIGHT_POSE))
        left_pose = np.array(struct.unpack_from('7d', buf, OFFSET_LEFT_POSE))
        
        # Trigger 和 Grip
        right_trigger = struct.unpack_from('d', buf, OFFSET_RIGHT_TRIGGER)[0]
        left_trigger = struct.unpack_from('d', buf, OFFSET_LEFT_TRIGGER)[0]
        right_grip = struct.unpack_from('d', buf, OFFSET_RIGHT_GRIP)[0]
        left_grip = struct.unpack_from('d', buf, OFFSET_LEFT_GRIP)[0]
        
        # 按钮解包
        btn = struct.unpack_from('I', buf, OFFSET_BUTTONS)[0]
        buttons = {
            'A': bool(btn & BTN_A),
            'B': bool(btn & BTN_B),
            'X': bool(btn & BTN_X),
            'Y': bool(btn & BTN_Y),
            'left_menu': bool(btn & BTN_LEFT_MENU),
            'right_menu': bool(btn & BTN_RIGHT_MENU),
            'left_stick': bool(btn & BTN_LEFT_STICK),
            'right_stick': bool(btn & BTN_RIGHT_STICK),
        }
        
        return {
            'timestamp': ts,
            'is_active': bool(flags & FLAG_IS_ACTIVE),
            'request_init': bool(flags & FLAG_REQUEST_INIT),
            'emergency_stop': bool(flags & FLAG_EMERGENCY_STOP),
            'reset_to_zero': bool(flags & FLAG_RESET_TO_ZERO),
            'right_pose': right_pose,
            'left_pose': left_pose,
            'right_trigger': right_trigger,
            'left_trigger': left_trigger,
            'right_grip': right_grip,
            'left_grip': left_grip,
            'buttons': buttons,
            'is_fresh': is_fresh,
        }
    
    def write_robot_feedback(self, ee_pos_right=None, ee_pos_left=None,
                             ik_success_rate=0.0,
                             connected=True, ik_ok=True, tracking=False):
        """写入机械臂反馈 (可选)"""
        buf = self.shm.buf
        
        struct.pack_into('d', buf, OFFSET_ROBOT_TIMESTAMP, time.time())
        
        status = 0
        if connected: status |= STATUS_CONNECTED
        if ik_ok:     status |= STATUS_IK_OK
        if tracking:  status |= STATUS_TRACKING
        struct.pack_into('I', buf, OFFSET_ROBOT_STATUS, status)
        
        if ee_pos_right is not None:
            struct.pack_into('3d', buf, OFFSET_ROBOT_EE_POS_R, *ee_pos_right[:3])
        if ee_pos_left is not None:
            struct.pack_into('3d', buf, OFFSET_ROBOT_EE_POS_L, *ee_pos_left[:3])
        
        struct.pack_into('d', buf, OFFSET_IK_SUCCESS_RATE, float(ik_success_rate))
    
    def close(self):
        """关闭共享内存连接 (不 unlink, 由 Writer 负责)"""
        try:
            self.shm.close()
            print(f"[Robot Reader] 共享内存连接已关闭")
        except Exception:
            pass
    
    def __del__(self):
        self.close()


# ======================== 测试工具 ========================

def test_writer():
    """测试写入端 (模拟 XR 数据)"""
    print("测试共享内存写入端 (模拟XR数据)...")
    writer = XRSharedMemWriter()
    
    try:
        t = 0
        while True:
            # 模拟手柄数据
            right_pose = [0.3 + 0.1 * np.sin(t), 0.2, 0.1, 0, 0, 0, 1]
            left_pose = [-0.3 + 0.1 * np.sin(t), 0.2, 0.1, 0, 0, 0, 1]
            
            writer.write_xr_data(
                right_pose=right_pose, left_pose=left_pose,
                right_trigger=abs(np.sin(t)), left_trigger=0.0,
                right_grip=0.0, left_grip=0.0,
                buttons_dict={}, is_active=True
            )
            
            # 读取机械臂反馈
            fb = writer.read_robot_feedback()
            fb_str = f"机械臂: 已连接" if fb and fb['robot_connected'] else "机械臂: 未连接"
            
            print(f"\r[Writer] t={t:.2f} right_x={right_pose[0]:.3f} trigger={abs(np.sin(t)):.2f} | {fb_str}    ",
                  end="", flush=True)
            
            t += 0.02
            time.sleep(0.002)  # 500Hz
            
    except KeyboardInterrupt:
        print("\n写入端结束")
    finally:
        writer.close()


def test_reader():
    """测试读取端"""
    print("测试共享内存读取端...")
    reader = XRSharedMemReader()
    
    try:
        while True:
            data = reader.read_xr_data()
            
            rp = data['right_pose']
            print(f"\r[Reader] fresh={data['is_fresh']} active={data['is_active']} "
                  f"right:[{rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f}] "
                  f"trigger={data['right_trigger']:.2f}    ", end="", flush=True)
            
            # 写入模拟反馈
            reader.write_robot_feedback(
                ee_pos_right=[0.2, 0.0, 0.1],
                connected=True, ik_ok=True, tracking=data['is_active']
            )
            
            time.sleep(0.002)
            
    except KeyboardInterrupt:
        print("\n读取端结束")
    finally:
        reader.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'reader':
        test_reader()
    else:
        test_writer()