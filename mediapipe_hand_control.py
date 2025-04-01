# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import math
import time # 用于可能的延迟控制
from collections import deque # 用于其他滤波方式（如果需要）
from pymodbus.client import ModbusTcpClient # 从您的代码中导入
from pymodbus.exceptions import ModbusException # 从您的代码中导入

# --- 从您提供的代码中复制 SmartHandModbusTCP 类 ---
class SmartHandModbusTCP:
    # (类的其余部分和之前一样，这里省略以保持简洁)
    # ... (确保 SmartHandModbusTCP 类的完整定义在这里或可导入) ...
    FINGER_SEGMENTS_INFO = {
        1: [  # finger1 (小拇指)
            {"name": "指端", "start": 3000, "bytes": 18, "shape": (3, 3)},
            {"name": "指尖", "start": 3018, "bytes": 192, "shape": (12, 8)},
            {"name": "指腹", "start": 3210, "bytes": 160, "shape": (10, 8)}
        ],
        2: [  # finger2 (无名指)
            {"name": "指端", "start": 3370, "bytes": 18, "shape": (3, 3)},
            {"name": "指尖", "start": 3388, "bytes": 192, "shape": (12, 8)},
            {"name": "指腹", "start": 3580, "bytes": 160, "shape": (10, 8)}
        ],
        3: [  # finger3 (中指)
            {"name": "指端", "start": 3740, "bytes": 18, "shape": (3, 3)},
            {"name": "指尖", "start": 3758, "bytes": 192, "shape": (12, 8)},
            {"name": "指腹", "start": 3950, "bytes": 160, "shape": (10, 8)}
        ],
        4: [  # finger4 (食指)
            {"name": "指端", "start": 4110, "bytes": 18, "shape": (3, 3)},
            {"name": "指尖", "start": 4128, "bytes": 192, "shape": (12, 8)},
            {"name": "指腹", "start": 4320, "bytes": 160, "shape": (10, 8)}
        ],
        5: [  # finger5 (大拇指)，共4段：指端/指尖/指中/指腹
            {"name": "指端", "start": 4480, "bytes": 18, "shape": (3, 3)},
            {"name": "指尖", "start": 4498, "bytes": 192, "shape": (12, 8)},
            {"name": "指中", "start": 4690, "bytes": 18, "shape": (3, 3)},
            {"name": "指腹", "start": 4708, "bytes": 192, "shape": (12, 8)}
        ],
    }

    def __init__(self, ip='192.168.11.210', port=6000, unit_id=0xFF):
        self.client = ModbusTcpClient(ip, port=port)
        self.unit_id = unit_id
        print(f"尝试连接到 {ip}:{port}...")

    def connect(self):
        if not self.client.connect():
            print(f"错误: 无法连接到设备 {self.client.host}:{self.client.port}")
            return False
        print("Modbus 连接成功.")
        return True

    def close(self):
        self.client.close()
        print("Modbus 连接已关闭.")

    def read_registers(self, address, count):
        if not self.client.is_socket_open():
            # print("错误: Modbus 连接未打开，无法读取.") # Reduce console spam
            return None
        try:
            result = self.client.read_holding_registers(address=address, count=count, slave=self.unit_id)
            if result.isError():
                # print(f"读取寄存器失败 (地址 {address}, 数量 {count}): {result}") # Reduce console spam
                return None
            data = np.array(result.registers, dtype='<u2')
            return data.tolist()
        except ModbusException as e:
            print(f"Modbus 读取异常: {e}")
            return None
        except Exception as e:
            print(f"读取寄存器时发生意外错误: {e}")
            return None

    def write_multiple_registers(self, address, values):
        if not self.client.is_socket_open():
            # print("错误: Modbus 连接未打开，无法写入.") # Reduce console spam
            return False
        # 确保值是整数列表
        int_values = [int(round(v)) for v in values]
        try:
            # print(f"DEBUG: 发送角度: {int_values}") # Debug: 打印实际发送的值
            result = self.client.write_registers(address=address, values=int_values, slave=self.unit_id)
            if result.isError():
                print(f"写入多个寄存器失败 (地址 {address}): {result}")
                return False
            return True
        except ModbusException as e:
            print(f"Modbus 写入异常: {e}")
            return False
        except Exception as e:
            print(f"写入多个寄存器时发生意外错误: {e}")
            return False

    def set_angles(self, angle_list):
        """
        设置各自由度的目标角度
        :param angle_list: 包含6个角度值的列表/Numpy数组，范围为 0~1000 或 -1
        """
        if len(angle_list) != 6:
            print("错误: set_angles 需要提供6个角度值")
            return False
        # 验证和转换在 write_multiple_registers 中处理
        return self.write_multiple_registers(1486, angle_list) # 1486 是目标角度寄存器地址

    # --- 其他方法 (get_angles, get_tactile_data 等保持不变) ---
    def get_angles(self):
        """获取当前各自由度的角度"""
        return self.read_registers(1546, 6) # 1546 是当前角度寄存器地址

    # 省略 get_tactile_data, get_finger_segments, save_config, clear_error, set_gesture...


# --- 视觉处理和控制逻辑 ---

# MediaPipe 初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 参数配置 ---
ROBOT_HAND_IP = '192.168.11.210' # !!! 修改为你的灵巧手 IP
ROBOT_HAND_PORT = 6000
CAMERA_INDEX = 0 # 摄像头索引，0 通常是默认摄像头

# --- 平滑滤波参数 ---
ALPHA = 0.3 # EMA 平滑因子 (0 < ALPHA <= 1)。值越小越平滑，但延迟越大。可以尝试 0.2, 0.3, 0.4 等

# !!! 关键：定义人手关节角度范围和机器人手角度范围的映射 !!!
# !! 再次确认: 机器人控制值 0 = 闭合/内收/极限A, 1000 = 张开/外展/极限B !!
# !! (请根据您的实际测试结果确认 0 和 1000 的具体含义，并调整这里的映射) !!
# 格式：(human_min_angle, human_max_angle, robot_limit_corresponding_to_min, robot_limit_corresponding_to_max)
# 注意：角度单位是度(degrees)

# 1. 手指弯曲/伸直 (用于 食指、中指、无名指、小指)
# 人手弯曲 (角度小, ~50度) -> 机器人值 A (e.g., 0)
# 人手伸直 (角度大, ~180度) -> 机器人值 B (e.g., 1000)
FINGER_ANGLE_RANGE = (50, 180, 0, 1000) # 假设 0=闭合, 1000=张开

# 2. 大拇指指尖弯曲/伸直 (Thumb IP Joint)
# 人手拇指弯曲 (角度小, ~70度) -> 机器人值 A (e.g., 0)
# 人手拇指伸直 (角度大, ~180度) -> 机器人值 B (e.g., 1000)
THUMB_IP_ANGLE_RANGE = (120, 140, 0, 1000) # 假设 0=闭合, 1000=张开

# 3. 大拇指旋转 (Thumb Rotation/Opposition) - ***需要仔细标定***
#   我们用 掌根(0) 到 食指根部(5) 的向量 与 掌根(0) 到 大拇指第二指节(2) 的向量 的夹角来近似
#   需要观察这个角度在拇指旋转时的变化范围 (打印 angle_thumb_rot 来观察)
#   假设：拇指靠近手掌（旋转角小）-> 机器人值 A (e.g., 0)
#   假设：拇指远离手掌（旋转角大）-> 机器人值 B (e.g., 1000)
#   !!! 这个范围 (20, 70) 是完全猜测的，您必须通过打印角度值来确定实际范围 !!!
THUMB_ROTATION_ANGLE_RANGE = (160, 172, 0, 1000)  # 假设 0=某个旋转极限, 1000=另一个旋转极限

# 辅助函数：计算三个点之间的角度 (保持不变)
def calculate_angle(p1, p2, p3):
    """计算点 p2 处的角度 (p1-p2-p3)"""
    # 将 landmark 转换为 numpy array (如果它们还不是)
    p1_np = np.array([p1.x, p1.y, p1.z]) if hasattr(p1, 'x') else np.array(p1)
    p2_np = np.array([p2.x, p2.y, p2.z]) if hasattr(p2, 'x') else np.array(p2)
    p3_np = np.array([p3.x, p3.y, p3.z]) if hasattr(p3, 'x') else np.array(p3)

    v1 = p1_np - p2_np
    v2 = p3_np - p2_np

    # 计算向量模长
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Use a small threshold instead of == 0
        # print("Warning: Zero length vector encountered in angle calculation.")
        # 返回一个中间值或者上一个有效值可能比返回固定值更好，但这里简化处理
        return 90.0 # 返回一个中间角度或标记为无效

    # 计算点积
    dot_product = np.dot(v1, v2)

    # 避免除以零和 arccos 输入超出范围
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# 辅助函数：将角度从一个范围映射到另一个范围 (保持不变)
def map_angle(angle, from_min, from_max, to_min, to_max):
    """将角度从 from_range 线性映射到 to_range"""
    # Clamp angle to the source range
    clip_min = min(from_min, from_max)
    clip_max = max(from_min, from_max)
    angle = np.clip(angle, clip_min, clip_max)

    # Map
    if abs(from_max - from_min) < 1e-6:
         # Avoid division by zero; return midpoint or one limit
         mapped_value = (to_min + to_max) / 2
    else:
        mapped_value = to_min + (angle - from_min) * (to_max - to_min) / (from_max - from_min)

    # Clamp result to the target range
    target_min = min(to_min, to_max)
    target_max = max(to_min, to_max)
    mapped_value = np.clip(mapped_value, target_min, target_max)

    return mapped_value # Return float for EMA

# 主函数
def main():
    # 初始化机器人手
    hand = SmartHandModbusTCP(ip=ROBOT_HAND_IP, port=ROBOT_HAND_PORT)
    if not hand.connect():
        print("无法启动，请检查机器人手连接。")
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头索引 {CAMERA_INDEX}")
        hand.close()
        return

    # 初始化滤波后的角度值 (使用 NumPy 数组方便计算)
    # 初始化为中间姿态可能更平稳 (假设 500 是中间值)
    initial_pose = [500.0] * 6 # 使用浮点数进行滤波计算
    # 或者如果 0 是张开/安全姿态，则用 [0.0] * 6
    # 或者如果 1000 是张开/安全姿态，则用 [1000.0] * 6
    filtered_robot_angles = np.array(initial_pose, dtype=float)
    first_frame = True # 标记是否是第一帧，用于初始化滤波器

    # 设置 MediaPipe Hands
    with mp_hands.Hands(
            model_complexity=0, # 0: Lite, 1: Full
            min_detection_confidence=0.7, # 稍提高置信度可能有助于减少噪声
            min_tracking_confidence=0.7,  # 同上
            max_num_hands=1) as hands: # 只处理一只手

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("忽略空的摄像头帧.")
                continue

            # 水平翻转图像，使显示更自然
            image = cv2.flip(image, 1)

            # 提高性能：将图像标记为不可写
            image.flags.writeable = False
            # 转换颜色 BGR -> RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # MediaPipe 处理
            results = hands.process(image_rgb)

            # 准备绘制 (图像设为可写)
            image.flags.writeable = True
            # 注意：不需要再转回 BGR，因为绘制可以直接在原图(已翻转)上进行

            # 初始化本次计算的目标角度列表
            target_robot_angles = np.zeros(6, dtype=float) # 使用浮点数
            hand_detected_this_frame = False

            # 如果检测到手
            if results.multi_hand_landmarks:
                hand_detected_this_frame = True
                # 只处理检测到的第一只手
                hand_landmarks = results.multi_hand_landmarks[0]
                lm = hand_landmarks.landmark # 获取关节点列表

                # --- 绘制手部骨架 ---
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- 计算关节角度 (根据用户最新的DOF映射) ---
                # DOF 0: 小指弯曲
                # DOF 1: 无名指弯曲
                # DOF 2: 中指弯曲
                # DOF 3: 食指弯曲
                # DOF 4: 大拇指弯曲 (IP)
                # DOF 5: 大拇指旋转

                try:
                    # DOF 0: 小指弯曲 (Pinky Finger PIP)
                    angle_pinky = calculate_angle(lm[17], lm[18], lm[19])
                    target_robot_angles[0] = map_angle(angle_pinky, *FINGER_ANGLE_RANGE)

                    # DOF 1: 无名指弯曲 (Ring Finger PIP)
                    angle_ring = calculate_angle(lm[13], lm[14], lm[15])
                    target_robot_angles[1] = map_angle(angle_ring, *FINGER_ANGLE_RANGE)

                    # DOF 2: 中指弯曲 (Middle Finger PIP)
                    angle_middle = calculate_angle(lm[9], lm[10], lm[11])
                    target_robot_angles[2] = map_angle(angle_middle, *FINGER_ANGLE_RANGE)

                    # DOF 3: 食指弯曲 (Index Finger PIP)
                    angle_index = calculate_angle(lm[5], lm[6], lm[7])
                    target_robot_angles[3] = map_angle(angle_index, *FINGER_ANGLE_RANGE)

                    # DOF 4: 大拇指指尖弯曲 (Thumb IP)
                    angle_thumb_ip = calculate_angle(lm[2], lm[3], lm[4])
                    target_robot_angles[4] = map_angle(angle_thumb_ip, *THUMB_IP_ANGLE_RANGE)
                    # print(f"Thumb Belt Angle (deg): {angle_thumb_ip:.1f}")

                    # DOF 5: 大拇指旋转 (Thumb Rotation)
                    # 计算 掌根(0)-食指根(5) 和 掌根(0)-拇指第二指节(2) 的夹角
                    angle_thumb_rot = calculate_angle(lm[3], lm[2], lm[1])
                    target_robot_angles[5] = map_angle(angle_thumb_rot, *THUMB_ROTATION_ANGLE_RANGE)
                    # 打印用于标定范围 (取消注释以观察):
                    print(f"Thumb Rotation Angle (deg): {angle_thumb_rot:.1f}")


                    # --- EMA 平滑滤波 ---
                    if first_frame:
                        # 第一帧用计算值直接初始化滤波器
                        filtered_robot_angles[:] = target_robot_angles
                        first_frame = False
                    else:
                        # 应用 EMA 公式
                        filtered_robot_angles = ALPHA * target_robot_angles + (1 - ALPHA) * filtered_robot_angles

                    # --- 发送角度到机器人手 ---
                    if hand.client.is_socket_open():
                        # 发送滤波后的角度 (set_angles 内部会处理取整)
                        success_write = hand.set_angles(filtered_robot_angles)
                        # 可以在写入失败时增加重试或日志
                        # if not success_write:
                        #      print("发送角度失败。")
                    else:
                        print("连接已断开，尝试重新连接...")
                        if hand.connect(): # 尝试重连后再次发送
                             hand.set_angles(filtered_robot_angles)

                except IndexError:
                     print("错误: 访问 landmarks 时索引越界。手部可能未完全检测到。")
                     # 保持上一个姿态
                except Exception as e:
                    print(f"处理手部数据或发送命令时出错: {e}")
                    # 保持上一个姿态

            # 如果没有检测到手
            else:
                 # 保持上一个姿态 (filtered_robot_angles 不会更新)
                 # 可以选择平滑恢复到默认姿态（见上一版代码注释）
                 first_frame = True # 下次检测到手时重新初始化滤波器

            # 显示图像
            # 在图像上显示 *滤波后* 的目标角度值 (取整后显示)
            display_angles = [int(round(a)) for a in filtered_robot_angles]
            cv2.putText(image, f"Target: {display_angles}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # (可选) 显示原始计算角度（用于调试）
            if hand_detected_this_frame:
                 raw_display = [int(round(a)) for a in target_robot_angles]
                 cv2.putText(image, f"Raw:    {raw_display}", (10, 60),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

            cv2.imshow('MediaPipe Hands Control', image)

            # 按 'q' 键退出
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()