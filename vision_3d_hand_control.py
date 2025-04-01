# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pyrealsense2 as rs # 导入 RealSense SDK
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException

# --- SmartHandModbusTCP 类 (保持不变) ---
# ... (省略类的定义) ...
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

# --- MediaPipe 初始化 (保持不变) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 参数配置 (部分修改) ---
ROBOT_HAND_IP = '192.168.11.210'
ROBOT_HAND_PORT = 6000
# CAMERA_INDEX 不再需要

# RealSense 相机设置 (示例)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# --- 平滑滤波参数 (保持不变) ---
ALPHA = 0.3

# --- 角度范围常量 (可能需要重新标定) ---
# !! 注意：基于3D坐标计算的角度范围可能与基于2D的不同，需要重新标定 !!
FINGER_ANGLE_RANGE = (50, 180, 0, 1000) # 假设 0=闭合, 1000=张开
THUMB_IP_ANGLE_RANGE = (70, 180, 0, 1000) # 假设 0=闭合, 1000=张开
THUMB_ROTATION_ANGLE_RANGE = (20, 70, 0, 1000) # !! 必须重新标定 !!

# --- 辅助函数 (calculate_angle, map_angle 保持不变) ---
def calculate_angle(p1_coord, p2_coord, p3_coord):
    """计算点 p2 处的角度 (p1-p2-p3)，输入为3D坐标 NumPy 数组"""
    # 确保输入是 NumPy 数组
    p1 = np.array(p1_coord)
    p2 = np.array(p2_coord)
    p3 = np.array(p3_coord)

    v1 = p1 - p2
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 90.0 # 或者返回上次有效值

    dot_product = np.dot(v1, v2)
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def map_angle(angle, from_min, from_max, to_min, to_max):
    """将角度从 from_range 线性映射到 to_range"""
    # (函数体保持不变)
    clip_min = min(from_min, from_max)
    clip_max = max(from_min, from_max)
    angle = np.clip(angle, clip_min, clip_max)
    if abs(from_max - from_min) < 1e-6:
         mapped_value = (to_min + to_max) / 2
    else:
        mapped_value = to_min + (angle - from_min) * (to_max - to_min) / (from_max - from_min)
    target_min = min(to_min, to_max)
    target_max = max(to_min, to_max)
    mapped_value = np.clip(mapped_value, target_min, target_max)
    return mapped_value

# --- 主函数 (主要修改部分) ---
def main():
    # 初始化机器人手 (保持不变)
    hand = SmartHandModbusTCP(ip=ROBOT_HAND_IP, port=ROBOT_HAND_PORT)
    if not hand.connect():
        print("无法启动机器人手连接。")
        return

    # --- 初始化 Intel RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    # 获取设备产品线，以启用与特定设备兼容的特性
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line)) # D400 系列等

    # 配置数据流
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)

    # 启动管道
    profile = pipeline.start(config)

    # 获取深度传感器的缩放比例 (米/单位)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度传感器缩放比例: {depth_scale}")

    # 创建对齐对象 (将深度帧对齐到彩色帧)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 获取内参 (用于反投影) - 在循环外获取一次即可
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_intrinsics = depth_profile.get_intrinsics()
    color_intrinsics = color_profile.get_intrinsics() # 彩色帧内参通常不直接用于反投影深度点

    # 初始化滤波和平滑变量 (保持不变)
    initial_pose = [500.0] * 6
    filtered_robot_angles = np.array(initial_pose, dtype=float)
    first_frame = True

    # --- MediaPipe Hands 初始化 ---
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1) as hands:

        try: # 使用 try...finally 确保 pipeline 停止
            while True:
                # --- 获取和处理 RealSense 帧 ---
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames) # 获取对齐后的帧集

                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    print("丢失帧...")
                    continue

                # 将 RealSense 彩色帧转换为 NumPy 数组
                color_image = np.asanyarray(color_frame.get_data())
                # 获取图像尺寸
                height, width, _ = color_image.shape

                # --- MediaPipe 处理 ---
                # 为了提高性能，将图像标记为不可写
                color_image.flags.writeable = False
                # BGR -> RGB
                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                # 处理
                results = hands.process(image_rgb)
                # 转回 BGR 用于绘制
                color_image.flags.writeable = True
                # color_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 在 color_image 上直接绘制即可

                # 初始化目标角度和 3D 坐标
                target_robot_angles = np.zeros(6, dtype=float)
                joint_3d_positions = [None] * 21 # 存储 21 个关节点的 3D 坐标
                hand_detected_this_frame = False

                # --- 处理检测结果 ---
                if results.multi_hand_landmarks:
                    hand_detected_this_frame = True
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # --- 绘制 2D 骨架 ---
                    mp_drawing.draw_landmarks(
                        color_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # --- 获取关节点的 3D 坐标 ---
                    valid_landmarks = True
                    for i, lm in enumerate(hand_landmarks.landmark):
                        # 将归一化坐标转为像素坐标
                        x_pixel = int(lm.x * width)
                        y_pixel = int(lm.y * height)

                        # 防止像素坐标越界
                        if 0 <= x_pixel < width and 0 <= y_pixel < height:
                            # 获取该像素的深度值 (单位：米)
                            depth_m = aligned_depth_frame.get_distance(x_pixel, y_pixel)

                            # 如果深度有效 (大于0)
                            if depth_m > 0:
                                # 反投影计算 3D 坐标 (相对于相机)
                                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel], depth_m)
                                joint_3d_positions[i] = np.array(point_3d)
                                # 可选：在图像上绘制 3D 坐标或深度值
                                cv2.putText(color_image, f"{point_3d[0]:.2f},{point_3d[1]:.2f},{point_3d[2]:.2f}",
                                            (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
                            else:
                                joint_3d_positions[i] = None # 深度无效
                                valid_landmarks = False
                        else:
                            joint_3d_positions[i] = None # 像素越界
                            valid_landmarks = False

                    # --- 如果所有需要的关节点都有效，则计算角度 ---
                    if valid_landmarks:
                        try:
                            # 检查计算角度所需的点是否都有效 (非 None)
                            # (这里简化，实际应检查每个 calculate_angle 用到的点)
                            required_indices = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
                            if all(joint_3d_positions[idx] is not None for idx in required_indices):

                                # --- 计算角度 (使用 3D 坐标) ---
                                # DOF 0: 小指弯曲
                                angle_pinky = calculate_angle(joint_3d_positions[17], joint_3d_positions[18], joint_3d_positions[19])
                                target_robot_angles[0] = map_angle(angle_pinky, *FINGER_ANGLE_RANGE)
                                # ... (其他手指 DOF 1-3 类似)
                                angle_ring = calculate_angle(joint_3d_positions[13], joint_3d_positions[14], joint_3d_positions[15])
                                target_robot_angles[1] = map_angle(angle_ring, *FINGER_ANGLE_RANGE)
                                angle_middle = calculate_angle(joint_3d_positions[9], joint_3d_positions[10], joint_3d_positions[11])
                                target_robot_angles[2] = map_angle(angle_middle, *FINGER_ANGLE_RANGE)
                                angle_index = calculate_angle(joint_3d_positions[5], joint_3d_positions[6], joint_3d_positions[7])
                                target_robot_angles[3] = map_angle(angle_index, *FINGER_ANGLE_RANGE)

                                # DOF 4: 大拇指指尖弯曲 (IP) - 使用正确的点
                                angle_thumb_ip = calculate_angle(joint_3d_positions[2], joint_3d_positions[3], joint_3d_positions[4])
                                target_robot_angles[4] = map_angle(angle_thumb_ip, *THUMB_IP_ANGLE_RANGE)

                                # DOF 5: 大拇指旋转
                                angle_thumb_rot = calculate_angle(joint_3d_positions[5], joint_3d_positions[0], joint_3d_positions[2])
                                target_robot_angles[5] = map_angle(angle_thumb_rot, *THUMB_ROTATION_ANGLE_RANGE)
                                # print(f"Thumb Rotation Angle (3D): {angle_thumb_rot:.1f}")

                                # --- EMA 平滑滤波 ---
                                if first_frame:
                                    filtered_robot_angles[:] = target_robot_angles
                                    first_frame = False
                                else:
                                    filtered_robot_angles = ALPHA * target_robot_angles + (1 - ALPHA) * filtered_robot_angles

                                # --- 发送角度到机器人手 ---
                                if hand.client.is_socket_open():
                                    hand.set_angles(filtered_robot_angles)
                                else:
                                    print("连接已断开，尝试重新连接...")
                                    if hand.connect():
                                        hand.set_angles(filtered_robot_angles)
                            else:
                                print("警告：计算角度所需的部分关节点深度无效，跳过计算。")
                                # 保持上一个姿态

                        except Exception as e:
                            print(f"计算或发送角度时出错: {e}")
                            # 保持上一个姿态

                    else: # valid_landmarks is False
                         print("警告：部分或全部关节点深度无效。")
                         # 保持上一个姿态
                         first_frame = True # 重置滤波器初始化标志

                else: # No hand detected
                    # 保持上一个姿态
                    first_frame = True # 下次检测到手时重新初始化

                # --- 显示图像 ---
                display_angles = [int(round(a)) for a in filtered_robot_angles]
                cv2.putText(color_image, f"Target: {display_angles}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # (可选) 显示 Raw 值
                # if hand_detected_this_frame and valid_landmarks:
                #      raw_display = [int(round(a)) for a in target_robot_angles]
                #      cv2.putText(color_image, f"Raw:    {raw_display}", (10, 60),
                #                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

                cv2.imshow('Depth Camera Hand Control', color_image)

                # 按 'q' 键退出
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        finally:
            # --- 清理 ---
            print("正在停止 RealSense 管道...")
            pipeline.stop()
            print("正在关闭 Modbus 连接...")
            if hand.client.is_socket_open():
                 # hand.set_angles([int(round(p)) for p in initial_pose]) # 可选：发送初始姿态
                 time.sleep(0.1)
            hand.close()
            cv2.destroyAllWindows()
            print("程序结束。")

if __name__ == '__main__':
    main()