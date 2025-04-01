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

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Parameters ---
ROBOT_HAND_IP = '192.168.11.210'
ROBOT_HAND_PORT = 6000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
ALPHA = 0.3 # Smoothing factor

# --- Angle Ranges (Recalibration Needed!) ---
FINGER_ANGLE_RANGE = (50, 180, 0, 1000)
THUMB_IP_ANGLE_RANGE = (120, 160, 0, 1000)
THUMB_ROTATION_ANGLE_RANGE = (20, 32, 0, 1000)

# --- Depth Search Parameters ---
MIN_VALID_DEPTH = 0.1 # meters, min distance to register depth
MAX_VALID_DEPTH = 2.0 # meters, max distance
NEIGHBOR_SEARCH_RADIUS = 2 # Search up to 2 pixels away (5x5 area)

# --- Helper Functions ---

def calculate_angle(p1_coord, p2_coord, p3_coord):
    """Calculates angle at p2. Returns angle in degrees or None if input is invalid."""
    if p1_coord is None or p2_coord is None or p3_coord is None:
        return None
    p1, p2, p3 = np.array(p1_coord), np.array(p2_coord), np.array(p3_coord)
    v1, v2 = p1 - p2, p3 - p2
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6: return None
    dot_product = np.dot(v1, v2)
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def map_angle(angle, from_min, from_max, to_min, to_max):
    """Maps angle from one range to another. Returns float."""
    clip_min, clip_max = min(from_min, from_max), max(from_min, from_max)
    angle = np.clip(angle, clip_min, clip_max)
    if abs(from_max - from_min) < 1e-6: return (to_min + to_max) / 2
    mapped_value = to_min + (angle - from_min) * (to_max - to_min) / (from_max - from_min)
    target_min, target_max = min(to_min, to_max), max(to_min, to_max)
    return np.clip(mapped_value, target_min, target_max)

def get_valid_3d_point(x_pixel, y_pixel, depth_frame, depth_intrinsics,
                       min_depth=0.1, max_depth=2.0, search_radius=1):
    """
    Tries to get a valid 3D point for a given pixel (x_pixel, y_pixel).
    If depth at the original pixel is invalid, searches neighbors in an expanding box.
    Uses the first valid neighbor found for deprojection.
    Returns 3D point (NumPy array) or None.
    """
    height, width = depth_intrinsics.height, depth_intrinsics.width

    # Check bounds for the original target pixel itself
    if not (0 <= x_pixel < width and 0 <= y_pixel < height):
        return None

    # 1. Try original pixel
    depth_m = depth_frame.get_distance(x_pixel, y_pixel)
    if min_depth < depth_m < max_depth:
        try:
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_pixel, y_pixel], depth_m)
            return np.array(point_3d)
        except Exception:
            pass # Fall through to neighbor search if deprojection fails

    # 2. If original invalid, search neighbors in expanding box
    for r in range(1, search_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # Skip if not on the boundary of the current search radius 'r'
                # (already checked inner radii)
                if abs(dx) < r and abs(dy) < r:
                    continue

                nx, ny = x_pixel + dx, y_pixel + dy

                # Check bounds for neighbor
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_depth_m = depth_frame.get_distance(nx, ny)
                    if min_depth < neighbor_depth_m < max_depth:
                        try:
                            # Use neighbor's pixel and depth for deprojection
                            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nx, ny], neighbor_depth_m)
                            # print(f" Landmark ({x_pixel},{y_pixel}): Used neighbor ({nx},{ny}) depth {neighbor_depth_m:.3f}") # Debug
                            return np.array(point_3d)
                        except Exception:
                            continue # Try next neighbor if deprojection fails

    # 3. If no valid depth found in neighbors
    # print(f" Landmark ({x_pixel},{y_pixel}): No valid depth found.") # Debug
    return None

# --- Main Function ---
def main():
    # --- Initializations (Robot Hand, RealSense Pipeline, Alignment, Intrinsics) ---
    # (Same as before)
    hand = SmartHandModbusTCP(ip=ROBOT_HAND_IP, port=ROBOT_HAND_PORT)
    if not hand.connect(): return
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Error configuring or starting RealSense pipeline: {e}")
        if hand.client.is_socket_open(): hand.close()
        return
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()

    # --- Smoothing variables ---
    initial_pose = [500.0] * 6
    filtered_robot_angles = np.array(initial_pose, dtype=float)
    first_run_valid_angles = True

    # --- MediaPipe Hands Initialization ---
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1) as hands:

        try:
            while True:
                # --- Get and Align Frames ---
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as e:
                    print(f"RealSense: Failed to get frames: {e}")
                    break # Exit loop on persistent frame error
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not aligned_depth_frame or not color_frame: continue

                color_image = np.asanyarray(color_frame.get_data())
                height, width, _ = color_image.shape

                # --- MediaPipe Processing ---
                color_image.flags.writeable = False
                image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                color_image.flags.writeable = True

                # --- Process Detections ---
                joint_3d_positions = [None] * 21
                hand_detected_this_frame = False
                # Start with previous filtered state for potentially missing DOFs
                temp_target_angles = np.copy(filtered_robot_angles)

                if results.multi_hand_landmarks:
                    hand_detected_this_frame = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())

                    # --- Get 3D Coordinates using Neighbor Search ---
                    num_valid_points = 0
                    for i, lm in enumerate(hand_landmarks.landmark):
                        x_pixel, y_pixel = int(lm.x * width), int(lm.y * height)
                        # Call the helper function to get 3D point with neighbor search
                        point_3d = get_valid_3d_point(
                            x_pixel, y_pixel,
                            aligned_depth_frame,
                            depth_intrinsics,
                            min_depth=MIN_VALID_DEPTH,
                            max_depth=MAX_VALID_DEPTH,
                            search_radius=NEIGHBOR_SEARCH_RADIUS
                        )
                        joint_3d_positions[i] = point_3d
                        if point_3d is not None:
                            num_valid_points += 1
                            # Optional: Draw marker at original pixel if point found
                            # cv2.circle(color_image, (x_pixel, y_pixel), 3, (0, 255, 0), -1) # Green dot
                        # else:
                            # Optional: Draw marker if point not found
                            # cv2.circle(color_image, (x_pixel, y_pixel), 3, (0, 0, 255), -1) # Red dot


                    # --- Calculate Angles Per DOF (if points are valid) ---
                    # Using threshold > 15 as an example, tune as needed
                    if num_valid_points > 15:
                        angle_calculation_successful = False
                        try:
                            # Define required indices for each angle calculation for clarity
                            req_indices = {
                                0: [17, 18, 19], # Pinky
                                1: [13, 14, 15], # Ring
                                2: [9, 10, 11],  # Middle
                                3: [5, 6, 7],    # Index
                                4: [2, 3, 4],    # Thumb IP
                                5: [5, 0, 2]     # Thumb Rot
                            }
                            angle_funcs = { # Map DOF index to calculation function
                                0: lambda p: calculate_angle(p[17], p[18], p[19]),
                                1: lambda p: calculate_angle(p[13], p[14], p[15]),
                                2: lambda p: calculate_angle(p[9], p[10], p[11]),
                                3: lambda p: calculate_angle(p[5], p[6], p[7]),
                                4: lambda p: calculate_angle(p[2], p[3], p[4]),
                                5: lambda p: calculate_angle(p[5], p[0], p[2])
                            }
                            range_map = { # Map DOF index to range constant
                                0: FINGER_ANGLE_RANGE, 1: FINGER_ANGLE_RANGE, 2: FINGER_ANGLE_RANGE, 3: FINGER_ANGLE_RANGE,
                                4: THUMB_IP_ANGLE_RANGE, 5: THUMB_ROTATION_ANGLE_RANGE
                            }

                            # Iterate through DOFs
                            for dof_idx in range(6):
                                indices_needed = req_indices[dof_idx]
                                # Check if all required points for this DOF are valid
                                if all(joint_3d_positions[idx] is not None for idx in indices_needed):
                                    # Calculate angle
                                    angle = angle_funcs[dof_idx](joint_3d_positions)
                                    if angle is not None:
                                        # Map angle and update temp target
                                        temp_target_angles[dof_idx] = map_angle(angle, *range_map[dof_idx])
                                        angle_calculation_successful = True
                                # else: Keep the previous value in temp_target_angles

                            # --- Apply Smoothing Filter ---
                            if angle_calculation_successful:
                                if first_run_valid_angles:
                                    filtered_robot_angles[:] = temp_target_angles
                                    first_run_valid_angles = False
                                else:
                                    filtered_robot_angles = ALPHA * temp_target_angles + (1 - ALPHA) * filtered_robot_angles
                            else:
                                first_run_valid_angles = True

                        except Exception as e:
                            print(f"Error during angle calculation or mapping: {e}")
                            first_run_valid_angles = True
                    else:
                        # print(f"警告：有效关节点数量不足 ({num_valid_points})，跳过计算。")
                        first_run_valid_angles = True

                else: # No hand detected
                    first_run_valid_angles = True # Reset filter init

                # --- Send Command ---
                if hand.client.is_socket_open():
                    hand.set_angles(filtered_robot_angles)
                else:
                    if hand.connect(): hand.set_angles(filtered_robot_angles)

                # --- Display Image ---
                display_angles = [int(round(a)) for a in filtered_robot_angles]
                cv2.putText(color_image, f"Target: {display_angles}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Depth Camera Hand Control', color_image)

                # --- Exit Condition ---
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        finally:
            # --- Cleanup ---
            print("正在停止 RealSense 管道...")
            pipeline.stop()
            print("正在关闭 Modbus 连接...")
            if hand.client.is_socket_open():
                 time.sleep(0.1)
            hand.close()
            cv2.destroyAllWindows()
            print("程序结束。")

if __name__ == '__main__':
    main()