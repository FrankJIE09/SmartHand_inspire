import time
import json  # 导入json库用于文件读写
import os    # 导入os库用于检查文件是否存在
from pynput import keyboard
from smart_hand_controller import SmartHandModbusTCP

# --- 配置 ---
POSITION_FILE = "hand_positions.json"  # 保存位置的文件名
STEP = 50  # 每次按键变化的步长

# --- 全局变量 ---
current_angles = [0, 0, 0, 0, 0, 0] # 初始化默认角度
saved_positions = {} # 用于存储已保存位置的字典 { '1': [angles], '2': [angles], ... }

# --- 按键映射 ---
# 增加键映射：q, w, e, r, t, y -> 自由度 5, 4, 3, 2, 1, 0
inc_keys = {'q': 5, 'w': 4, 'e': 3, 'r': 2, 't': 1, 'y': 0}
# 减少键映射：a, s, d, f, g, h -> 自由度 5, 4, 3, 2, 1, 0
dec_keys = {'a': 5, 's': 4, 'd': 3, 'f': 2, 'g': 1, 'h': 0}
# 保存键映射：'1' 到 '6'
save_keys = {'1', '2', '3', '4', '5', '6'}

# --- 函数 ---
def load_positions():
    """从文件加载已保存的位置"""
    global saved_positions
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, 'r') as f:
                saved_positions = json.load(f)
                print(f"已从 {POSITION_FILE} 加载 {len(saved_positions)} 个位置。")
        except json.JSONDecodeError:
            print(f"警告: {POSITION_FILE} 文件格式错误，将使用空的位置列表。")
            saved_positions = {}
        except Exception as e:
            print(f"加载位置文件时出错: {e}")
            saved_positions = {}
    else:
        print(f"位置文件 {POSITION_FILE} 不存在，将创建新的文件。")
        saved_positions = {}

def save_positions_to_file():
    """将当前保存的位置写入文件"""
    try:
        with open(POSITION_FILE, 'w') as f:
            json.dump(saved_positions, f, indent=4) # indent参数使json文件更易读
        # print(f"位置已保存到 {POSITION_FILE}") # 可以在每次保存时打印，但可能会刷屏
    except Exception as e:
        print(f"保存位置到文件时出错: {e}")

def on_press(key):
    """处理按键事件"""
    global current_angles, saved_positions

    # 按下 ESC 键退出监听
    if key == keyboard.Key.esc:
        print("收到退出信号...")
        return False # 返回 False 停止监听器

    try:
        k = key.char.lower() # 获取按键字符并转为小写
    except AttributeError:
        return # 忽略非字符按键 (如 Shift, Ctrl, Alt, Esc等)

    # --- 调整角度逻辑 ---
    joint_updated = False
    if k in inc_keys:
        idx = inc_keys[k]
        new_angle = current_angles[idx] + STEP
        if new_angle > 1000: new_angle = 1000 # 限制最大值
        if new_angle != current_angles[idx]:
            current_angles[idx] = new_angle
            joint_updated = True
            print(f"增加: 自由度 {idx} -> {new_angle}")
    elif k in dec_keys:
        idx = dec_keys[k]
        new_angle = current_angles[idx] - STEP
        if new_angle < 0: new_angle = 0 # 限制最小值
        if new_angle != current_angles[idx]:
            current_angles[idx] = new_angle
            joint_updated = True
            print(f"减少: 自由度 {idx} -> {new_angle}")

    # 如果角度有更新，则发送到机械手
    if joint_updated:
         success = hand.set_angles(current_angles)
         if not success:
             print("错误：设置角度失败！")
         # print(f"当前角度: {current_angles}") # 可以在每次更新后打印当前所有角度



    # --- 保存位置逻辑 ---
    elif k in save_keys:
        position_key = k
        # 使用 list(current_angles) 创建一个副本，防止后续修改影响已保存的值
        saved_positions[position_key] = list(current_angles)
        print(f"位置 {position_key} 已保存: {saved_positions[position_key]}")
        save_positions_to_file() # 保存到文件

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 加载已保存的位置（如果存在）
    load_positions()

    # 2. 初始化并连接设备
    hand = SmartHandModbusTCP()
    print("正在尝试连接设备...")
    if not hand.connect():
         print("错误：无法连接到 SmartHand 设备。请检查连接。程序将退出。")
         exit() # 连接失败则退出
    print("设备连接成功！")


    # 3. 获取初始角度 (如果连接成功)
    initial_angles = hand.get_angles()
    if initial_angles and len(initial_angles) >= 6:
        current_angles = initial_angles[:6]
        print(f"成功获取初始角度: {current_angles}")
    else:
        print(f"警告：无法获取设备初始角度，将使用默认值: {current_angles}")
        # 可以选择将默认值写入设备
        # print("尝试将默认角度写入设备...")
        # hand.set_angles(current_angles)

    # 4. 启动键盘监听器
    print("\n--- 操作说明 ---")
    print("  调整角度:")
    print("    增加 (Q,W,E,R,T,Y -> 关节 5,4,3,2,1,0)")
    print("    减少 (A,S,D,F,G,H -> 关节 5,4,3,2,1,0)")
    print("  保存位置:")
    print("    按数字键 '1' 到 '6' 保存当前姿态")
    print("  退出: 按 'ESC' 键")
    print("-----------------\n")
    print("监听键盘输入...")

    # 使用 pynput 创建键盘监听器
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join() # 等待监听器结束 (直到 on_press 返回 False)
        except Exception as e:
            print(f"监听器发生错误: {e}")

    # 5. 清理资源
    print("程序结束，正在关闭设备连接...")
    hand.close()
    print("连接已关闭。")