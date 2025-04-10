import time
import json
import os
import datetime # Import datetime module
from pynput import keyboard
try:
    from smart_hand_controller import SmartHandModbusTCP
except ImportError:
    print("错误：无法导入 SmartHandModbusTCP。")
    print("请确保 smart_hand_controller.py 文件在当前目录或 Python 路径中。")
    exit()

# --- 配置 ---
POSITION_FILE = "hand_positions.json"  # 主保存文件，用于加载和 F 键保存
SNAPSHOT_FILENAME_PREFIX = "hand_positions_" # 时间戳快照文件名前缀
STEP = 50

# --- 全局变量 ---
current_angles = [0, 0, 0, 0, 0, 0]
saved_positions = {}
hand = None
ctrl_pressed = False # Track Ctrl key state

# --- 按键映射 ---
inc_keys = {'q': 5, 'w': 4, 'e': 3, 'r': 2, 't': 1, 'y': 0}
dec_keys = {'a': 5, 's': 4, 'd': 3, 'f': 2, 'g': 1, 'h': 0}
fkey_to_save_slot = {
    keyboard.Key.f1: '1', keyboard.Key.f2: '2', keyboard.Key.f3: '3',
    keyboard.Key.f4: '4', keyboard.Key.f5: '5', keyboard.Key.f6: '6',
}
apply_keys_chars = {'1', '2', '3', '4', '5', '6'}

# --- 函数 ---
def load_positions():
    """从主文件加载位置"""
    global saved_positions
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, 'r') as f:
                loaded_data = json.load(f)
                # Basic validation
                valid_positions = {}
                if isinstance(loaded_data, dict):
                    for key, value in loaded_data.items():
                        if isinstance(key, str) and key.isdigit() and isinstance(value, list) and len(value) == 6:
                            valid_positions[key] = value
                        else:
                            print(f"警告：加载时忽略主文件中无效的位置条目：'{key}': {value}")
                    saved_positions = valid_positions
                    print(f"已从 {POSITION_FILE} 加载 {len(saved_positions)} 个有效位置。")
                else:
                     print(f"警告: {POSITION_FILE} 文件内容不是预期的字典格式，将使用空的位置列表。")
                     saved_positions = {}

        except json.JSONDecodeError:
            print(f"警告: {POSITION_FILE} 文件格式错误或为空，将使用空的位置列表。")
            saved_positions = {}
        except Exception as e:
            print(f"加载主位置文件时出错: {e}")
            saved_positions = {}
    else:
        print(f"主位置文件 {POSITION_FILE} 不存在，将创建新的文件（通过 F1-F6 保存时）。")
        saved_positions = {}

def save_positions_to_main_file():
    """将当前保存的位置写入主文件 (供下次加载)"""
    global saved_positions
    try:
        with open(POSITION_FILE, 'w') as f:
            json.dump(saved_positions, f, indent=4)
        # print(f"主位置文件 {POSITION_FILE} 已更新。") # 可以取消注释以获取更详细的日志
    except Exception as e:
        print(f"保存主位置文件 {POSITION_FILE} 时出错: {e}")

def save_snapshot():
    """保存当前所有已定义姿态到带时间戳的文件"""
    global saved_positions
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_FILENAME_PREFIX}{timestamp}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(saved_positions, f, indent=4)
        print(f"\n[快照] 当前 {len(saved_positions)} 个已存姿态已另存为: {filename}")
    except Exception as e:
        print(f"\n[错误] 保存快照文件 {filename} 失败: {e}")

def apply_saved_pose(slot_key):
    """应用指定槽位的姿态"""
    global current_angles, hand, saved_positions
    if slot_key in saved_positions:
        angles_to_apply = saved_positions[slot_key]
        print(f"\n正在应用已保存的姿态 {slot_key}: {angles_to_apply}")
        success = hand.set_angles(angles_to_apply)
        if success:
            current_angles = list(angles_to_apply)
            print(f"姿态 {slot_key} 应用成功。")
        else:
             print("错误：设备未连接或未初始化，无法应用姿态。")
    else:
        print(f"\n错误：姿态 {slot_key} 尚未保存在内存中 (请使用 F{slot_key} 键保存)。")

# --- 键盘事件处理 ---

def on_press(key):
    """处理按键按下事件"""
    global current_angles, saved_positions, hand, ctrl_pressed

    # 1. 检查是否按下 Ctrl 键
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = True
        return # 仅记录状态，不执行其他操作

    # 2. 检查退出键
    if key == keyboard.Key.esc:
        print("\n收到退出信号...")
        return False # 停止监听器

    # 3. 检查 F 键 (保存到主文件并更新内存)
    if key in fkey_to_save_slot:
        slot_key = fkey_to_save_slot[key]
        saved_positions[slot_key] = list(current_angles)
        print(f"\n姿态已保存到内存槽位 {slot_key}: {saved_positions[slot_key]}")
        save_positions_to_main_file() # 更新主文件
        return # 处理完毕

    # --- 尝试获取字符键 ---
    try:
        # 特别注意：对于 Ctrl+S，key 可能没有 char 属性，
        # 或者 char 可能是特殊控制字符，如 \x13。
        # 我们优先使用 ctrl_pressed 标志来判断。
        k = None
        if hasattr(key, 'char'):
             k = key.char.lower()
        # print(f"Debug: Key={key}, Char='{k}', Ctrl={ctrl_pressed}") # 用于调试

    except AttributeError:
         # 非字符键（除了上面处理的 F 键、Ctrl、ESC）被忽略
        return

    # 4. 检查 Ctrl + S (保存快照)
    # 检查 k 是否为 's' 并且 ctrl 键是按下的状态
    if k == 's' and ctrl_pressed:
        save_snapshot()
        return # 处理完毕，防止 's' 触发角度调整

    # 5. 检查数字键 (应用姿态)
    if k in apply_keys_chars:
        apply_saved_pose(k)
        return # 处理完毕

    # 6. 检查调整角度键 (确保 Ctrl 未按下时才调整)
    if not ctrl_pressed:
        joint_updated = False
        new_angle = -1

        if k in inc_keys:
            idx = inc_keys[k]
            if current_angles[idx] < 1000:
                new_angle = min(current_angles[idx] + STEP, 1000)
                current_angles[idx] = new_angle
                joint_updated = True
                print(f"\n增加: 自由度 {idx} -> {new_angle}")
        elif k in dec_keys:
            # 's' 键也在这里，但因为上面的 Ctrl+S 检查有 return，
            # 所以只有在 Ctrl 未按下时才会执行这里的逻辑
            idx = dec_keys[k]
            if current_angles[idx] > 0:
                new_angle = max(current_angles[idx] - STEP, 0)
                current_angles[idx] = new_angle
                joint_updated = True
                print(f"\n减少: 自由度 {idx} -> {new_angle}")

        if joint_updated:
             success = hand.set_angles(current_angles)
             if not success:
                 print("错误：设置角度失败！")
             # print(f"当前角度: {current_angles}") # 可以取消注释


def on_release(key):
    """处理按键释放事件"""
    global ctrl_pressed
    # 检查是否释放了 Ctrl 键
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False
        # print("Ctrl released") # 用于调试

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 加载主位置文件
    load_positions()

    # 2. 初始化并连接设备
    hand = SmartHandModbusTCP()
    print("正在尝试连接设备...")
    if not hand.connect():
         print("错误：无法连接到 SmartHand 设备。请检查连接。程序将退出。")
         exit()
    print("设备连接成功！")

    # 3. 获取/设置初始角度
    initial_angles = hand.get_angles()
    if initial_angles and len(initial_angles) >= 6:
        current_angles = initial_angles[:6]
        print(f"成功获取初始角度: {current_angles}")
    else:
        print(f"警告：无法获取设备初始角度，使用默认值: {current_angles}")
        print("尝试将默认角度写入设备...")
        if not hand.set_angles(current_angles):
            print("警告：设置初始默认角度失败。")

    # 4. 启动键盘监听器 (包含 on_release)
    print("\n--- 操作说明 ---")
    print("  调整角度: QWERTY (增) / ASDFGH (减) -> 关节 543210")
    print("  保存姿态 (更新主文件 & 内存): F1 到 F6 -> 槽位 1 到 6")
    print("  应用姿态 (从内存加载): 数字键 1 到 6 -> 对应槽位")
    print("  保存快照 (所有已存姿态另存为文件): Ctrl + S")
    print("  退出: ESC")
    print("-----------------\n")
    print(f"当前角度: {current_angles}")
    print("监听键盘输入...")

    # 传入 on_press 和 on_release
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except Exception as e:
            print(f"\n监听器发生错误: {e}")

    # 5. 清理资源
    print("\n程序结束，正在关闭设备连接...")
    if hand:
        hand.close()
    print("连接已关闭。")