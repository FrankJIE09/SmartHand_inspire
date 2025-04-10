import time
import json
import os
from pynput import keyboard
from smart_hand_controller import SmartHandModbusTCP

# --- 配置 ---
POSITION_FILE = "hand_positions.json"  # 保存位置的文件名

# --- 全局变量 ---
saved_positions = {} # 用于存储从文件加载的位置

# --- 函数 ---
def load_positions():
    """从文件加载已保存的位置"""
    global saved_positions
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, 'r') as f:
                saved_positions = json.load(f)
                print(f"已从 {POSITION_FILE} 加载 {len(saved_positions)} 个位置。")
                # 可以在这里打印加载的位置进行调试
                # for key, angles in saved_positions.items():
                #     print(f"  位置 {key}: {angles}")
                if not saved_positions:
                    print("警告：位置文件为空。")
        except json.JSONDecodeError:
            print(f"错误: {POSITION_FILE} 文件格式错误。请先运行保存程序创建有效文件。")
            return False # 加载失败
        except Exception as e:
            print(f"加载位置文件时出错: {e}")
            return False # 加载失败
    else:
        print(f"错误: 位置文件 {POSITION_FILE} 不存在。请先运行保存程序保存至少一个位置。")
        return False # 加载失败
    return True # 加载成功

def on_press(key):
    """处理按键事件"""
    global saved_positions

    # 按下 ESC 键退出监听
    if key == keyboard.Key.esc:
        print("收到退出信号...")
        return False # 停止监听器

    try:
        k = key.char.lower() # 获取按键字符并转为小写
    except AttributeError:
        return # 忽略非字符按键

    # --- 应用位置逻辑 ---
    if k in saved_positions: # 检查按键是否是已保存位置的键 ('1' 到 '6')
        angles_to_apply = saved_positions[k]
        print(f"正在应用位置 {k}: {angles_to_apply}")
        success = hand.set_angles(angles_to_apply)
        if success:
            print(f"位置 {k} 应用成功。")
        else:
            print(f"错误：应用位置 {k} 失败！")

    elif k in {'1', '2', '3', '4', '5', '6'}: # 如果按了数字键但未保存
        print(f"位置 {k} 尚未保存。请先使用 save_hand_positions.py 保存。")


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 加载保存的位置
    print("正在加载保存的位置...")
    if not load_positions():
        print("无法加载位置，程序将退出。")
        exit() # 如果无法加载位置文件，则退出

    # 2. 初始化并连接设备
    hand = SmartHandModbusTCP()
    print("正在尝试连接设备...")
    if not hand.connect():
        print("错误：无法连接到 SmartHand 设备。请检查连接。程序将退出。")
        exit()
    print("设备连接成功！")

    # 3. 启动键盘监听器
    print("\n--- 操作说明 ---")
    print("  应用位置:")
    print("    按数字键 '1' 到 '6' 应用对应的已保存姿态")
    print("  退出: 按 'ESC' 键")
    print("-----------------\n")
    print("监听键盘输入...")

    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except Exception as e:
            print(f"监听器发生错误: {e}")

    # 4. 清理资源
    print("程序结束，正在关闭设备连接...")
    hand.close()
    print("连接已关闭。")