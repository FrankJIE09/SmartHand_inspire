import time
from pynput import keyboard  # pip install pynput
from smart_hand_controller import SmartHandModbusTCP

# 全局变量：保存当前6个自由度的角度（列表长度为6）
current_angles = None

# 按键映射：按键对应的自由度索引（反向映射）
# 增加键映射：q, w, e, r, t, y 分别对应自由度 5, 4, 3, 2, 1, 0
inc_keys = {
    'q': 5,
    'w': 4,
    'e': 3,
    'r': 2,
    't': 1,
    'y': 0
}

# 减少键映射：a, s, d, f, g, h 分别对应自由度 5, 4, 3, 2, 1, 0
dec_keys = {
    'a': 5,
    's': 4,
    'd': 3,
    'f': 2,
    'g': 1,
    'h': 0
}

# 定义每次按键时变化的步长
step = 50

# 初始化并连接设备
hand = SmartHandModbusTCP()
hand.connect()

# 获取当前角度，确保列表长度为6；若读取失败则初始化为全0
angles = hand.get_angles()
if angles is None or len(angles) < 6:
    current_angles = [0, 0, 0, 0, 0, 0]
else:
    current_angles = angles[:6]
print("初始角度：", current_angles)

def on_press(key):
    global current_angles
    # 按下 ESC 键退出监听
    if key == keyboard.Key.esc:
        return False
    try:
        k = key.char.lower()  # 获取按键字符并转为小写
    except AttributeError:
        return  # 忽略非字符按键

    if k in inc_keys:
        idx = inc_keys[k]
        new_angle = current_angles[idx] + step
        if new_angle > 1000:
            new_angle = 1000
        current_angles[idx] = new_angle
        hand.set_angles(current_angles)
        print(f"增加：自由度 {idx} 的角度更新为 {new_angle}")
    elif k in dec_keys:
        idx = dec_keys[k]
        new_angle = current_angles[idx] - step
        if new_angle < 0:
            new_angle = 0
        current_angles[idx] = new_angle
        hand.set_angles(current_angles)
        print(f"减少：自由度 {idx} 的角度更新为 {new_angle}")

# 使用 pynput 创建键盘监听器
with keyboard.Listener(on_press=on_press) as listener:
    print("按键说明：")
    print("  增加：q, w, e, r, t, y 分别控制自由度 5,4,3,2,1,0 的角度增加")
    print("  减少：a, s, d, f, g, h 分别控制自由度 5,4,3,2,1,0 的角度减少")
    print("  按 ESC 键退出")
    listener.join()

hand.close()
