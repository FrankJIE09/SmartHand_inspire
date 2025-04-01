import time
import numpy as np
import matplotlib.pyplot as plt
from smart_hand_controller import SmartHandModbusTCP
from pymodbus.exceptions import ModbusException

# 定义每个手指 3 个触觉分段的信息（单位：字节）
# 注意：每两个字节构成一个寄存器
FINGER_SEGMENTS = {
    "finger1": {  # 小拇指
        "tip": {"start": 3000, "bytes": 18},  # 指端：18字节
        "mid": {"start": 3018, "bytes": 192},  # 指尖：192字节
        "base": {"start": 3210, "bytes": 160},  # 指腹：160字节
    },
    "finger2": {  # 无名指
        "tip": {"start": 3370, "bytes": 18},
        "mid": {"start": 3388, "bytes": 192},
        "base": {"start": 3580, "bytes": 160},
    },
    "finger3": {  # 中指
        "tip": {"start": 3740, "bytes": 18},
        "mid": {"start": 3758, "bytes": 192},
        "base": {"start": 3950, "bytes": 160},
    },
    "finger4": {  # 食指
        "tip": {"start": 4110, "bytes": 18},
        "mid": {"start": 4128, "bytes": 192},
        "base": {"start": 4320, "bytes": 160},
    },
    "finger5": {  # 大拇指（这里按照3段处理：指端、指尖、指腹）
        "tip": {"start": 4480, "bytes": 18},
        "mid": {"start": 4498, "bytes": 192},
        "base": {"start": 4708, "bytes": 192}  # 注意：大拇指指腹段字节数可能与其他手指不同
    }
}


def get_segment_avg(hand, start, byte_count):
    """
    根据指定起始地址和字节数，读取数据并计算平均值。
    每个寄存器2字节。
    """
    reg_count = byte_count // 2
    try:
        data = hand.read_registers(start, reg_count)
        if data is None or len(data) == 0:
            return 0
        return np.mean(data)
    except ModbusException as e:
        print(f"读取地址 {start} 失败: {e}")
        return 0


def display_finger_avg_tactile():
    """
    实时显示 5 个手指的触觉数据平均值，每个手指分为 3 段（指端、指尖、指腹）。
    最终以 5×3 的矩阵图形化显示（行：finger1~finger5，列：tip, mid, base）。
    """
    # 初始化设备
    hand = SmartHandModbusTCP()
    hand.connect()

    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Finger Tactile Average Values", fontsize=16)

        # 初始化 5×3 矩阵用于保存平均值
        avg_matrix = np.zeros((5, 3))
        im = ax.imshow(avg_matrix, cmap='plasma', aspect='auto')

        # 在每个单元格中添加文本显示平均值
        texts = []
        for i in range(5):
            row_texts = []
            for j in range(3):
                txt = ax.text(j, i, "0.0", ha="center", va="center", fontsize=12, color="white")
                row_texts.append(txt)
            texts.append(row_texts)

        # 设置坐标轴标签
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["Tip", "Mid", "Base"], fontsize=12)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(["Finger1", "Finger2", "Finger3", "Finger4", "Finger5"], fontsize=12)
        plt.colorbar(im, ax=ax)

        while True:
            # 对于每个手指，依次计算每段的平均值
            for i, finger in enumerate(["finger1", "finger2", "finger3", "finger4", "finger5"]):
                segments = FINGER_SEGMENTS[finger]
                avg_values = []
                for seg in ["tip", "mid", "base"]:
                    cfg = segments[seg]
                    avg = get_segment_avg(hand, cfg["start"], cfg["bytes"])
                    avg_values.append(avg)
                avg_matrix[i, :] = avg_values

            im.set_data(avg_matrix)
            im.set_clim(np.min(avg_matrix), np.max(avg_matrix))

            # 更新每个单元格的文本显示
            for i in range(5):
                for j in range(3):
                    texts[i][j].set_text(f"{avg_matrix[i, j]:.1f}")

            plt.draw()
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("退出实时显示")
    finally:
        hand.close()
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    display_finger_avg_tactile()
