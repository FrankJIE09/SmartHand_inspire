from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
import numpy as np


class SmartHandModbusTCP:
    FINGER_SEGMENTS_INFO = {
        1: [  # finger1 (小拇指)
            {"name": "指端", "start": 3000, "bytes": 18, "shape": (3, 3)},  # 3*3=9寄存器=18字节
            {"name": "指尖", "start": 3018, "bytes": 192, "shape": (12, 8)},  # 12*8=96寄存器=192字节
            {"name": "指腹", "start": 3210, "bytes": 160, "shape": (10, 8)}  # 10*8=80寄存器=160字节
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
        """
        初始化 Modbus TCP 客户端
        :param ip: 灵巧手 IP 地址（默认 192.168.11.210）
        :param port: 端口号（默认 6000）
        :param unit_id: 单元标识符，通常为设备的从站地址
        """
        self.client = ModbusTcpClient(ip, port=port)
        self.unit_id = unit_id

    def connect(self):
        """建立与设备的连接"""
        if not self.client.connect():
            raise ConnectionError("无法连接到设备")
        return True

    def close(self):
        """关闭与设备的连接"""
        self.client.close()

    def read_registers(self, address, count):
        """
        读取保持寄存器的值
        :param address: 起始寄存器地址
        :param count: 要读取的寄存器数量
        :return: 整数列表，每个整数对应一个寄存器（16位，无符号，低位在前）
        """
        try:
            result = self.client.read_holding_registers(address=address, count=count, slave=self.unit_id)
            if result.isError():
                raise ModbusException(f"读取寄存器失败: {result}")
            # 将返回的列表转换为16位无符号整数（little-endian格式）
            # 这里使用 dtype='<u2' 强制指定小端无符号16位
            data = np.array(result.registers, dtype='<u2')
            return data.tolist()
        except ModbusException as e:
            print(f"Modbus 异常: {e}")
            return None

    def write_single_register(self, address, value):
        """
        写入单个寄存器
        :param address: 寄存器地址
        :param value: 要写入的值
        """
        try:
            result = self.client.write_register(address=address, value=value, slave=self.unit_id)
            if result.isError():
                raise ModbusException(f"写入寄存器失败: {result}")
        except ModbusException as e:
            print(f"Modbus 异常: {e}")

    def write_multiple_registers(self, address, values):
        """
        写入多个寄存器
        :param address: 起始寄存器地址
        :param values: 要写入的值列表
        """
        try:
            result = self.client.write_registers(address=address, values=values, slave=self.unit_id)
            if result.isError():
                raise ModbusException(f"写入多个寄存器失败: {result}")
        except ModbusException as e:
            print(f"Modbus 异常: {e}")

    def save_config(self):
        """保存当前配置到设备"""
        self.write_single_register(1005, 1)

    def clear_error(self):
        """清除设备故障状态"""
        self.write_single_register(1004, 1)

    def set_gesture(self, gesture_id):
        """
        设置目标手势
        :param gesture_id: 手势编号（1~40）
        """
        self.write_single_register(1575, gesture_id)

    def get_angles(self):
        """
        获取当前各自由度的角度
        :return: 角度列表（长度6）
        """
        return self.read_registers(1546, 6)

    def set_angles(self, angle_list):
        """
        设置各自由度的目标角度
        :param angle_list: 包含6个角度值的列表，范围为 0~1000 或 -1
        """
        if len(angle_list) != 6:
            raise ValueError("需要提供6个角度值")
        self.write_multiple_registers(1486, angle_list)
        return True


    def get_tactile_data(self, sensor):
        """
        读取触觉传感器数据
        :param sensor: 触觉传感器名称，可选值为：'finger1', 'finger2', 'finger3', 'finger4', 'finger5', 'palm'
        :return: 触觉数据列表，每个寄存器对应一个数据点（2字节，低位在前）
        """
        # 根据用户手册，各传感器数据长度（字节）如下：
        # finger1: 小拇指触觉 370 byte, finger2: 无名指 370 byte,
        # finger3: 中指 370 byte, finger4: 食指 370 byte,
        # finger5: 大拇指 420 byte, palm: 手掌 224 byte
        tactile_info = {
            "finger1": {"address": 3000, "bytes": 370},
            "finger2": {"address": 3370, "bytes": 370},
            "finger3": {"address": 3740, "bytes": 370},
            "finger4": {"address": 4110, "bytes": 370},
            "finger5": {"address": 4480, "bytes": 420},
            "palm": {"address": 4900, "bytes": 224},
        }
        if sensor not in tactile_info:
            raise ValueError(
                "无效的触觉传感器名称，请选择 'finger1', 'finger2', 'finger3', 'finger4', 'finger5' 或 'palm'")
        info = tactile_info[sensor]
        # 每个寄存器 2 字节，计算需要读取的寄存器数
        reg_count = info["bytes"] // 2
        return self.read_registers(info["address"], reg_count)

    def get_finger_segments(self, finger_num: int):
        """
        读取指定手指的触觉数据，并按“指端 / 指尖 / 指腹”三段（或拇指四段）分段返回。
        :param finger_num: 手指编号 (1=小拇指,2=无名指,3=中指,4=食指,5=大拇指)
        :return: 一个列表，每段是一个字典，如:
            [
              {
                "name": "指端",
                "data": 二维numpy数组 (shape参照手册)
              },
              {
                "name": "指尖",
                "data": 二维numpy数组
              },
              ...
            ]
        """
        if finger_num not in self.FINGER_SEGMENTS_INFO:
            raise ValueError("无效的手指编号，必须是1~5")

        segments_config = self.FINGER_SEGMENTS_INFO[finger_num]
        result = []

        for seg_cfg in segments_config:
            start_addr = seg_cfg["start"]
            total_bytes = seg_cfg["bytes"]
            shape = seg_cfg["shape"]
            # 每个寄存器2字节
            reg_count = total_bytes // 2

            # 读取原始数据
            raw_data = self.read_registers(start_addr, reg_count)
            if raw_data is None:
                # 读取失败，返回空
                arr_2d = np.zeros(shape, dtype=int)
            else:
                arr_2d = np.array(raw_data, dtype=int).reshape(shape)

            result.append({
                "name": seg_cfg["name"],
                "data": arr_2d
            })

        return result


if __name__ == '__main__':
    if __name__ == '__main__':
        hand = SmartHandModbusTCP()
        try:
            hand.connect()

            # 读取小拇指(1号)三段触觉
            pinky_segments = hand.get_finger_segments(1)
            for seg in pinky_segments:
                print(f"段名: {seg['name']}, 形状: {seg['data'].shape}")
                print(seg['data'])  # 这里打印二维数组

            # 读取大拇指(5号)四段触觉
            thumb_segments = hand.get_finger_segments(5)
            for seg in thumb_segments:
                print(f"段名: {seg['name']}, 形状: {seg['data'].shape}")
                # 如果要看具体数据，可打印 seg['data']

        finally:
            hand.close()
