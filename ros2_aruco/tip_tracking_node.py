import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Empty
# import keyboard
import collections
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from datetime import datetime

class TipTrackingNode(Node):
    def __init__(self):
        super().__init__('tip_tracking_node')
        self.get_logger().info('Tip tracking node started')
        self.create_subscription(PointStamped, 'tool_tip_position', self.tip_callback, 10)
        # 创建srv client
        self.client = self.create_client(Empty, 'calibrate_tip')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.call_calibrate_pivot_service()
        # ros2 service call /calibrate_pivot std_srvs/srv/Empty
        # 创建一个队列来存储最近100个点
        self.tip_positions=  collections.deque(maxlen=100)
        # 监听键盘按键事件
        # keyboard.on_press_key("r", self.on_r_key_pressed)

        # 订阅键盘输入
        self.create_subscription(String, 'keyboard_input', self.keyboard_callback, 10)


    def tip_callback(self, msg):
        # 将针尖位置存入队列中
        self.tip_positions.append(msg.point)

        # self.get_logger().info(f"tool tip position: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")

    
    def save_positions_to_file(self):
            # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime('%m%d_%H%M%S')
        # 保存位置到 txt 文件中
        filename = f'/home/daichang/Desktop/teeth_ws/src/aruco_pivot/pin_cali_res/positions_{current_time}.txt'
        with open(filename, 'w') as file:
            for point in self.tip_positions:
                file.write(f"{point.x}, {point.y}, {point.z}\n")

        self.get_logger().info("Saved 100 positions to positions.txt.")

    def keyboard_callback(self, msg):
        # 当收到 'r' 键时，计算并打印最近100个针尖位置的平均值
        if msg.data == 'r':
            if len(self.tip_positions) < 100:
                self.get_logger().info(f"Not enough positions to calculate average. Current length: {len(self.tip_positions)}.")
                return

            
            avg_x = sum([point.x for point in self.tip_positions]) / len(self.tip_positions)
            avg_y = sum([point.y for point in self.tip_positions]) / len(self.tip_positions)
            avg_z = sum([point.z for point in self.tip_positions]) / len(self.tip_positions)

            self.get_logger().info(f"Recorded 100 positions. Average position: x={avg_x}, y={avg_y}, z={avg_z}")
            self.save_positions_to_file()
            self.tip_positions.clear()
            # self.get_logger().info(f"All positions: {[(p.x, p.y, p.z) for p in self.tip_positions]}")


    
    def call_calibrate_pivot_service(self):
        req = Empty.Request()
        self.future = self.client.call_async(req)
        self.future.add_done_callback(self.handle_service_response)
    
    def handle_service_response(self, future):
        try:
            response = future.result()
            self.get_logger().info('Pivot calibration service call succeeded')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TipTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
