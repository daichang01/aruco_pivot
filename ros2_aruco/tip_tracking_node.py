import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Empty

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


    def tip_callback(self, msg):
        self.get_logger().info(f"tool tip position: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")
        # 在这里处理针尖位置的数据，进行跟踪或其他操作
    
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
