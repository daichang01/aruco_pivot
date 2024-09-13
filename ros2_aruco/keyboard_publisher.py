import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import select
import termios
import tty

class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        self.publisher_ = self.create_publisher(String, 'keyboard_input', 10)
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.get_logger().info("Keyboard publisher node started. Press 'r' to trigger.")

    def publish_key(self):
        if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
            key = sys.stdin.read(1)
            msg = String()
            msg.data = key
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published key: {key}")

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()

    try:
        while rclpy.ok():
            node.publish_key()
    except KeyboardInterrupt:
        pass

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
