import rclpy
from rclpy.node import Node

class HelloClassNode(Node):
    def __init__(self):
        super().__init__('hello_class_node')
        self.get_logger().info('Hello Class')

def main(args=None):
    rclpy.init(args=args)
    node = HelloClassNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()