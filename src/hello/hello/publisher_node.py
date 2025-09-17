import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from datetime import datetime
from zoneinfo import ZoneInfo
from std_msgs.msg import String

class PublisherNode(Node):

    def __init__(self):
        super().__init__('time_publisher')
        self.publisher_ = self.create_publisher(String, 'tokyo_time', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        tokyo_time = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S")
        msg.data = tokyo_time
        self.publisher_.publish(msg)
        self.get_logger().info('I publish Tokyo time: "%s"' % msg.data)


def main(args=None):
    try:
        with rclpy.init(args=args):
            minimal_publisher = PublisherNode()

            rclpy.spin(minimal_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
