from custom_interfaces.srv import TimeRetrevalMsg

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class TimeClientAsync(Node):

    def __init__(self):
        super().__init__('time_client_async')
        self.cli = self.create_client(TimeRetrevalMsg, 'global_time_server')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = TimeRetrevalMsg.Request()

    def send_request(self, zone):
        self.req.zone = zone
        return self.cli.call_async(self.req)


def main(args=None):
    try:
        with rclpy.init(args=args):
            time_client = TimeClientAsync()
            future = time_client.send_request("Asia/Tokyo")
            rclpy.spin_until_future_complete(time_client, future)
            response = future.result()
            time_client.get_logger().info(
                'Current time of the region: %s is %s' %
                (time_client.req.zone, response.time))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()