from custom_interfaces.srv import TimeRetrevalMsg

from datetime import datetime
from zoneinfo import ZoneInfo

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class TimeServer(Node):

    def __init__(self):
        super().__init__('time_server')
        self.srv = self.create_service(TimeRetrevalMsg, 'global_time_server', self.get_time_from_zone)

    def get_time_from_zone(self, request, response):
        response.time = datetime.now(ZoneInfo(request.zone)).strftime("%Y-%m-%d %H:%M:%S")
        self.get_logger().info('Incoming request for zone: %s' % (request.zone))

        return response


def main():
    try:
        with rclpy.init():
            time_server = TimeServer()

            rclpy.spin(time_server)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()