from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String


class SafetyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("ap_navigation_safety_monitor")

        self.declare_parameter("robot_pose_topic", "/odom")
        self.declare_parameter(
            "safe_to_navigate_topic",
            "/active_perception/navigation/safe_to_navigate",
        )
        self.declare_parameter(
            "status_topic", "/active_perception/navigation/safety_status"
        )
        self.declare_parameter("odom_timeout_sec", 1.0)
        self.declare_parameter("check_period_sec", 0.25)

        self.robot_pose_topic = self.get_parameter("robot_pose_topic").value
        self.safe_to_navigate_topic = self.get_parameter("safe_to_navigate_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.check_period_sec = float(self.get_parameter("check_period_sec").value)

        self.last_odom_time: Optional[Time] = None
        self.last_safe_state: Optional[bool] = None

        self.odom_sub = self.create_subscription(
            Odometry, self.robot_pose_topic, self.odom_callback, 10
        )
        self.safe_pub = self.create_publisher(Bool, self.safe_to_navigate_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.timer = self.create_timer(self.check_period_sec, self.check_safety)

        self.get_logger().info(
            "Safety monitor ready: odom='%s', safe='%s', timeout=%.2fs"
            % (
                self.robot_pose_topic,
                self.safe_to_navigate_topic,
                self.odom_timeout_sec,
            )
        )

    def odom_callback(self, _: Odometry) -> None:
        self.last_odom_time = self.get_clock().now()

    def publish_state(self, safe: bool, message: str) -> None:
        safe_msg = Bool()
        safe_msg.data = safe
        self.safe_pub.publish(safe_msg)

        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)

        if self.last_safe_state != safe:
            self.get_logger().info(message)
        self.last_safe_state = safe

    def check_safety(self) -> None:
        now = self.get_clock().now()

        if self.last_odom_time is None:
            self.publish_state(False, "SAFE_STOP: waiting for odometry.")
            return

        odom_age = (now - self.last_odom_time).nanoseconds / 1e9
        if odom_age > self.odom_timeout_sec:
            self.publish_state(
                False,
                "SAFE_STOP: odometry is stale (age=%.2fs, limit=%.2fs)."
                % (odom_age, self.odom_timeout_sec),
            )
            return

        self.publish_state(True, "RUN: odometry heartbeat is healthy.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
