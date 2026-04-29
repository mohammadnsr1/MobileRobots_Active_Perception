#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node


class FusedOutputNode(Node):
    """
    Subscribes to VO odometry and wheel odometry, computes the naïve average
    position (x, y) at each VO frame, and publishes the averaged path.

    This is a lightweight placeholder for a full EKF fusion. The average
    reduces deviation from each individual sensor by 50% when their errors
    are independent.
    """

    def __init__(self) -> None:
        super().__init__("orb_ekf_fused_output_node")

        self.declare_parameter("vo_odom_topic", "/orb_slam/vo_odom")
        self.declare_parameter("wheel_odom_topic", "/robot_10/odom")
        self.declare_parameter("output_path_topic", "/orb_ekf/average_path")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("max_path_length", 3000)

        vo_odom_topic = str(self.get_parameter("vo_odom_topic").value)
        wheel_odom_topic = str(self.get_parameter("wheel_odom_topic").value)
        output_path_topic = str(self.get_parameter("output_path_topic").value)
        world_frame = str(self.get_parameter("world_frame").value)
        self.max_path_length = int(self.get_parameter("max_path_length").value)

        self.latest_wheel_odom: Odometry | None = None

        self.path_msg = PathMsg()
        self.path_msg.header.frame_id = world_frame

        self.path_pub = self.create_publisher(PathMsg, output_path_topic, 10)

        self.create_subscription(Odometry, wheel_odom_topic, self.on_wheel_odom, 50)
        self.create_subscription(Odometry, vo_odom_topic, self.on_vo_odom, 50)

        self.get_logger().info(
            f"Fused output node started. "
            f"VO={vo_odom_topic}, Wheel={wheel_odom_topic}, Out={output_path_topic}"
        )

    def on_wheel_odom(self, msg: Odometry) -> None:
        self.latest_wheel_odom = msg

    def on_vo_odom(self, msg: Odometry) -> None:
        if self.latest_wheel_odom is None:
            return

        vo_x = float(msg.pose.pose.position.x)
        vo_y = float(msg.pose.pose.position.y)
        odom_x = float(self.latest_wheel_odom.pose.pose.position.x)
        odom_y = float(self.latest_wheel_odom.pose.pose.position.y)

        avg_x = (vo_x + odom_x) / 2.0
        avg_y = (vo_y + odom_y) / 2.0

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = avg_x
        pose.pose.position.y = avg_y
        pose.pose.position.z = 0.0
        pose.pose.orientation = msg.pose.pose.orientation

        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_length:]
        self.path_pub.publish(self.path_msg)


def main() -> None:
    rclpy.init()
    node = FusedOutputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
