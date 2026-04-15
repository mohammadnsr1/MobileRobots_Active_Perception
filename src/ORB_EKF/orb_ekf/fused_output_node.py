#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node


class FusedOutputNode(Node):
    def __init__(self) -> None:
        super().__init__("orb_ekf_fused_output_node")

        self.declare_parameter("input_odom_topic", "/odometry/filtered")
        self.declare_parameter("output_path_topic", "/orb_ekf/fused_path")
        self.declare_parameter("output_pose_topic", "/orb_ekf/fused_pose")
        self.declare_parameter("max_path_length", 3000)

        self.path_pub = self.create_publisher(PathMsg, str(self.get_parameter("output_path_topic").value), 10)
        self.pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("output_pose_topic").value), 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("input_odom_topic").value),
            self.on_fused_odom,
            50,
        )

        self.path_msg = PathMsg()
        self.max_path_length = int(self.get_parameter("max_path_length").value)

        self.get_logger().info("Fused output node started.")

    def on_fused_odom(self, msg: Odometry) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.pose_pub.publish(pose)

        self.path_msg.header = msg.header
        self.path_msg.poses.append(pose)
        if len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_length :]
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
