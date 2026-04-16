import math
from collections import deque
from typing import Deque, List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class ConfidenceEvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__("ap_navigation_confidence_evaluator")

        self.declare_parameter("input_pose_topic", "/active_perception/target_pose")
        self.declare_parameter(
            "output_confidence_topic", "/active_perception/navigation/confidence"
        )
        self.declare_parameter(
            "output_nbv_topic", "/active_perception/navigation/nbv_required"
        )
        self.declare_parameter(
            "status_topic", "/active_perception/navigation/confidence_status"
        )
        self.declare_parameter("window_size", 8)
        self.declare_parameter("confidence_threshold", 0.80)
        self.declare_parameter("position_variance_threshold", 0.010)
        self.declare_parameter("yaw_variance_threshold", 0.050)
        self.declare_parameter("publish_debug_logs", True)

        self.input_pose_topic = self.get_parameter("input_pose_topic").value
        self.output_confidence_topic = self.get_parameter("output_confidence_topic").value
        self.output_nbv_topic = self.get_parameter("output_nbv_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.window_size = int(self.get_parameter("window_size").value)
        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.position_variance_threshold = float(
            self.get_parameter("position_variance_threshold").value
        )
        self.yaw_variance_threshold = float(
            self.get_parameter("yaw_variance_threshold").value
        )
        self.publish_debug_logs = bool(self.get_parameter("publish_debug_logs").value)

        self.pose_window: Deque[PoseStamped] = deque(maxlen=self.window_size)

        self.pose_sub = self.create_subscription(
            PoseStamped, self.input_pose_topic, self.pose_callback, 10
        )
        self.confidence_pub = self.create_publisher(
            Float32, self.output_confidence_topic, 10
        )
        self.nbv_pub = self.create_publisher(Bool, self.output_nbv_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.get_logger().info(
            "Navigation confidence evaluator ready: pose='%s', confidence='%s', nbv='%s'"
            % (
                self.input_pose_topic,
                self.output_confidence_topic,
                self.output_nbv_topic,
            )
        )

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def circular_variance(self, angles: np.ndarray) -> float:
        if len(angles) == 0:
            return float("inf")
        mean_cos = np.mean(np.cos(angles))
        mean_sin = np.mean(np.sin(angles))
        r = math.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
        return max(0.0, 1.0 - r)

    def compute_confidence(self, poses: List[PoseStamped]) -> float:
        if len(poses) < 2:
            return 0.0

        positions = np.array(
            [
                [
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ]
                for pose in poses
            ],
            dtype=np.float64,
        )
        yaws = np.array(
            [
                self.quaternion_to_yaw(
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                )
                for pose in poses
            ],
            dtype=np.float64,
        )

        position_variance = float(np.mean(np.var(positions, axis=0)))
        yaw_variance = float(self.circular_variance(yaws))

        position_score = max(
            0.0,
            1.0 - position_variance / max(self.position_variance_threshold, 1e-9),
        )
        yaw_score = max(
            0.0,
            1.0 - yaw_variance / max(self.yaw_variance_threshold, 1e-9),
        )

        confidence = 0.7 * position_score + 0.3 * yaw_score
        return float(np.clip(confidence, 0.0, 1.0))

    def publish_outputs(self, confidence: float) -> None:
        nbv_required = confidence < self.confidence_threshold

        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)

        nbv_msg = Bool()
        nbv_msg.data = nbv_required
        self.nbv_pub.publish(nbv_msg)

        status_msg = String()
        status_msg.data = (
            f"confidence={confidence:.3f}; nbv_required={str(nbv_required).lower()}; "
            f"window={len(self.pose_window)}/{self.window_size}"
        )
        self.status_pub.publish(status_msg)

        if self.publish_debug_logs:
            self.get_logger().info(status_msg.data)

    def pose_callback(self, msg: PoseStamped) -> None:
        self.pose_window.append(msg)
        confidence = self.compute_confidence(list(self.pose_window))
        self.publish_outputs(confidence)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfidenceEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
