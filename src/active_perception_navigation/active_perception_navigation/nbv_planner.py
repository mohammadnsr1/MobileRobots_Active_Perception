import math
from dataclasses import dataclass
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


@dataclass
class CandidateView:
    pose: PoseStamped
    utility: float
    distance_to_robot: float
    heading_error: float


class NBVPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("ap_navigation_nbv_planner")

        self.declare_parameter("target_pose_topic", "/active_perception/target_pose")
        self.declare_parameter(
            "confidence_topic", "/active_perception/navigation/confidence"
        )
        self.declare_parameter(
            "nbv_required_topic", "/active_perception/navigation/nbv_required"
        )
        self.declare_parameter("robot_pose_topic", "/odom")
        self.declare_parameter("goal_topic", "/active_perception/navigation/nbv_goal")
        self.declare_parameter("status_topic", "/active_perception/navigation/nbv_status")
        self.declare_parameter("sampling_radius", 0.80)
        self.declare_parameter("num_candidates", 8)
        self.declare_parameter("preferred_standoff", 0.80)
        self.declare_parameter("min_confidence_to_stop", 0.80)
        self.declare_parameter("publish_only_when_required", True)

        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.confidence_topic = self.get_parameter("confidence_topic").value
        self.nbv_required_topic = self.get_parameter("nbv_required_topic").value
        self.robot_pose_topic = self.get_parameter("robot_pose_topic").value
        self.goal_topic = self.get_parameter("goal_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.sampling_radius = float(self.get_parameter("sampling_radius").value)
        self.num_candidates = int(self.get_parameter("num_candidates").value)
        self.preferred_standoff = float(self.get_parameter("preferred_standoff").value)
        self.min_confidence_to_stop = float(
            self.get_parameter("min_confidence_to_stop").value
        )
        self.publish_only_when_required = bool(
            self.get_parameter("publish_only_when_required").value
        )

        self.latest_confidence = 0.0
        self.latest_nbv_required = True
        self.robot_xy = np.array([0.0, 0.0], dtype=np.float64)
        self.robot_frame = "odom"

        self.target_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self.target_pose_callback, 10
        )
        self.confidence_sub = self.create_subscription(
            Float32, self.confidence_topic, self.confidence_callback, 10
        )
        self.nbv_required_sub = self.create_subscription(
            Bool, self.nbv_required_topic, self.nbv_required_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.robot_pose_topic, self.odom_callback, 10
        )

        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.get_logger().info(
            "Navigation NBV planner ready: target='%s', confidence='%s', goal='%s'"
            % (self.target_pose_topic, self.confidence_topic, self.goal_topic)
        )

    def normalize_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def yaw_to_quaternion(self, yaw: float):
        half_yaw = 0.5 * yaw
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def odom_callback(self, msg: Odometry) -> None:
        self.robot_xy = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float64
        )
        self.robot_frame = msg.header.frame_id or "odom"

    def confidence_callback(self, msg: Float32) -> None:
        self.latest_confidence = float(msg.data)

    def nbv_required_callback(self, msg: Bool) -> None:
        self.latest_nbv_required = bool(msg.data)

    def build_candidate(self, target_pose: PoseStamped, angle: float) -> CandidateView:
        target_xy = np.array(
            [target_pose.pose.position.x, target_pose.pose.position.y], dtype=np.float64
        )
        candidate_xy = target_xy + self.sampling_radius * np.array(
            [math.cos(angle), math.sin(angle)], dtype=np.float64
        )
        yaw_toward_target = math.atan2(
            target_xy[1] - candidate_xy[1], target_xy[0] - candidate_xy[0]
        )
        qx, qy, qz, qw = self.yaw_to_quaternion(yaw_toward_target)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.robot_frame
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(candidate_xy[0])
        pose_msg.pose.position.y = float(candidate_xy[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        distance_to_robot = float(np.linalg.norm(candidate_xy - self.robot_xy))
        standoff_error = abs(self.sampling_radius - self.preferred_standoff)
        heading_error = abs(self.normalize_angle(yaw_toward_target - angle))
        utility = (
            (-1.0 * distance_to_robot)
            - (0.4 * standoff_error)
            - (0.1 * heading_error)
        )
        return CandidateView(
            pose=pose_msg,
            utility=utility,
            distance_to_robot=distance_to_robot,
            heading_error=heading_error,
        )

    def select_best_candidate(self, target_pose: PoseStamped) -> CandidateView:
        candidates: List[CandidateView] = []
        for i in range(self.num_candidates):
            angle = (2.0 * math.pi * i) / max(self.num_candidates, 1)
            candidates.append(self.build_candidate(target_pose, angle))
        return max(candidates, key=lambda candidate: candidate.utility)

    def publish_status(self, message: str) -> None:
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        self.get_logger().info(message)

    def target_pose_callback(self, msg: PoseStamped) -> None:
        if (
            self.latest_confidence >= self.min_confidence_to_stop
            and not self.latest_nbv_required
        ):
            self.publish_status(
                f"Confidence {self.latest_confidence:.3f} is above threshold; no new NBV goal published."
            )
            return

        if self.publish_only_when_required and not self.latest_nbv_required:
            self.publish_status("NBV not required; planner is waiting.")
            return

        best_candidate = self.select_best_candidate(msg)
        self.goal_pub.publish(best_candidate.pose)
        self.publish_status(
            "Published NBV goal at (%.3f, %.3f) with utility %.3f; current confidence=%.3f"
            % (
                best_candidate.pose.pose.position.x,
                best_candidate.pose.pose.position.y,
                best_candidate.utility,
                self.latest_confidence,
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NBVPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
