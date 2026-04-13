#!/usr/bin/env python3

import copy
import math
from collections import deque
from enum import Enum, auto
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from active_perception.msg import PoseEstimateSample
from active_perception.srv import EvaluatePoseConfidence, PlanNBV


class OrchestratorState(Enum):
    IDLE = auto()
    WAITING_FOR_POSE = auto()
    EVALUATING = auto()
    PLANNING_NBV = auto()
    READY_TO_NAVIGATE = auto()
    DONE = auto()


class ActivePerceptionOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__('active_perception_orchestrator')

        self.declare_parameter('target_pose_topic', '/active_perception/target_pose')
        self.declare_parameter(
            'pose_sample_topic', '/active_perception/pose_estimate_sample'
        )
        self.declare_parameter('robot_pose_topic', '/odom')
        self.declare_parameter(
            'confidence_service_name', '/active_perception/evaluate_pose_confidence'
        )
        self.declare_parameter('nbv_service_name', '/active_perception/plan_nbv')
        self.declare_parameter('history_size', 10)
        self.declare_parameter('desired_confidence_threshold', 0.75)
        self.declare_parameter('min_history_length', 5)
        self.declare_parameter('nbv_num_candidates', 8)
        self.declare_parameter('nbv_radius', 0.8)
        self.declare_parameter('nbv_min_radius', 0.5)
        self.declare_parameter('nbv_max_radius', 1.2)
        self.declare_parameter('nbv_use_adaptive_radius', True)

        self.target_pose_topic = (
            self.get_parameter('target_pose_topic').get_parameter_value().string_value
        )
        self.pose_sample_topic = (
            self.get_parameter('pose_sample_topic').get_parameter_value().string_value
        )
        self.robot_pose_topic = (
            self.get_parameter('robot_pose_topic').get_parameter_value().string_value
        )
        self.confidence_service_name = (
            self.get_parameter('confidence_service_name')
            .get_parameter_value()
            .string_value
        )
        self.nbv_service_name = (
            self.get_parameter('nbv_service_name').get_parameter_value().string_value
        )
        self.history_size = (
            self.get_parameter('history_size').get_parameter_value().integer_value
        )
        self.desired_confidence_threshold = (
            self.get_parameter('desired_confidence_threshold')
            .get_parameter_value()
            .double_value
        )
        self.min_history_length = (
            self.get_parameter('min_history_length').get_parameter_value().integer_value
        )
        self.nbv_num_candidates = (
            self.get_parameter('nbv_num_candidates')
            .get_parameter_value()
            .integer_value
        )
        self.nbv_radius = (
            self.get_parameter('nbv_radius').get_parameter_value().double_value
        )
        self.nbv_min_radius = (
            self.get_parameter('nbv_min_radius').get_parameter_value().double_value
        )
        self.nbv_max_radius = (
            self.get_parameter('nbv_max_radius').get_parameter_value().double_value
        )
        self.nbv_use_adaptive_radius = (
            self.get_parameter('nbv_use_adaptive_radius')
            .get_parameter_value()
            .bool_value
        )

        self.target_pose_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self.target_pose_callback, 10
        )
        self.pose_sample_sub = self.create_subscription(
            PoseEstimateSample, self.pose_sample_topic, self.pose_sample_callback, 10
        )
        self.robot_pose_sub = self.create_subscription(
            Odometry, self.robot_pose_topic, self.robot_pose_callback, 10
        )

        self.confidence_client = self.create_client(
            EvaluatePoseConfidence, self.confidence_service_name
        )
        self.nbv_client = self.create_client(PlanNBV, self.nbv_service_name)

        self.state = OrchestratorState.WAITING_FOR_POSE
        self.history = deque(maxlen=max(int(self.history_size), 1))
        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_robot_pose: Optional[PoseStamped] = None
        self.previous_pose: Optional[PoseStamped] = None
        self.previous_score = 0.0
        self.previous_selected_nbv: Optional[PoseStamped] = None
        self.iteration_count = 0

        self.get_logger().info(
            "Orchestrator ready: target_pose='%s', pose_sample='%s', robot_pose='%s'"
            % (
                self.target_pose_topic,
                self.pose_sample_topic,
                self.robot_pose_topic,
            )
        )

    def quaternion_to_yaw(self, orientation) -> float:
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def target_pose_callback(self, msg: PoseStamped) -> None:
        self.latest_target_pose = copy.deepcopy(msg)

    def robot_pose_callback(self, msg: Odometry) -> None:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.latest_robot_pose = pose

    def pose_sample_to_pose_stamped(self, sample: PoseEstimateSample) -> PoseStamped:
        pose = PoseStamped()
        pose.header = sample.header
        pose.pose = sample.pose
        return pose

    def pose_sample_callback(self, msg: PoseEstimateSample) -> None:
        self.latest_target_pose = self.pose_sample_to_pose_stamped(msg)
        self.history.append(copy.deepcopy(msg))

        if self.state == OrchestratorState.DONE:
            return

        if self.state in (
            OrchestratorState.EVALUATING,
            OrchestratorState.PLANNING_NBV,
            OrchestratorState.READY_TO_NAVIGATE,
        ):
            return

        if self.latest_robot_pose is None:
            self.state = OrchestratorState.WAITING_FOR_POSE
            self.get_logger().info('Waiting for robot pose before evaluating target.')
            return

        self.start_confidence_evaluation()

    def start_confidence_evaluation(self) -> None:
        if self.latest_target_pose is None or not self.history:
            self.state = OrchestratorState.WAITING_FOR_POSE
            return

        if not self.confidence_client.wait_for_service(timeout_sec=0.1):
            self.state = OrchestratorState.WAITING_FOR_POSE
            self.get_logger().warn(
                "Confidence evaluator service '%s' is not available yet."
                % self.confidence_service_name
            )
            return

        request = EvaluatePoseConfidence.Request()
        request.history = list(self.history)
        request.desired_confidence_threshold = float(
            self.desired_confidence_threshold
        )
        request.min_history_length = int(self.min_history_length)

        self.state = OrchestratorState.EVALUATING
        future = self.confidence_client.call_async(request)
        future.add_done_callback(self.handle_confidence_response)

    def handle_confidence_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.state = OrchestratorState.WAITING_FOR_POSE
            self.get_logger().warn(
                'Confidence evaluation call failed: %s' % str(exc)
            )
            return

        if response is None or not response.success:
            self.state = OrchestratorState.WAITING_FOR_POSE
            diagnostic = 'no response'
            if response is not None:
                diagnostic = response.diagnostic_message
            self.get_logger().warn(
                'Confidence evaluation did not succeed: %s' % diagnostic
            )
            return

        self.previous_pose = copy.deepcopy(self.latest_target_pose)
        self.previous_score = float(response.confidence_score)

        self.get_logger().info(
            'Confidence score=%.3f stop=%s plan_nbv=%s | %s'
            % (
                response.confidence_score,
                response.should_stop,
                response.should_plan_nbv,
                response.diagnostic_message,
            )
        )

        if response.should_stop:
            self.state = OrchestratorState.DONE
            self.get_logger().info(
                'Active perception complete after %d planned iterations.'
                % self.iteration_count
            )
            return

        self.start_nbv_planning()

    def start_nbv_planning(self) -> None:
        if self.latest_target_pose is None or self.latest_robot_pose is None:
            self.state = OrchestratorState.WAITING_FOR_POSE
            return

        if not self.nbv_client.wait_for_service(timeout_sec=0.1):
            self.state = OrchestratorState.WAITING_FOR_POSE
            self.get_logger().warn(
                "NBV planner service '%s' is not available yet."
                % self.nbv_service_name
            )
            return

        request = PlanNBV.Request()
        request.target_pose = self.latest_target_pose
        request.robot_pose = self.latest_robot_pose
        request.num_candidates = int(self.nbv_num_candidates)
        request.radius = float(self.nbv_radius)
        request.min_radius = float(self.nbv_min_radius)
        request.max_radius = float(self.nbv_max_radius)
        request.use_adaptive_radius = bool(self.nbv_use_adaptive_radius)

        self.state = OrchestratorState.PLANNING_NBV
        future = self.nbv_client.call_async(request)
        future.add_done_callback(self.handle_nbv_response)

    def handle_nbv_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.state = OrchestratorState.WAITING_FOR_POSE
            self.get_logger().warn('NBV planning call failed: %s' % str(exc))
            return

        if response is None or not response.success:
            self.state = OrchestratorState.WAITING_FOR_POSE
            diagnostic = 'no response'
            if response is not None:
                diagnostic = response.diagnostic_message
            self.get_logger().warn('NBV planning did not succeed: %s' % diagnostic)
            return

        self.previous_selected_nbv = copy.deepcopy(response.best_view)
        self.iteration_count += 1
        self.state = OrchestratorState.READY_TO_NAVIGATE

        best_pose = response.best_view.pose
        best_yaw = self.quaternion_to_yaw(best_pose.orientation)
        self.get_logger().info(
            'READY_TO_NAVIGATE: iteration=%d selected view index=%d at '
            '(%.3f, %.3f, %.3f) yaw=%.3f rad'
            % (
                self.iteration_count,
                response.selected_index,
                best_pose.position.x,
                best_pose.position.y,
                best_pose.position.z,
                best_yaw,
            )
        )
        self.get_logger().info(
            'TODO: send response.best_view to Nav2 here, wait for navigation to '
            'finish, then return to WAITING_FOR_POSE for the next observation.'
        )

    # Later, when finder and pose_estimator become services, this node can move
    # from passive topic subscriptions to explicit service calls per iteration.
    # For now it stays compatible with the current topic-driven pipeline and only
    # orchestrates the history-based confidence check and the NBV planner call.


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActivePerceptionOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
