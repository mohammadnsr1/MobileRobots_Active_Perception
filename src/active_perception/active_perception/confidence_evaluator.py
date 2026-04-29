#!/usr/bin/env python3

import math
from typing import Iterable

import numpy as np

import rclpy
from rclpy.node import Node

from active_perception_interfaces.msg import PoseEstimateSample
from active_perception_interfaces.srv import EvaluatePoseConfidence


class ConfidenceEvaluatorNode(Node):
    POSITION_VARIANCE_NORM = 0.005
    YAW_VARIANCE_NORM = 0.08
    POINT_COUNT_NORM = 200.0
    ANISOTROPY_NORM = 0.5

    def __init__(self) -> None:
        super().__init__('confidence_evaluator')

        self.declare_parameter(
            'service_name', '/robot_10/active_perception/evaluate_pose_confidence'
        )
        self.declare_parameter('weight_position', 0.3)
        self.declare_parameter('weight_yaw', 0.2)
        self.declare_parameter('weight_point_count', 0.1)
        self.declare_parameter('weight_anisotropy', 0.15)

        self.service_name = (
            self.get_parameter('service_name').get_parameter_value().string_value
        )
        self.weight_position = self.get_parameter('weight_position').value
        self.weight_yaw = self.get_parameter('weight_yaw').value
        self.weight_point_count = self.get_parameter('weight_point_count').value
        self.weight_anisotropy = self.get_parameter('weight_anisotropy').value

        self.service = self.create_service(
            EvaluatePoseConfidence,
            self.service_name,
            self.evaluate_pose_confidence_callback,
        )

        self.get_logger().info(
            "Confidence evaluator ready on '%s'" % self.service_name
        )

    def quaternion_to_yaw(self, orientation) -> float:
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute_position_variance(
        self, samples: Iterable[PoseEstimateSample]
    ) -> float:
        xy = np.array(
            [
                [float(sample.pose.position.x), float(sample.pose.position.y)]
                for sample in samples
            ],
            dtype=np.float64,
        )
        if len(xy) == 0:
            return 0.0

        mean_xy = np.mean(xy, axis=0)
        deltas = xy - mean_xy
        squared_distance = np.sum(deltas * deltas, axis=1)
        return float(np.mean(squared_distance))

    def compute_yaw_variance(
        self, samples: Iterable[PoseEstimateSample]
    ) -> float:
        yaws = np.array(
            [self.quaternion_to_yaw(sample.pose.orientation) for sample in samples],
            dtype=np.float64,
        )
        if len(yaws) == 0:
            return 0.0

        mean_sin = float(np.mean(np.sin(yaws)))
        mean_cos = float(np.mean(np.cos(yaws)))
        mean_yaw = math.atan2(mean_sin, mean_cos)
        yaw_errors = np.array(
            [self.wrap_angle(float(yaw - mean_yaw)) for yaw in yaws], dtype=np.float64
        )
        return float(np.mean(yaw_errors * yaw_errors))

    def evaluate_pose_confidence_callback(
        self,
        request: EvaluatePoseConfidence.Request,
        response: EvaluatePoseConfidence.Response,
    ) -> EvaluatePoseConfidence.Response:
        history = list(request.history)
        history_count = len(history)

        response.should_stop = False
        response.should_plan_nbv = True
        response.success = False
        response.confidence_score = 0.0
        response.position_variance = 0.0
        response.yaw_variance = 0.0
        response.mean_point_count = 0.0
        response.mean_anisotropy_ratio = 0.0

        if history_count == 0:
            response.diagnostic_message = 'No pose history provided.'
            return response

        response.position_variance = self.compute_position_variance(history)
        response.yaw_variance = self.compute_yaw_variance(history)

        point_counts = np.array(
            [float(sample.point_count) for sample in history], dtype=np.float64
        )
        anisotropy = np.array(
            [float(sample.anisotropy_ratio) for sample in history], dtype=np.float64
        )

        response.mean_point_count = float(np.mean(point_counts))
        response.mean_anisotropy_ratio = float(np.mean(anisotropy))

        position_stability = 1.0 / (
            1.0 + response.position_variance / max(self.POSITION_VARIANCE_NORM, 1e-9)
        )
        yaw_stability = 1.0 / (
            1.0 + response.yaw_variance / max(self.YAW_VARIANCE_NORM, 1e-9)
        )
        point_score = float(
            np.clip(response.mean_point_count / max(self.POINT_COUNT_NORM, 1e-9), 0.0, 1.0)
        )
        anisotropy_score = float(
            np.clip(
                response.mean_anisotropy_ratio / max(self.ANISOTROPY_NORM, 1e-9),
                0.0,
                1.0,
            )
        )
        self.get_logger().info(
            f'Components -> pos:{position_stability:.3f}, yaw:{yaw_stability:.3f}, '
            f'points:{point_score:.3f}, anisotropy:{anisotropy_score:.3f}'
        )
        response.confidence_score = (
            self.weight_position * position_stability
            + self.weight_yaw * yaw_stability
            + self.weight_point_count * point_score
            + self.weight_anisotropy * anisotropy_score
        )
        response.success = True

        min_history_length = max(int(request.min_history_length), 1)
        threshold = float(request.desired_confidence_threshold)
        response.should_stop = (
            history_count >= min_history_length
            and response.confidence_score >= threshold
        )
        response.should_plan_nbv = not response.should_stop

        if history_count < min_history_length:
            stop_reason = (
                'provisional score only, history shorter than required window'
            )
        elif response.should_stop:
            stop_reason = 'confidence threshold reached'
        else:
            stop_reason = 'confidence below threshold, continue active perception'

        response.diagnostic_message = (
            'history=%d score=%.3f threshold=%.3f pos_var=%.4f yaw_var=%.4f '
            'mean_points=%.1f mean_anisotropy=%.3f: %s'
            % (
                history_count,
                response.confidence_score,
                threshold,
                response.position_variance,
                response.yaw_variance,
                response.mean_point_count,
                response.mean_anisotropy_ratio,
                stop_reason,
            )
        )
        return response


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


if __name__ == '__main__':
    main()
