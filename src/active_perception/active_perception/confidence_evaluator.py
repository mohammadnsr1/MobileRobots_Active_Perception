#!/usr/bin/env python3

import math
from typing import Iterable, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from active_perception_interfaces.msg import PoseEstimateSample
from active_perception_interfaces.srv import EvaluatePoseConfidence


class ConfidenceEvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__('confidence_evaluator')

        self.declare_parameter(
            'service_name', '/robot_10/active_perception/evaluate_pose_confidence'
        )
        self.declare_parameter('weight_position_stability', 0.35)
        self.declare_parameter('weight_yaw_stability', 0.25)
        self.declare_parameter('weight_point_count', 0.15)
        self.declare_parameter('weight_anisotropy', 0.15)
        self.declare_parameter('weight_pca_yaw_fraction', 0.10)
        self.declare_parameter('position_variance_norm', 0.01)
        self.declare_parameter('yaw_variance_norm', 0.08)
        self.declare_parameter('point_count_norm', 200.0)
        self.declare_parameter('anisotropy_norm', 0.5)
        self.declare_parameter('use_recency_weighting', True)

        self.service_name = (
            self.get_parameter('service_name').get_parameter_value().string_value
        )
        self.weight_position_stability = (
            self.get_parameter('weight_position_stability')
            .get_parameter_value()
            .double_value
        )
        self.weight_yaw_stability = (
            self.get_parameter('weight_yaw_stability')
            .get_parameter_value()
            .double_value
        )
        self.weight_point_count = (
            self.get_parameter('weight_point_count').get_parameter_value().double_value
        )
        self.weight_anisotropy = (
            self.get_parameter('weight_anisotropy').get_parameter_value().double_value
        )
        self.weight_pca_yaw_fraction = (
            self.get_parameter('weight_pca_yaw_fraction')
            .get_parameter_value()
            .double_value
        )
        self.position_variance_norm = (
            self.get_parameter('position_variance_norm')
            .get_parameter_value()
            .double_value
        )
        self.yaw_variance_norm = (
            self.get_parameter('yaw_variance_norm').get_parameter_value().double_value
        )
        self.point_count_norm = (
            self.get_parameter('point_count_norm').get_parameter_value().double_value
        )
        self.anisotropy_norm = (
            self.get_parameter('anisotropy_norm').get_parameter_value().double_value
        )
        self.use_recency_weighting = (
            self.get_parameter('use_recency_weighting')
            .get_parameter_value()
            .bool_value
        )

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

    def build_weights(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.float64)

        if self.use_recency_weighting:
            weights = np.arange(1, count + 1, dtype=np.float64)
        else:
            weights = np.ones(count, dtype=np.float64)

        weights_sum = np.sum(weights)
        if weights_sum <= 1e-12:
            return np.ones(count, dtype=np.float64) / float(count)
        return weights / weights_sum

    def compute_position_variance(
        self, samples: Iterable[PoseEstimateSample], weights: np.ndarray
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

        mean_xy = np.average(xy, axis=0, weights=weights)
        deltas = xy - mean_xy
        squared_distance = np.sum(deltas * deltas, axis=1)
        return float(np.average(squared_distance, weights=weights))

    def compute_yaw_variance(
        self, samples: Iterable[PoseEstimateSample], weights: np.ndarray
    ) -> Tuple[float, float]:
        yaws = np.array(
            [self.quaternion_to_yaw(sample.pose.orientation) for sample in samples],
            dtype=np.float64,
        )
        if len(yaws) == 0:
            return 0.0, 0.0

        mean_sin = float(np.average(np.sin(yaws), weights=weights))
        mean_cos = float(np.average(np.cos(yaws), weights=weights))
        mean_yaw = math.atan2(mean_sin, mean_cos)
        yaw_errors = np.array(
            [self.wrap_angle(float(yaw - mean_yaw)) for yaw in yaws], dtype=np.float64
        )
        yaw_variance = float(np.average(yaw_errors * yaw_errors, weights=weights))
        return mean_yaw, yaw_variance

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

        weights = self.build_weights(history_count)
        response.position_variance = self.compute_position_variance(history, weights)
        _, response.yaw_variance = self.compute_yaw_variance(history, weights)

        point_counts = np.array(
            [float(sample.point_count) for sample in history], dtype=np.float64
        )
        anisotropy = np.array(
            [float(sample.anisotropy_ratio) for sample in history], dtype=np.float64
        )
        pca_yaw_flags = np.array(
            [1.0 if sample.yaw_source.lower() == 'pca' else 0.0 for sample in history],
            dtype=np.float64,
        )

        response.mean_point_count = float(np.average(point_counts, weights=weights))
        response.mean_anisotropy_ratio = float(np.average(anisotropy, weights=weights))
        mean_pca_fraction = float(np.average(pca_yaw_flags, weights=weights))

        position_stability = 1.0 / (
            1.0 + response.position_variance / max(self.position_variance_norm, 1e-9)
        )
        yaw_stability = 1.0 / (
            1.0 + response.yaw_variance / max(self.yaw_variance_norm, 1e-9)
        )
        point_score = float(
            np.clip(response.mean_point_count / max(self.point_count_norm, 1e-9), 0.0, 1.0)
        )
        anisotropy_score = float(
            np.clip(
                response.mean_anisotropy_ratio / max(self.anisotropy_norm, 1e-9),
                0.0,
                1.0,
            )
        )

        score_weights = np.array(
            [
                self.weight_position_stability,
                self.weight_yaw_stability,
                self.weight_point_count,
                self.weight_anisotropy,
                self.weight_pca_yaw_fraction,
            ],
            dtype=np.float64,
        )
        score_terms = np.array(
            [
                position_stability,
                yaw_stability,
                point_score,
                anisotropy_score,
                mean_pca_fraction,
            ],
            dtype=np.float64,
        )
        total_weight = float(np.sum(score_weights))
        if total_weight <= 1e-12:
            response.diagnostic_message = 'Confidence weights sum to zero.'
            return response

        response.confidence_score = float(np.dot(score_weights, score_terms) / total_weight)
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
            'mean_points=%.1f mean_anisotropy=%.3f pca_fraction=%.2f: %s'
            % (
                history_count,
                response.confidence_score,
                threshold,
                response.position_variance,
                response.yaw_variance,
                response.mean_point_count,
                response.mean_anisotropy_ratio,
                mean_pca_fraction,
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
