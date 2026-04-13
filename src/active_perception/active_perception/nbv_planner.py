#!/usr/bin/env python3

import math
from typing import List, Tuple

import numpy as np

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from active_perception.srv import PlanNBV


class NBVPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('nbv_planner')

        self.declare_parameter('service_name', '/active_perception/plan_nbv')
        self.declare_parameter('planning_frame', 'odom')
        self.declare_parameter('default_num_candidates', 8)
        self.declare_parameter('default_radius', 0.8)
        self.declare_parameter('candidate_marker_topic', '/active_perception/nbv_markers')
        self.declare_parameter('weight_radius_error', 0.2)
        self.declare_parameter('weight_travel_distance', 0.6)
        self.declare_parameter('weight_heading_change', 0.2)

        self.service_name = (
            self.get_parameter('service_name').get_parameter_value().string_value
        )
        self.planning_frame = (
            self.get_parameter('planning_frame').get_parameter_value().string_value
        )
        self.default_num_candidates = (
            self.get_parameter('default_num_candidates')
            .get_parameter_value()
            .integer_value
        )
        self.default_radius = (
            self.get_parameter('default_radius').get_parameter_value().double_value
        )
        self.candidate_marker_topic = (
            self.get_parameter('candidate_marker_topic')
            .get_parameter_value()
            .string_value
        )
        self.weight_radius_error = (
            self.get_parameter('weight_radius_error').get_parameter_value().double_value
        )
        self.weight_travel_distance = (
            self.get_parameter('weight_travel_distance')
            .get_parameter_value()
            .double_value
        )
        self.weight_heading_change = (
            self.get_parameter('weight_heading_change')
            .get_parameter_value()
            .double_value
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, self.candidate_marker_topic, 10
        )
        self.service = self.create_service(
            PlanNBV, self.service_name, self.plan_nbv_callback
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            "NBV planner ready on '%s' with planning_frame='%s'"
            % (self.service_name, self.planning_frame)
        )

    def wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, orientation) -> float:
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def make_quaternion_from_yaw(self, yaw: float) -> Tuple[float, float, float, float]:
        half_yaw = 0.5 * yaw
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def lookup_transform(
        self, target_frame: str, source_frame: str, stamp
    ) -> TransformStamped:
        if target_frame == source_frame:
            identity = TransformStamped()
            identity.header.frame_id = target_frame
            identity.header.stamp = stamp
            identity.child_frame_id = source_frame
            identity.transform.rotation.w = 1.0
            return identity

        timeout = Duration(seconds=0.2)
        target_time = Time.from_msg(stamp)

        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                target_time,
                timeout=timeout,
            )
        except TransformException as exc:
            if stamp.sec == 0 and stamp.nanosec == 0:
                raise exc

            self.get_logger().warn(
                "TF lookup at pose stamp failed (%s). Retrying with latest transform."
                % str(exc)
            )
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )

    def transform_pose_to_frame(
        self, pose_msg: PoseStamped, target_frame: str
    ) -> PoseStamped:
        source_frame = pose_msg.header.frame_id or target_frame
        transform = self.lookup_transform(target_frame, source_frame, pose_msg.header.stamp)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        tx = float(translation.x)
        ty = float(translation.y)
        tz = float(translation.z)
        qx = float(rotation.x)
        qy = float(rotation.y)
        qz = float(rotation.z)
        qw = float(rotation.w)

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-12:
            raise ValueError('Transform quaternion has zero norm')

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        rotation_matrix = np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )

        point = np.array(
            [
                float(pose_msg.pose.position.x),
                float(pose_msg.pose.position.y),
                float(pose_msg.pose.position.z),
            ],
            dtype=np.float64,
        )
        transformed_point = rotation_matrix @ point + np.array(
            [tx, ty, tz], dtype=np.float64
        )

        yaw_source = self.quaternion_to_yaw(pose_msg.pose.orientation)
        transform_yaw = self.quaternion_to_yaw(rotation)
        transformed_yaw = self.wrap_angle(yaw_source + transform_yaw)
        qx_out, qy_out, qz_out, qw_out = self.make_quaternion_from_yaw(transformed_yaw)

        transformed_pose = PoseStamped()
        transformed_pose.header.frame_id = target_frame
        transformed_pose.header.stamp = pose_msg.header.stamp
        transformed_pose.pose.position.x = float(transformed_point[0])
        transformed_pose.pose.position.y = float(transformed_point[1])
        transformed_pose.pose.position.z = float(transformed_point[2])
        transformed_pose.pose.orientation.x = qx_out
        transformed_pose.pose.orientation.y = qy_out
        transformed_pose.pose.orientation.z = qz_out
        transformed_pose.pose.orientation.w = qw_out
        return transformed_pose

    def create_pose(self, x: float, y: float, z: float, yaw: float) -> Pose:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        qx, qy, qz, qw = self.make_quaternion_from_yaw(yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def publish_markers(
        self,
        candidates: PoseArray,
        selected_index: int,
        target_pose: PoseStamped,
        radius: float,
    ) -> None:
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        ring_marker = Marker()
        ring_marker.header = target_pose.header
        ring_marker.ns = 'nbv_ring'
        ring_marker.id = 0
        ring_marker.type = Marker.LINE_STRIP
        ring_marker.action = Marker.ADD
        ring_marker.pose.orientation.w = 1.0
        ring_marker.scale.x = 0.02
        ring_marker.color.r = 0.1
        ring_marker.color.g = 0.8
        ring_marker.color.b = 1.0
        ring_marker.color.a = 0.7

        for i in range(33):
            angle = (2.0 * math.pi * i) / 32.0
            ring_marker.points.append(
                Point(
                    x=float(target_pose.pose.position.x + radius * math.cos(angle)),
                    y=float(target_pose.pose.position.y + radius * math.sin(angle)),
                    z=0.05,
                )
            )

        marker_array.markers.append(ring_marker)

        for idx, pose in enumerate(candidates.poses):
            marker = Marker()
            marker.header = candidates.header
            marker.ns = 'nbv_candidates'
            marker.id = 100 + idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.25
            marker.scale.y = 0.06
            marker.scale.z = 0.06
            marker.color = (
                ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.95)
                if idx == selected_index
                else ColorRGBA(r=1.0, g=0.6, b=0.1, a=0.6)
            )
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def plan_nbv_callback(
        self, request: PlanNBV.Request, response: PlanNBV.Response
    ) -> PlanNBV.Response:
        response.success = False
        response.selected_index = -1

        try:
            target_pose = self.transform_pose_to_frame(
                request.target_pose, self.planning_frame
            )
            robot_pose = self.transform_pose_to_frame(
                request.robot_pose, self.planning_frame
            )
        except (TransformException, ValueError) as exc:
            response.diagnostic_message = 'Failed to transform poses: %s' % str(exc)
            return response

        num_candidates = int(request.num_candidates)
        if num_candidates <= 0:
            num_candidates = int(self.default_num_candidates)
        if num_candidates < 3:
            num_candidates = 3

        min_radius = float(request.min_radius) if request.min_radius > 0.0 else 0.4
        max_radius = float(request.max_radius) if request.max_radius > 0.0 else 1.5
        if max_radius < min_radius:
            min_radius, max_radius = max_radius, min_radius

        desired_radius = (
            float(request.radius) if request.radius > 0.0 else float(self.default_radius)
        )
        desired_radius = float(np.clip(desired_radius, min_radius, max_radius))

        target_xy = np.array(
            [target_pose.pose.position.x, target_pose.pose.position.y], dtype=np.float64
        )
        robot_xy = np.array(
            [robot_pose.pose.position.x, robot_pose.pose.position.y], dtype=np.float64
        )
        robot_yaw = self.quaternion_to_yaw(robot_pose.pose.orientation)

        current_distance = float(np.linalg.norm(robot_xy - target_xy))
        radius = desired_radius
        if request.use_adaptive_radius and current_distance > 1e-6:
            radius = float(np.clip(current_distance, min_radius, max_radius))

        base_angle = math.atan2(
            float(robot_xy[1] - target_xy[1]), float(robot_xy[0] - target_xy[0])
        )

        candidates = PoseArray()
        candidates.header.frame_id = self.planning_frame
        candidates.header.stamp = target_pose.header.stamp

        candidate_costs: List[float] = []
        for idx in range(num_candidates):
            angle = base_angle + (2.0 * math.pi * idx) / float(num_candidates)
            x = float(target_xy[0] + radius * math.cos(angle))
            y = float(target_xy[1] + radius * math.sin(angle))
            z = 0.0
            yaw = math.atan2(float(target_xy[1] - y), float(target_xy[0] - x))

            pose = self.create_pose(x, y, z, yaw)
            candidates.poses.append(pose)

            candidate_xy = np.array([x, y], dtype=np.float64)
            travel_distance = float(np.linalg.norm(candidate_xy - robot_xy))
            heading_change = abs(self.wrap_angle(yaw - robot_yaw))
            radius_error = abs(radius - desired_radius)

            radius_term = radius_error / max(desired_radius, 1e-6)
            travel_term = travel_distance / max(max_radius, 1e-6)
            heading_term = heading_change / math.pi

            total_cost = (
                self.weight_radius_error * radius_term
                + self.weight_travel_distance * travel_term
                + self.weight_heading_change * heading_term
            )
            candidate_costs.append(float(total_cost))

        if not candidate_costs:
            response.diagnostic_message = 'No candidate viewpoints generated.'
            return response

        selected_index = int(np.argmin(candidate_costs))
        response.success = True
        response.selected_index = selected_index
        response.candidate_views = candidates
        response.best_view.header = candidates.header
        response.best_view.pose = candidates.poses[selected_index]
        response.diagnostic_message = (
            'selected=%d radius=%.3f num_candidates=%d min_cost=%.3f'
            % (
                selected_index,
                radius,
                num_candidates,
                candidate_costs[selected_index],
            )
        )

        self.publish_markers(candidates, selected_index, target_pose, radius)
        return response


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


if __name__ == '__main__':
    main()
