import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker


@dataclass
class PcaResult:
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class PoseEstimate:
    centroid: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    yaw: float
    yaw_source: str
    anisotropy_ratio: float


class PoseEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_estimator")

        self.declare_parameter(
            "target_cloud_topic", "/active_perception/target_cloud"
        )
        self.declare_parameter(
            "output_pose_topic", "/active_perception/target_pose"
        )
        self.declare_parameter("base_frame", "oakd_rgb_camera_optical_frame")
        self.declare_parameter("anisotropy_threshold", 0.2)
        self.declare_parameter("min_points", 30)
        self.declare_parameter("broadcast_tf", True)

        self.target_cloud_topic = (
            self.get_parameter("target_cloud_topic")
            .get_parameter_value()
            .string_value
        )
        self.output_pose_topic = (
            self.get_parameter("output_pose_topic")
            .get_parameter_value()
            .string_value
        )
        self.output_sample_topic = (
            self.get_parameter("output_sample_topic")
            .get_parameter_value()
            .string_value
        )
        self.base_frame = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )
        self.anisotropy_threshold = (
            self.get_parameter("anisotropy_threshold")
            .get_parameter_value()
            .double_value
        )
        self.min_points = (
            self.get_parameter("min_points").get_parameter_value().integer_value
        )
        self.broadcast_tf = (
            self.get_parameter("broadcast_tf").get_parameter_value().bool_value
        )

        self.pose_pub = self.create_publisher(PoseStamped, self.output_pose_topic, 10)
        self.sample_pub = self.create_publisher(
            PoseEstimateSample, self.output_sample_topic, 10
        )
        self.axes_pub = self.create_publisher(
            Marker, "/active_perception/target_axes", 10
        )
        self.centroid_pub = self.create_publisher(
            Marker, "/active_perception/target_centroid", 10
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.target_cloud_topic, self.target_cloud_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            "Pose estimator ready: cloud='%s', pose='%s', base_frame='%s'"
            % (self.target_cloud_topic, self.output_pose_topic, self.base_frame)
        )

    def pointcloud2_to_xyz_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        field_offsets = {field.name: field.offset for field in cloud_msg.fields}
        required_fields = ("x", "y", "z")
        if not all(name in field_offsets for name in required_fields):
            raise ValueError("PointCloud2 message does not contain x, y, z fields")

        point_count = int(cloud_msg.width) * int(cloud_msg.height)
        expected_size = point_count * int(cloud_msg.point_step)
        if len(cloud_msg.data) < expected_size:
            raise ValueError("PointCloud2 data buffer is smaller than expected")

        buffer = np.frombuffer(cloud_msg.data, dtype=np.uint8, count=expected_size)
        point_bytes = buffer.reshape(point_count, cloud_msg.point_step)

        float_dtype = np.dtype(">f4" if cloud_msg.is_bigendian else "<f4")
        xyz = np.empty((point_count, 3), dtype=np.float32)

        for axis_idx, field_name in enumerate(required_fields):
            start = field_offsets[field_name]
            stop = start + 4
            xyz[:, axis_idx] = (
                point_bytes[:, start:stop].copy().view(float_dtype).reshape(-1)
            )

        finite_mask = np.all(np.isfinite(xyz), axis=1)
        return xyz[finite_mask]

    def compute_centroid(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            raise ValueError("Cannot compute centroid of an empty point cloud")
        return np.mean(points, axis=0)

    def compute_pca(self, points: np.ndarray) -> PcaResult:
        if len(points) < 2:
            raise ValueError("At least two points are required for PCA")

        centroid = self.compute_centroid(points)
        centered = points - centroid
        covariance = (centered.T @ centered) / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        return PcaResult(
            covariance=covariance,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def make_quaternion_from_yaw(self, yaw: float) -> Tuple[float, float, float, float]:
        half_yaw = 0.5 * yaw
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def compute_pose_from_cloud(self, points_base: np.ndarray) -> PoseEstimate:
        centroid = self.compute_centroid(points_base)
        pca_result = self.compute_pca(points_base)

        centered_xy = points_base[:, :2] - centroid[:2]
        if len(points_base) < 2:
            planar_covariance = np.zeros((2, 2), dtype=np.float64)
        else:
            planar_covariance = (centered_xy.T @ centered_xy) / max(
                len(points_base) - 1, 1
            )

        planar_eigenvalues, planar_eigenvectors = np.linalg.eigh(planar_covariance)
        planar_order = np.argsort(planar_eigenvalues)[::-1]
        planar_eigenvalues = planar_eigenvalues[planar_order]
        planar_eigenvectors = planar_eigenvectors[:, planar_order]

        lambda1 = float(planar_eigenvalues[0]) if len(planar_eigenvalues) > 0 else 0.0
        lambda2 = float(planar_eigenvalues[1]) if len(planar_eigenvalues) > 1 else 0.0
        anisotropy_ratio = (lambda1 - lambda2) / (lambda1 + 1e-9)

        yaw_fallback = math.atan2(float(centroid[1]), float(centroid[0]))
        use_pca_yaw = (
            anisotropy_ratio > self.anisotropy_threshold and lambda1 > 1e-9
        )

        if use_pca_yaw:
            major_axis_xy = planar_eigenvectors[:, 0]
            if np.linalg.norm(major_axis_xy) < 1e-9:
                yaw = yaw_fallback
                yaw_source = "centroid-bearing"
            else:
                bearing_xy = centroid[:2]
                if np.linalg.norm(bearing_xy) > 1e-9 and np.dot(
                    major_axis_xy, bearing_xy
                ) < 0.0:
                    major_axis_xy = -major_axis_xy

                yaw = math.atan2(float(major_axis_xy[1]), float(major_axis_xy[0]))
                yaw_source = "pca"
        else:
            yaw = yaw_fallback
            yaw_source = "centroid-bearing"

        return PoseEstimate(
            centroid=centroid,
            eigenvalues=pca_result.eigenvalues,
            eigenvectors=pca_result.eigenvectors,
            yaw=yaw,
            yaw_source=yaw_source,
            anisotropy_ratio=anisotropy_ratio,
        )

    def publish_markers(
        self, estimate: PoseEstimate, stamp, axis_length: float = 0.15
    ) -> None:
        centroid_marker = Marker()
        centroid_marker.header.frame_id = self.base_frame
        centroid_marker.header.stamp = stamp
        centroid_marker.ns = "target_centroid"
        centroid_marker.id = 0
        centroid_marker.type = Marker.SPHERE
        centroid_marker.action = Marker.ADD
        centroid_marker.pose.position.x = float(estimate.centroid[0])
        centroid_marker.pose.position.y = float(estimate.centroid[1])
        centroid_marker.pose.position.z = float(estimate.centroid[2])
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.scale.x = 0.06
        centroid_marker.scale.y = 0.06
        centroid_marker.scale.z = 0.06
        centroid_marker.color.r = 1.0
        centroid_marker.color.g = 0.85
        centroid_marker.color.b = 0.1
        centroid_marker.color.a = 0.95
        self.centroid_pub.publish(centroid_marker)

        yaw = estimate.yaw
        x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        y_axis = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        origin = estimate.centroid
        axes_marker = Marker()
        axes_marker.header.frame_id = self.base_frame
        axes_marker.header.stamp = stamp
        axes_marker.ns = "target_axes"
        axes_marker.id = 0
        axes_marker.type = Marker.LINE_LIST
        axes_marker.action = Marker.ADD
        axes_marker.pose.orientation.w = 1.0
        axes_marker.scale.x = 0.015

        axes = (
            (x_axis, ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)),
            (y_axis, ColorRGBA(r=0.1, g=1.0, b=0.1, a=1.0)),
            (z_axis, ColorRGBA(r=0.1, g=0.4, b=1.0, a=1.0)),
        )

        for axis_vector, color in axes:
            start = Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
            end_point = origin + axis_length * axis_vector
            end = Point(
                x=float(end_point[0]),
                y=float(end_point[1]),
                z=float(end_point[2]),
            )
            axes_marker.points.extend([start, end])
            axes_marker.colors.extend([color, color])

        self.axes_pub.publish(axes_marker)

    def lookup_cloud_transform(self, source_frame: str, stamp) -> TransformStamped:
        if source_frame == self.base_frame:
            identity = TransformStamped()
            identity.header.frame_id = self.base_frame
            identity.header.stamp = stamp
            identity.child_frame_id = source_frame
            identity.transform.rotation.w = 1.0
            return identity

        timeout = Duration(seconds=0.2)
        target_time = Time.from_msg(stamp)

        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                target_time,
                timeout=timeout,
            )
        except TransformException as exc:
            if stamp.sec == 0 and stamp.nanosec == 0:
                raise exc

            self.get_logger().warn(
                "TF lookup at cloud stamp failed (%s). Retrying with latest transform."
                % str(exc)
            )
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )

    def transform_points_to_base(
        self, points: np.ndarray, transform: TransformStamped
    ) -> np.ndarray:
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
            raise ValueError("Transform quaternion has zero norm")

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
        translation_vector = np.array([tx, ty, tz], dtype=np.float64)
        return points @ rotation_matrix.T + translation_vector

    def publish_pose(self, estimate: PoseEstimate, stamp) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = float(estimate.centroid[0])
        pose_msg.pose.position.y = float(estimate.centroid[1])
        pose_msg.pose.position.z = float(estimate.centroid[2])

        qx, qy, qz, qw = self.make_quaternion_from_yaw(estimate.yaw)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub.publish(pose_msg)

        if not self.broadcast_tf:
            return

        transform_msg = TransformStamped()
        transform_msg.header = pose_msg.header
        transform_msg.child_frame_id = "target_object"
        transform_msg.transform.translation.x = pose_msg.pose.position.x
        transform_msg.transform.translation.y = pose_msg.pose.position.y
        transform_msg.transform.translation.z = pose_msg.pose.position.z
        transform_msg.transform.rotation.x = pose_msg.pose.orientation.x
        transform_msg.transform.rotation.y = pose_msg.pose.orientation.y
        transform_msg.transform.rotation.z = pose_msg.pose.orientation.z
        transform_msg.transform.rotation.w = pose_msg.pose.orientation.w
        self.tf_broadcaster.sendTransform(transform_msg)

    def target_cloud_callback(self, cloud_msg: PointCloud2) -> None:
        try:
            points = self.pointcloud2_to_xyz_array(cloud_msg)
        except ValueError as exc:
            self.get_logger().error("Failed to parse PointCloud2: %s" % str(exc))
            return

        point_count = len(points)
        self.get_logger().info("Received target cloud with %d valid points" % point_count)

        if point_count < self.min_points:
            self.get_logger().warn(
                "Skipping pose estimation: only %d points, need at least %d"
                % (point_count, self.min_points)
            )
            return

        try:
            transform = self.lookup_cloud_transform(
                cloud_msg.header.frame_id, cloud_msg.header.stamp
            )
            points_base = self.transform_points_to_base(points, transform)
            estimate = self.compute_pose_from_cloud(points_base)
        except (TransformException, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warn("Pose estimation failed: %s" % str(exc))
            return

        centroid_str = np.array2string(estimate.centroid, precision=3, suppress_small=True)
        eigenvalues_str = np.array2string(
            estimate.eigenvalues, precision=4, suppress_small=True
        )
        self.get_logger().info(
            "Centroid in %s: %s | eigenvalues=%s | yaw_source=%s | anisotropy=%.3f"
            % (
                self.base_frame,
                centroid_str,
                eigenvalues_str,
                estimate.yaw_source,
                estimate.anisotropy_ratio,
            )
        )

        self.publish_pose(estimate, cloud_msg.header.stamp)
        self.publish_markers(estimate, cloud_msg.header.stamp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
