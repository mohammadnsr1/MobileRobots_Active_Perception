#!/usr/bin/env python3

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray


class PipelineConfig:
    """
    Holds parameters for the point cloud processing pipeline.
    The layout mirrors the cylinder finder so the two nodes stay easy to compare.
    """

    def __init__(self):
        # Topic settings
        self.topic = '/robot_10/oakd/points'

        # Voxel Downsampling
        self.voxel_size = 0.01

        # Passthrough/Box Filter (Min/Max XYZ)
        self.box_min = np.array([-1.5, -3.0, 0.2])
        self.box_max = np.array([1.5, 1.0, 4.0])

        # Plane RANSAC
        self.floor_dist = 0.02
        self.target_normal = np.array([0.0, 1.0, 0.0])  # Assuming Y-up for floor
        self.normal_thresh = 0.85

        # Euclidean clustering
        self.cluster_k = 15
        self.cluster_tolerance = 0.06
        self.min_cluster_size = 400
        self.max_cluster_size = 5000

        # Box fitting heuristics
        self.min_box_points = 80
        self.box_axis_align_thresh = 0.6
        self.min_box_dimensions = np.array([0.05, 0.05, 0.05])
        self.max_box_dimensions = np.array([0.6, 0.6, 0.6])

        # Color filtering for brown cardboard-like boxes
        self.brown_hue_min = 0.04
        self.brown_hue_max = 0.13
        self.brown_saturation_min = 0.2
        self.brown_saturation_max = 0.8
        self.brown_value_min = 0.15
        self.brown_value_max = 0.95
        self.brown_rg_min = 0.03
        self.brown_gb_min = 0.015


@dataclass
class BoxDetection:
    center: np.ndarray
    rotation_matrix: np.ndarray
    dimensions: np.ndarray
    inlier_points: np.ndarray
    inlier_colors: np.ndarray
    inlier_count: int
    score: float
    display_color: np.ndarray
    label: str


class BoxVisualizer:
    """
    Handles the creation and publishing of RViz MarkerArrays to represent
    detected boxes.
    """

    def __init__(self, publisher):
        self.pub_markers = publisher

    def rotation_matrix_to_quaternion(
        self, rotation_matrix: np.ndarray
    ) -> Tuple[float, float, float, float]:
        trace = float(np.trace(rotation_matrix))

        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and (
            rotation_matrix[0, 0] > rotation_matrix[2, 2]
        ):
            s = 2.0 * np.sqrt(
                1.0
                + rotation_matrix[0, 0]
                - rotation_matrix[1, 1]
                - rotation_matrix[2, 2]
            )
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(
                1.0
                + rotation_matrix[1, 1]
                - rotation_matrix[0, 0]
                - rotation_matrix[2, 2]
            )
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(
                1.0
                + rotation_matrix[2, 2]
                - rotation_matrix[0, 0]
                - rotation_matrix[1, 1]
            )
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            qz = 0.25 * s

        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        quaternion_norm = np.linalg.norm(quaternion)
        if quaternion_norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)

        quaternion /= quaternion_norm
        return (
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )

    def create_box_marker(
        self,
        center: np.ndarray,
        rotation_matrix: np.ndarray,
        dimensions: np.ndarray,
        rgb: np.ndarray,
        marker_id: int,
        frame_id: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])

        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rotation_matrix)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        marker.scale.x = float(max(dimensions[0], 1e-3))
        marker.scale.y = float(max(dimensions[1], 1e-3))
        marker.scale.z = float(max(dimensions[2], 1e-3))

        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        marker.color.a = 0.8
        return marker

    def publish_viz(self, detections: List[BoxDetection], frame_id: str):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for i, detection in enumerate(detections):
            marker = self.create_box_marker(
                detection.center,
                detection.rotation_matrix,
                detection.dimensions,
                detection.display_color,
                2000 + i,
                frame_id,
            )
            marker_array.markers.append(marker)

        self.pub_markers.publish(marker_array)


class BoxPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def get_neighbors(self, pts, queries, k=15):
        """
        Calculates k-nearest neighbors using a KDTree.
        """
        if len(pts) < k:
            return None

        tree = cKDTree(pts)
        _, idxs = tree.query(queries, k=k)
        return idxs

    def box_filter(self, pts, colors):
        """
        Removes points outside the specified XYZ bounding box.
        """
        mask = np.all((pts >= self.cfg.box_min) & (pts <= self.cfg.box_max), axis=1)
        return pts[mask], colors[mask]

    def downsample(self, pts, colors):
        """
        Reduces point cloud density using a voxel grid approach.
        """
        if len(pts) == 0:
            return pts, colors

        voxel_coords = np.floor(pts / self.cfg.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        unique_indices.sort()
        return pts[unique_indices], colors[unique_indices]

    def color_filter(self, pts, colors):
        """
        Keeps only points whose RGB appearance is consistent with a brown box.
        """
        if len(pts) == 0:
            return pts, colors

        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape[1] != 3:
            return pts, colors

        rgb_max = np.max(colors, axis=1)
        rgb_min = np.min(colors, axis=1)
        chroma = rgb_max - rgb_min

        hue = np.zeros(len(colors), dtype=np.float32)
        nonzero = chroma > 1e-6

        red_is_max = nonzero & (rgb_max == colors[:, 0])
        green_is_max = nonzero & (rgb_max == colors[:, 1])
        blue_is_max = nonzero & (rgb_max == colors[:, 2])

        hue[red_is_max] = (
            ((colors[red_is_max, 1] - colors[red_is_max, 2]) / chroma[red_is_max])
            % 6.0
        ) / 6.0
        hue[green_is_max] = (
            ((colors[green_is_max, 2] - colors[green_is_max, 0]) / chroma[green_is_max])
            + 2.0
        ) / 6.0
        hue[blue_is_max] = (
            ((colors[blue_is_max, 0] - colors[blue_is_max, 1]) / chroma[blue_is_max])
            + 4.0
        ) / 6.0

        saturation = np.zeros(len(colors), dtype=np.float32)
        value_nonzero = rgb_max > 1e-6
        saturation[value_nonzero] = chroma[value_nonzero] / rgb_max[value_nonzero]
        value = rgb_max

        brown_mask = (
            (hue >= self.cfg.brown_hue_min)
            & (hue <= self.cfg.brown_hue_max)
            & (saturation >= self.cfg.brown_saturation_min)
            & (saturation <= self.cfg.brown_saturation_max)
            & (value >= self.cfg.brown_value_min)
            & (value <= self.cfg.brown_value_max)
            & ((colors[:, 0] - colors[:, 1]) >= self.cfg.brown_rg_min)
            & ((colors[:, 1] - colors[:, 2]) >= self.cfg.brown_gb_min)
        )

        return pts[brown_mask], colors[brown_mask]

    def find_plane_ransac(self, pts, iters=100):
        """
        Fits a plane model (ax + by + cz + d = 0) to the cloud using RANSAC.
        """
        if len(pts) < 3:
            return None, None

        target_normal = np.asarray(self.cfg.target_normal, dtype=np.float64)

        best_model = None
        best_inliers = None
        best_count = 0

        for _ in range(iters):
            sample_idxs = np.random.choice(len(pts), size=3, replace=False)
            p1, p2, p3 = pts[sample_idxs]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm <= 1e-12:
                continue

            normal = normal / normal_norm
            alignment = abs(np.dot(normal, target_normal))
            if alignment < self.cfg.normal_thresh:
                continue

            d = -np.dot(normal, p1)
            distances = np.abs(pts @ normal + d)
            inliers = distances <= self.cfg.floor_dist
            inlier_count = np.count_nonzero(inliers)

            if inlier_count > best_count:
                best_count = inlier_count
                best_model = (normal, d)
                best_inliers = inliers

        if best_model is None:
            return None, None

        return best_model, best_inliers

    def fit_box(self, pts):
        """
        Estimates an upright box from a top-down footprint and vertical extent.
        """
        if len(pts) < 3:
            return None

        vertical = np.asarray(self.cfg.target_normal, dtype=np.float64)
        vertical_norm = np.linalg.norm(vertical)
        if vertical_norm <= 1e-12:
            return None
        vertical = vertical / vertical_norm

        reference_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(reference_axis, vertical)) > 0.9:
            reference_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        plane_x = reference_axis - vertical * np.dot(reference_axis, vertical)
        plane_x_norm = np.linalg.norm(plane_x)
        if plane_x_norm <= 1e-12:
            return None
        plane_x = plane_x / plane_x_norm

        plane_y = np.cross(vertical, plane_x)
        plane_y_norm = np.linalg.norm(plane_y)
        if plane_y_norm <= 1e-12:
            return None
        plane_y = plane_y / plane_y_norm

        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        footprint_2d = np.column_stack((centered @ plane_x, centered @ plane_y))
        if len(footprint_2d) < 2:
            return None

        covariance_2d = (footprint_2d.T @ footprint_2d) / max(len(footprint_2d) - 1, 1)
        eigenvalues_2d, eigenvectors_2d = np.linalg.eigh(covariance_2d)
        order_2d = np.argsort(eigenvalues_2d)[::-1]
        eigenvectors_2d = eigenvectors_2d[:, order_2d]

        major_coeffs = eigenvectors_2d[:, 0]
        major_axis = plane_x * major_coeffs[0] + plane_y * major_coeffs[1]
        major_norm = np.linalg.norm(major_axis)
        if major_norm <= 1e-12:
            return None
        major_axis = major_axis / major_norm

        vertical_axis = vertical
        minor_axis = np.cross(vertical_axis, major_axis)
        minor_norm = np.linalg.norm(minor_axis)
        if minor_norm <= 1e-12:
            return None
        minor_axis = minor_axis / minor_norm

        major_axis = np.cross(minor_axis, vertical_axis)
        major_norm = np.linalg.norm(major_axis)
        if major_norm <= 1e-12:
            return None
        major_axis = major_axis / major_norm

        rotation_matrix = np.column_stack((major_axis, minor_axis, vertical_axis))
        local_points = centered @ rotation_matrix

        local_min = np.percentile(local_points, 2.0, axis=0)
        local_max = np.percentile(local_points, 98.0, axis=0)
        dimensions = local_max - local_min

        if np.any(dimensions < self.cfg.min_box_dimensions):
            return None
        if np.any(dimensions > self.cfg.max_box_dimensions):
            return None

        padding = max(0.01, 0.5 * float(self.cfg.voxel_size))
        inlier_mask = np.all(
            (local_points >= (local_min - padding))
            & (local_points <= (local_max + padding)),
            axis=1,
        )
        if np.count_nonzero(inlier_mask) < 3:
            return None

        box_local_points = local_points[inlier_mask]

        local_min = np.min(box_local_points, axis=0)
        local_max = np.max(box_local_points, axis=0)
        dimensions = local_max - local_min
        local_center = 0.5 * (local_min + local_max)
        center = centroid + rotation_matrix @ local_center

        if np.any(dimensions < self.cfg.min_box_dimensions):
            return None
        if np.any(dimensions > self.cfg.max_box_dimensions):
            return None

        horizontal_dims = dimensions[:2]
        largest_horizontal = float(np.max(horizontal_dims))
        smallest_horizontal = float(np.min(horizontal_dims))

        # Keep the ratio only as a soft preference so partial views of the box
        # are not rejected outright when the observed extents are imbalanced.
        side_ratio = smallest_horizontal / max(largest_horizontal, 1e-9)

        score = float(np.count_nonzero(inlier_mask)) * float(0.5 + 0.5 * side_ratio)
        return center, rotation_matrix, dimensions, score, inlier_mask

    def euclidean_clustering(self, pts):
        """
        Groups nearby points into connected components using kNN expansion.
        """
        if len(pts) == 0:
            return []

        cluster_k = min(int(self.cfg.cluster_k), len(pts))
        if cluster_k <= 0:
            return []

        neighbor_idxs = self.get_neighbors(pts, pts, k=cluster_k)
        if neighbor_idxs is None:
            return []

        neighbor_idxs = np.atleast_2d(neighbor_idxs)
        cluster_tol_sq = float(self.cfg.cluster_tolerance) ** 2
        valid_neighbors = []

        for i, idxs in enumerate(neighbor_idxs):
            deltas = pts[idxs] - pts[i]
            dist_sq = np.sum(deltas * deltas, axis=1)
            valid_neighbors.append(idxs[dist_sq <= cluster_tol_sq])

        visited = np.zeros(len(pts), dtype=bool)
        clusters = []

        for start_idx in range(len(pts)):
            if visited[start_idx]:
                continue

            queue = deque([start_idx])
            visited[start_idx] = True
            cluster = []

            while queue:
                idx = queue.popleft()
                cluster.append(idx)

                for neighbor_idx in valid_neighbors[idx]:
                    neighbor_idx = int(neighbor_idx)
                    if visited[neighbor_idx]:
                        continue
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

            cluster_size = len(cluster)
            if self.cfg.min_cluster_size <= cluster_size <= self.cfg.max_cluster_size:
                clusters.append(np.asarray(cluster, dtype=np.int32))

        return clusters


class BoxProcessorNode(Node):
    def __init__(self):
        super().__init__('box_processor_node')
        self.cfg = PipelineConfig()
        self.pipeline = BoxPipeline(self.cfg)

        self.pub_stage0 = self.create_publisher(PointCloud2, '/robot_10/active_perception/stage0_box', 10)
        self.pub_stage3 = self.create_publisher(
            PointCloud2, '/robot_10/active_perception/stage3_candidates', 10
        )
        self.pub_target_cloud = self.create_publisher(
            PointCloud2, '/robot_10/active_perception/target_cloud', 10
        )

        marker_pub = self.create_publisher(MarkerArray, '/robot_10/viz/detections', 10)
        self.visualizer = BoxVisualizer(marker_pub)

        self.sub = self.create_subscription(
            PointCloud2, self.cfg.topic, self.listener_callback, 10
        )

    def numpy_to_pc2_rgb(self, pts, colors, frame_id, stamp=None):
        """
        Converts Nx3 XYZ coordinates and Nx3 RGB color arrays into a PointCloud2.
        """
        pts = np.asarray(pts, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)

        msg = PointCloud2()
        msg.header.frame_id = frame_id
        if stamp is not None:
            msg.header.stamp = stamp
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.is_dense = True
        msg.row_step = 16 * len(pts)

        c = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint32)
        rgb_packed = (255 << 24) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        data = np.hstack([pts, rgb_packed.view(np.float32).reshape(-1, 1)])
        msg.data = data.tobytes()
        return msg

    def pointcloud2_to_xyz_rgb(
        self, cloud_msg: PointCloud2
    ) -> Tuple[np.ndarray, np.ndarray]:
        stride = cloud_msg.point_step // 4
        raw_data = np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, stride)

        pts = raw_data[:, :3]
        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]

        rgb_index = self._get_rgb_field_index(cloud_msg)
        if rgb_index is None:
            colors = np.zeros((len(pts), 3), dtype=np.float32)
            return pts, colors

        rgb_uint32 = raw_data[finite_mask, rgb_index].view(np.uint32)
        colors = np.vstack(
            [
                ((rgb_uint32 >> 16) & 0xFF) / 255.0,
                ((rgb_uint32 >> 8) & 0xFF) / 255.0,
                (rgb_uint32 & 0xFF) / 255.0,
            ]
        ).T.astype(np.float32)
        return pts, colors

    def _get_rgb_field_index(self, cloud_msg: PointCloud2) -> Optional[int]:
        for field in cloud_msg.fields:
            if field.name in ('rgb', 'rgba'):
                return field.offset // 4
        self.get_logger().warn(
            'Incoming PointCloud2 has no rgb/rgba field; using zero colors.'
        )
        return None

    def average_display_color(self, colors: np.ndarray) -> np.ndarray:
        if colors is None or len(colors) == 0:
            return np.array([0.2, 0.8, 1.0], dtype=np.float32)

        avg_color = np.mean(colors, axis=0).astype(np.float32)
        avg_color = np.clip(avg_color, 0.0, 1.0)
        if np.linalg.norm(avg_color) <= 1e-6:
            return np.array([0.2, 0.8, 1.0], dtype=np.float32)
        return avg_color

    def evaluate_cluster(
        self,
        cluster_idx: int,
        cluster_pts: np.ndarray,
        cluster_colors: np.ndarray,
    ) -> Optional[BoxDetection]:
        cluster_size = len(cluster_pts)
        self.get_logger().info(f'Cluster {cluster_idx}: size={cluster_size}')

        if cluster_size < self.cfg.min_box_points:
            self.get_logger().info(
                f'Cluster {cluster_idx}: rejected, too few points for box fitting'
            )
            return None

        box_fit = self.pipeline.fit_box(cluster_pts)
        if box_fit is None:
            self.get_logger().info(
                f'Cluster {cluster_idx}: rejected, box extents/orientation not valid'
            )
            return None

        center, rotation_matrix, dimensions, score, inlier_mask = box_fit
        box_pts = cluster_pts[inlier_mask]
        box_colors = cluster_colors[inlier_mask]
        if len(box_pts) < max(30, self.cfg.min_box_points // 2):
            self.get_logger().info(
                f'Cluster {cluster_idx}: rejected, refined box subset too small'
            )
            return None

        display_color = self.average_display_color(box_colors)

        dims_str = np.array2string(dimensions, precision=3, suppress_small=True)
        self.get_logger().info(
            f'Cluster {cluster_idx}: accepted, dims={dims_str}, '
            f'score={score:.1f}, box_points={len(box_pts)}'
        )

        return BoxDetection(
            center=center,
            rotation_matrix=rotation_matrix,
            dimensions=dimensions,
            inlier_points=box_pts,
            inlier_colors=box_colors,
            inlier_count=len(box_pts),
            score=score,
            display_color=display_color,
            label='box',
        )

    def publish_candidate_debug_cloud(
        self,
        detections: List[BoxDetection],
        frame_id: str,
        stamp=None,
    ) -> None:
        if detections:
            pts_box = np.concatenate(
                [detection.inlier_points for detection in detections], axis=0
            )
            colors_box = np.concatenate(
                [detection.inlier_colors for detection in detections], axis=0
            )
        else:
            pts_box = np.empty((0, 3), dtype=np.float32)
            colors_box = np.empty((0, 3), dtype=np.float32)

        self.pub_stage3.publish(
            self.numpy_to_pc2_rgb(pts_box, colors_box, frame_id, stamp=stamp)
        )

    def select_best_detection(
        self, detections: List[BoxDetection]
    ) -> Optional[BoxDetection]:
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.score)

    def listener_callback(self, msg: PointCloud2):
        """
        Main ROS callback. Orchestrates the flow from PointCloud2 to box detection.
        """
        frame_id = msg.header.frame_id
        stamp = msg.header.stamp
        pts, raw_colors = self.pointcloud2_to_xyz_rgb(msg)

        pts_box, colors_box = self.pipeline.box_filter(pts, raw_colors)
        pts_v, colors_v = self.pipeline.downsample(pts_box, colors_box)
        floor_model, floor_inliers = self.pipeline.find_plane_ransac(pts_v)

        pts_candidates = pts_v
        colors_candidates = colors_v
        if floor_model is not None and floor_inliers is not None:
            pts_candidates = pts_v[~floor_inliers]
            colors_candidates = colors_v[~floor_inliers]
            self.pub_stage0.publish(
                self.numpy_to_pc2_rgb(pts_box, colors_box, frame_id, stamp=stamp)
            )
        else:
            self.pub_stage0.publish(
                self.numpy_to_pc2_rgb(pts_v, colors_v, frame_id, stamp=stamp)
            )

        pts_candidates, colors_candidates = self.pipeline.color_filter(
            pts_candidates, colors_candidates
        )
        self.get_logger().info(
            f'Brown color-filtered candidate points: {len(pts_candidates)}'
        )

        clusters = self.pipeline.euclidean_clustering(pts_candidates)
        self.get_logger().info(
            f'Euclidean clusters after floor removal and color filtering: {len(clusters)}'
        )

        detections: List[BoxDetection] = []
        for cluster_idx, idxs in enumerate(clusters):
            cluster_pts = pts_candidates[idxs]
            cluster_colors = colors_candidates[idxs]
            detection = self.evaluate_cluster(cluster_idx, cluster_pts, cluster_colors)
            if detection is not None:
                detections.append(detection)

        self.publish_candidate_debug_cloud(detections, frame_id, stamp=stamp)
        self.visualizer.publish_viz(detections, frame_id)

        best_detection = self.select_best_detection(detections)
        if best_detection is None:
            self.get_logger().info(
                'No box target selected; nothing published on /active_perception/target_cloud'
            )
            return

        dims_str = np.array2string(
            best_detection.dimensions, precision=3, suppress_small=True
        )
        self.get_logger().info(
            f"Selected target '{best_detection.label}' with dims={dims_str} "
            f'and points={best_detection.inlier_count}'
        )
        self.pub_target_cloud.publish(
            self.numpy_to_pc2_rgb(
                best_detection.inlier_points,
                best_detection.inlier_colors,
                frame_id,
                stamp=stamp,
            )
        )


def main():
    rclpy.init()
    node = BoxProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
