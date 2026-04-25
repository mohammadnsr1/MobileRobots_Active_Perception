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


# ==========================================
# CONFIGURATION CLASS
# ==========================================
class PipelineConfig:
    """
    Holds parameters for the point cloud processing pipeline.
    You can add, change or remove any of the parameters here.
    """

    def __init__(self):
        # Topic settings
        self.topic = '/robot_10/oakd/points'

        # Voxel Downsampling
        self.voxel_size = 0.02

        # Passthrough/Box Filter (Min/Max XYZ)
        self.box_min = np.array([-0.6, -2.0, 0.2])
        self.box_max = np.array([0.6, 1.0, 2.0])

        # Plane RANSAC
        self.floor_dist = 0.02
        self.cyl_dist = 0.01
        self.target_normal = np.array([0, 1, 0])  # Assuming Y-up for floor
        self.normal_thresh = 0.85

        # Euclidean clustering
        self.cluster_k = 15
        self.cluster_tolerance = 0.06
        self.min_cluster_size = 100
        self.max_cluster_size = 1000

        # Cylinder RANSAC
        self.cyl_radius = 0.055
        self.max_cylinders = 3


@dataclass
class CylinderDetection:
    model: Tuple[np.ndarray, np.ndarray, float]
    inlier_points: np.ndarray
    inlier_colors: np.ndarray
    inlier_count: int
    display_color: np.ndarray
    label: str


# ==========================================
# VISUALIZER CLASS
# ==========================================
class CylinderVisualizer:
    """
    Handles the creation and publishing of RViz MarkerArrays to represent
    detected cylinders.
    """

    def __init__(self, publisher):
        self.pub_markers = publisher

    def create_cylinder_marker(self, center, radius, rgb, marker_id, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.id = marker_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD

        m.pose.position.x = float(center[0])
        m.pose.position.y = float(0.0)  # Snap to floor level for visualization
        m.pose.position.z = float(center[2])

        # Rotate cylinder to stand upright
        m.pose.orientation.x = 0.7071
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 0.7071

        m.scale.x = float(radius * 2.0)
        m.scale.y = float(radius * 2.0)
        m.scale.z = 0.4

        m.color.r = float(rgb[0])
        m.color.g = float(rgb[1])
        m.color.b = float(rgb[2])
        m.color.a = 0.8
        return m

    def publish_viz(self, detections: List[CylinderDetection], frame_id: str):
        ma = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        ma.markers.append(clear_marker)

        for i, detection in enumerate(detections):
            center, _, radius = detection.model
            marker = self.create_cylinder_marker(
                center,
                radius,
                detection.display_color,
                2000 + i,
                frame_id,
            )
            ma.markers.append(marker)

        self.pub_markers.publish(ma)


# ==========================================
# PIPELINE LOGIC
# ==========================================
class CylinderPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def rgb_to_hsv(self, r, g, b):
        """
        Converts a single RGB point to HSV color space.

        :param r: Red component (0.0 - 1.0)
        :param g: Green component (0.0 - 1.0)
        :param b: Blue component (0.0 - 1.0)
        :return: Tuple (h, s, v) where H is [0, 360], S and V are [0, 1]
        """
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        else:
            h = (60 * ((r - g) / df) + 240) % 360

        s = 0 if mx == 0 else (df / mx)
        v = mx

        return h, s, v

    def classify_cylinder_color(self, colors):
        """
        Classify a verified cylinder cluster by averaging its RGB colors,
        converting the average to HSV, and thresholding in HSV space.

        :param colors: Nx3 RGB array with values in [0, 1]
        :return: (display_rgb, color_name)
        """
        unknown_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if colors is None or len(colors) == 0:
            return unknown_rgb, "unknown"

        avg_rgb = np.mean(colors, axis=0).astype(np.float32)
        avg_rgb = np.clip(avg_rgb, 0.0, 1.0)
        h, s, v = self.rgb_to_hsv(avg_rgb[0], avg_rgb[1], avg_rgb[2])

        if v < 0.15:
            return unknown_rgb, "unknown"

        if ((h < 25.0) or (h >= 335.0)):
            if s < 0.45 and v > 0.5:
                return np.array([1.0, 0.5, 0.7], dtype=np.float32), "pink"
            return np.array([1.0, 0.2, 0.2], dtype=np.float32), "red"

        if 80.0 <= h < 170.0:
            return np.array([0.2, 1.0, 0.2], dtype=np.float32), "green"

        if 190.0 <= h < 280.0:
            return np.array([0.2, 0.4, 1.0], dtype=np.float32), "blue"

        return unknown_rgb, "unknown"

    def get_neighbors(self, pts, queries, k=15):
        """
        Calculates k-nearest neighbors using a KDTree.

        :param pts: The source point cloud (Nx3).
        :param queries: The points for which we want neighbors (Mx3).
        :param k: Number of neighbors to find.
        :return: Indices of neighbors in the 'pts' array.
        """
        if len(pts) < k:
            return None

        tree = cKDTree(pts)
        _, idxs = tree.query(queries, k=k)
        return idxs

    def box_filter(self, pts, colors):
        """
        Removes points outside the specified XYZ bounding box.

        :param pts: Input XYZ array.
        :param colors: Input RGB array.
        :return: Tuple of (filtered_pts, filtered_colors).
        """
        mask = np.all((pts >= self.cfg.box_min) & (pts <= self.cfg.box_max), axis=1)
        filtered_pts = pts[mask]
        filtered_colors = colors[mask]
        return filtered_pts, filtered_colors

    def downsample(self, pts, colors):
        """
        Reduces point cloud density using a voxel grid approach.

        Implementation Hint: Convert points to integer coordinates by dividing
        by voxel_size, then use np.unique to find one point per voxel.
        """
        if len(pts) == 0:
            return pts, colors

        voxel_coords = np.floor(pts / self.cfg.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        unique_indices.sort()
        return pts[unique_indices], colors[unique_indices]

    def estimate_normals(self, pts, k=15):
        """
        Estimates a surface normal for every point.

        Implementation Hint:
        1. For each point, find k-neighbors.
        2. Compute the Singular Value Decomposition (SVD), using np.linalg.svd, of the centered neighbors.
        3. The normal is the eigenvector corresponding to the smallest eigenvalue.
        """
        if len(pts) == 0:
            return None

        neighbor_idxs = self.get_neighbors(pts, pts, k=k)
        if neighbor_idxs is None:
            return None

        normals = np.zeros((len(pts), 3), dtype=np.float32)
        for i, idxs in enumerate(neighbor_idxs):
            neighbors = pts[idxs]
            centered = neighbors - neighbors.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i] = normal / norm

        return normals

    def find_plane_ransac(self, pts, iters=100):
        """
        Fits a plane model (ax + by + cz + d = 0) to the cloud using RANSAC.

        Implementation Hint:
        1. Sample 3 random points to define a plane.
        2. Calculate the normal and check if it aligns with self.cfg.target_normal.
        3. Count how many points are within self.cfg.floor_dist of the plane.
        4. Return the model with the most inliers.
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

    def find_single_cylinder(self, pts, normals, iters=300):
        """
        Fits a cylinder model to the remaining points using RANSAC.

        Implementation Hint:
        1. Sample 2 points and their normals.
        2. The cylinder axis is roughly the cross product of the two normals.
        3. Check axis alignment with the vertical.
        4. Project points and find distance to the axis; compare to self.cfg.cyl_radius.
        """
        if len(pts) < 2:
            return None, None

        vertical = np.asarray(self.cfg.target_normal, dtype=np.float64)
        vertical_norm = np.linalg.norm(vertical)
        if vertical_norm <= 1e-12:
            return None, None
        vertical = vertical / vertical_norm

        radius = float(self.cfg.cyl_radius)
        dist_thresh = float(self.cfg.cyl_dist)

        best_model = None
        best_inliers = None
        best_count = 0

        for _ in range(iters):
            sample_idxs = np.random.choice(len(pts), size=2, replace=False)
            p1, p2 = pts[sample_idxs]
            n1, n2 = normals[sample_idxs]

            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            if n1_norm <= 1e-12 or n2_norm <= 1e-12:
                continue

            n1_unit = n1 / n1_norm
            n2_unit = n2 / n2_norm

            axis = np.cross(n1, n2)
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 1e-12:
                continue

            axis = axis / axis_norm
            alignment = abs(np.dot(axis, vertical))
            if alignment < self.cfg.normal_thresh:
                continue

            if np.dot(axis, vertical) < 0:
                axis = -axis

            c1 = p1 - radius * n1_unit
            c2 = p2 - radius * n2_unit
            axis_point = 0.5 * (c1 + c2)

            v = pts - axis_point
            radial_vec = v - np.outer(v @ axis, axis)
            radial_distance = np.linalg.norm(radial_vec, axis=1)

            inliers = np.abs(radial_distance - radius) <= dist_thresh
            inlier_count = np.count_nonzero(inliers)

            if inlier_count > best_count:
                best_count = inlier_count
                best_model = (axis_point, axis, radius)
                best_inliers = inliers

        if best_model is None:
            return None, None

        return best_model, best_inliers

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


# ==========================================
# ROS NODE
# ==========================================
class CylinderProcessorNode(Node):
    def __init__(self):
        super().__init__('cylinder_processor_node')
        self.cfg = PipelineConfig()
        self.pipeline = CylinderPipeline(self.cfg)
        self.min_cylinder_inliers = 80

        # Publishers for debugging the pipeline stages in RViz
        self.pub_stage0 = self.create_publisher(PointCloud2, 'pipeline/stage0_box', 10)
        self.pub_stage3 = self.create_publisher(PointCloud2, 'pipeline/stage3_candidates', 10)
        self.pub_target_cloud = self.create_publisher(PointCloud2, '/active_perception/target_cloud', 10)

        # Marker publisher for the final detection results
        marker_pub = self.create_publisher(MarkerArray, 'viz/detections', 10)
        self.visualizer = CylinderVisualizer(marker_pub)

        self.sub = self.create_subscription(PointCloud2, self.cfg.topic, self.listener_callback, 10)

    def numpy_to_pc2_rgb(self, pts, colors, frame_id, stamp=None):
        """
        Converts Nx3 XYZ coordinates and Nx3 RGB color arrays into a ROS 2 PointCloud2 message.

        This utility handles the conversion of floating-point spatial data and the packing
        of three 8-bit color channels (R, G, B) into a single 32-bit float field, which is
        the standard format for RGB point clouds in ROS and RViz.

        :param pts: A numpy array of shape (N, 3) containing [x, y, z] coordinates.
        :param colors: A numpy array of shape (N, 3) containing [r, g, b] values (0.0 to 1.0).
        :param frame_id: The TF frame string (e.g., 'camera_link') for the message header.
        :param stamp: Optional ROS time stamp to preserve timing from the input cloud.
        :return: A sensor_msgs/PointCloud2 message ready for publishing.
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

    def pointcloud2_to_xyz_rgb(self, cloud_msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
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
            "Incoming PointCloud2 has no rgb/rgba field; using zero colors."
        )
        return None

    def evaluate_cluster(
        self,
        cluster_idx: int,
        cluster_pts: np.ndarray,
        cluster_colors: np.ndarray,
    ) -> Optional[CylinderDetection]:
        cluster_size = len(cluster_pts)
        self.get_logger().info(f"Cluster {cluster_idx}: size={cluster_size}")

        if cluster_size < self.min_cylinder_inliers:
            self.get_logger().info(
                f"Cluster {cluster_idx}: rejected, too few points for cylinder fitting"
            )
            return None

        normals_cluster = self.pipeline.estimate_normals(cluster_pts)
        if normals_cluster is None:
            self.get_logger().info(
                f"Cluster {cluster_idx}: rejected, normal estimation failed"
            )
            return None

        cyl_model, cyl_inliers = self.pipeline.find_single_cylinder(
            cluster_pts, normals_cluster
        )
        if cyl_model is None or cyl_inliers is None:
            self.get_logger().info(
                f"Cluster {cluster_idx}: rejected, no cylinder model found"
            )
            return None

        inlier_count = int(np.count_nonzero(cyl_inliers))
        if inlier_count < self.min_cylinder_inliers:
            self.get_logger().info(
                f"Cluster {cluster_idx}: rejected, cylinder inliers={inlier_count}"
            )
            return None

        inlier_pts = cluster_pts[cyl_inliers]
        inlier_colors = cluster_colors[cyl_inliers]
        display_color, color_name = self.pipeline.classify_cylinder_color(inlier_colors)

        self.get_logger().info(
            f"Cluster {cluster_idx}: accepted, cylinder inliers={inlier_count}"
        )

        return CylinderDetection(
            model=cyl_model,
            inlier_points=inlier_pts,
            inlier_colors=inlier_colors,
            inlier_count=inlier_count,
            display_color=display_color,
            label=color_name,
        )

    def publish_candidate_debug_cloud(
        self,
        detections: List[CylinderDetection],
        frame_id: str,
        stamp=None,
    ) -> None:
        if detections:
            pts_cylinder = np.concatenate(
                [detection.inlier_points for detection in detections], axis=0
            )
            colors_cylinder = np.concatenate(
                [detection.inlier_colors for detection in detections], axis=0
            )
        else:
            pts_cylinder = np.empty((0, 3), dtype=np.float32)
            colors_cylinder = np.empty((0, 3), dtype=np.float32)

        self.pub_stage3.publish(
            self.numpy_to_pc2_rgb(pts_cylinder, colors_cylinder, frame_id, stamp=stamp)
        )

    def select_best_detection(
        self, detections: List[CylinderDetection]
    ) -> Optional[CylinderDetection]:
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.inlier_count)

    def listener_callback(self, msg: PointCloud2):
        """
        Main ROS callback. Orchestrates the flow from PointCloud2 to cylinder detection.
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

        clusters = self.pipeline.euclidean_clustering(pts_candidates)
        self.get_logger().info(
            f"Euclidean clusters after floor removal: {len(clusters)}"
        )

        detections: List[CylinderDetection] = []
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
                "No cylinder target selected; nothing published on /active_perception/target_cloud"
            )
            return

        self.get_logger().info(
            f"Selected target '{best_detection.label}' with inliers={best_detection.inlier_count}"
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
    node = CylinderProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
