#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Transform, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[2, 1] - rotation[1, 2]) / s,
            ],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform: Transform) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    q = transform.rotation
    mat[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    mat[0, 3] = float(transform.translation.x)
    mat[1, 3] = float(transform.translation.y)
    mat[2, 3] = float(transform.translation.z)
    return mat


def odom_pose_to_matrix(msg: Odometry) -> np.ndarray:
    pose = msg.pose.pose
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quaternion_to_rotation_matrix(
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    mat[0, 3] = float(pose.position.x)
    mat[1, 3] = float(pose.position.y)
    mat[2, 3] = float(pose.position.z)
    return mat


@dataclass
class StereoCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline: float
    camera_frame: str
    left_map1: np.ndarray
    left_map2: np.ndarray
    right_map1: np.ndarray
    right_map2: np.ndarray

    @classmethod
    def from_camera_info(cls, left: CameraInfo, right: CameraInfo) -> "StereoCalibration":
        size = (int(left.width), int(left.height))
        left_k = np.asarray(left.k, dtype=np.float64).reshape(3, 3)
        right_k = np.asarray(right.k, dtype=np.float64).reshape(3, 3)
        left_d = np.asarray(left.d, dtype=np.float64)
        right_d = np.asarray(right.d, dtype=np.float64)
        left_r = np.asarray(left.r, dtype=np.float64).reshape(3, 3)
        right_r = np.asarray(right.r, dtype=np.float64).reshape(3, 3)
        left_p = np.asarray(left.p, dtype=np.float64).reshape(3, 4)
        right_p = np.asarray(right.p, dtype=np.float64).reshape(3, 4)

        left_rect_k = left_p[:, :3]
        right_rect_k = right_p[:, :3]

        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            left_k, left_d, left_r, left_rect_k, size, cv2.CV_32FC1
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            right_k, right_d, right_r, right_rect_k, size, cv2.CV_32FC1
        )

        tx_left = float(left_p[0, 3]) / float(left_p[0, 0]) if abs(float(left_p[0, 0])) > 1e-12 else 0.0
        tx_right = float(right_p[0, 3]) / float(right_p[0, 0]) if abs(float(right_p[0, 0])) > 1e-12 else 0.0
        baseline = abs(tx_left - tx_right)

        return cls(
            fx=float(left_rect_k[0, 0]),
            fy=float(left_rect_k[1, 1]),
            cx=float(left_rect_k[0, 2]),
            cy=float(left_rect_k[1, 2]),
            width=size[0],
            height=size[1],
            baseline=baseline,
            camera_frame=str(left.header.frame_id),
            left_map1=left_map1,
            left_map2=left_map2,
            right_map1=right_map1,
            right_map2=right_map2,
        )


class OrbStereoVONode(Node):
    def __init__(self) -> None:
        super().__init__("orb_stereo_vo_node")

        self.declare_parameter("left_image_topic", "/robot_10/oakd/left/image_raw")
        self.declare_parameter("right_image_topic", "/robot_10/oakd/right/image_raw")
        self.declare_parameter("left_camera_info_topic", "/robot_10/oakd/left/camera_info")
        self.declare_parameter("right_camera_info_topic", "/robot_10/oakd/right/camera_info")
        self.declare_parameter("vo_odom_topic", "/orb_slam/vo_odom")
        self.declare_parameter("vo_path_topic", "/orb_slam/vo_path")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_vo_tf", False)
        self.declare_parameter("require_base_tf", True)
        self.declare_parameter("wheel_odom_topic", "/robot_10/odom")
        self.declare_parameter("align_vo_to_odom_on_start", True)
        self.declare_parameter("vocabulary_file", "")
        self.declare_parameter("settings_file", "")
        self.declare_parameter("orbslam_backend_library", "")
        self.declare_parameter("generated_settings_file", "/tmp/orb_ekf_orbslam3_stereo.yaml")
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("max_sync_delta_ms", 15.0)
        self.declare_parameter("enable_sync_gate", True)
        self.declare_parameter("use_rectification", True)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.path_pub = self.create_publisher(PathMsg, str(self.get_parameter("vo_path_topic").value), 10)
        self.odom_pub = self.create_publisher(Odometry, str(self.get_parameter("vo_odom_topic").value), 10)

        self.path_msg = PathMsg()
        self.path_msg.header.frame_id = str(self.get_parameter("world_frame").value)

        self.calibration: Optional[StereoCalibration] = None
        self.backend = None
        self.backend_module = None
        self.left_info_msg: Optional[CameraInfo] = None
        self.right_info_msg: Optional[CameraInfo] = None
        self.t_camera_base: Optional[np.ndarray] = None
        self.last_tf_lookup_fail_ns = 0
        self.last_align_wait_log_ns = 0
        self.last_timestamp: Optional[float] = None
        self.accepted_pairs = 0
        self.rejected_pairs = 0
        self.latest_wheel_odom: Optional[Odometry] = None
        self.t_odom_orbworld: Optional[np.ndarray] = None

        self.left_info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("left_camera_info_topic").value),
            self.on_left_camera_info,
            qos_profile_sensor_data,
        )
        self.wheel_odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("wheel_odom_topic").value),
            self.on_wheel_odom,
            50,
        )
        self.right_info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("right_camera_info_topic").value),
            self.on_right_camera_info,
            qos_profile_sensor_data,
        )

        sync_slop = float(self.get_parameter("sync_slop").value)
        self.left_image_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("left_image_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self.right_image_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("right_image_topic").value),
            qos_profile=qos_profile_sensor_data,
        )
        self.image_sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub], queue_size=30, slop=sync_slop
        )
        self.image_sync.registerCallback(self.on_stereo)

        self.get_logger().info("ORB stereo VO node started. Waiting for camera_info.")

    def on_wheel_odom(self, msg: Odometry) -> None:
        self.latest_wheel_odom = msg

    def on_left_camera_info(self, msg: CameraInfo) -> None:
        if self.left_info_msg is None:
            self.left_info_msg = msg
            self.get_logger().info(f"Left camera info received: frame='{msg.header.frame_id}'")
        self.maybe_start()

    def on_right_camera_info(self, msg: CameraInfo) -> None:
        if self.right_info_msg is None:
            self.right_info_msg = msg
            self.get_logger().info(f"Right camera info received: frame='{msg.header.frame_id}'")
        self.maybe_start()

    def maybe_start(self) -> None:
        if self.calibration is None and self.left_info_msg is not None and self.right_info_msg is not None:
            self.calibration = StereoCalibration.from_camera_info(self.left_info_msg, self.right_info_msg)
            self.get_logger().info(
                f"Stereo calibration ready: fx={self.calibration.fx:.3f}, "
                f"baseline={self.calibration.baseline:.5f} m, "
                f"size={self.calibration.width}x{self.calibration.height}"
            )
        if self.backend is None and self.calibration is not None:
            self.start_backend()

    def _load_backend_module(self):
        if self.backend_module is not None:
            return self.backend_module

        try:
            self.backend_module = importlib.import_module("orbslam3_backend")
            return self.backend_module
        except Exception:
            pass

        explicit_path = str(self.get_parameter("orbslam_backend_library").value).strip()
        candidate_files = []
        if explicit_path:
            candidate_files.append(Path(explicit_path))

        env_lib = os.environ.get("ORBSLAM3_BACKEND_PATH", "").strip()
        if env_lib:
            candidate_files.append(Path(env_lib))

        search_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]]
        for directory in search_dirs:
            if directory.exists():
                candidate_files.extend(sorted(directory.glob("orbslam3_backend*.so")))

        for candidate in candidate_files:
            if not candidate.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location("orbslam3_backend", str(candidate))
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.backend_module = module
                self.get_logger().info(f"Loaded orbslam3_backend from '{candidate}'.")
                return self.backend_module
            except Exception as exc:
                self.get_logger().warn(f"Failed loading backend from '{candidate}': {exc}")

        self.get_logger().error(
            "Could not import orbslam3_backend. Install it in the Python env, "
            "or set parameter 'orbslam_backend_library' to the compiled .so path, "
            "or set ORBSLAM3_BACKEND_PATH."
        )
        return None

    def find_default_vocabulary(self) -> str:
        candidates = [
            str(Path.cwd() / "vendor" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"),
            str(Path(__file__).resolve().parents[2] / "vendor" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"),
            str(Path.home() / "ORBvoc.txt"),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return ""

    def start_backend(self) -> None:
        assert self.calibration is not None
        if self.backend is not None:
            return
        backend_module = self._load_backend_module()
        if backend_module is None:
            return

        vocabulary_file = str(self.get_parameter("vocabulary_file").value).strip()
        if not vocabulary_file:
            vocabulary_file = self.find_default_vocabulary()
        if not vocabulary_file:
            self.get_logger().error("Missing vocabulary file. Set parameter 'vocabulary_file' to ORBvoc.txt path.")
            return

        settings_file = str(self.get_parameter("settings_file").value).strip()
        if not settings_file:
            settings_file = self.write_settings_file()

        self.backend = backend_module.StereoSystem(
            vocabulary_file=vocabulary_file,
            settings_file=settings_file,
            use_viewer=False,
        )
        self.get_logger().info("ORB-SLAM backend initialized.")

    def write_settings_file(self) -> str:
        assert self.calibration is not None
        output_path = Path(str(self.get_parameter("generated_settings_file").value))
        fps = int(round(float(self.get_parameter("camera_fps").value)))

        yaml_text = "\n".join(
            [
                "%YAML:1.0",
                "",
                "File.version: \"1.0\"",
                "",
                "Camera.type: \"Rectified\"",
                f"Camera1.fx: {self.calibration.fx:.9f}",
                f"Camera1.fy: {self.calibration.fy:.9f}",
                f"Camera1.cx: {self.calibration.cx:.9f}",
                f"Camera1.cy: {self.calibration.cy:.9f}",
                f"Camera.width: {self.calibration.width}",
                f"Camera.height: {self.calibration.height}",
                f"Camera.fps: {fps}",
                "Camera.RGB: 0",
                "",
                f"Stereo.b: {self.calibration.baseline:.9f}",
                "Stereo.ThDepth: 40.0",
                "",
                "ORBextractor.nFeatures: 1800",
                "ORBextractor.scaleFactor: 1.2",
                "ORBextractor.nLevels: 8",
                "ORBextractor.iniThFAST: 12",
                "ORBextractor.minThFAST: 5",
                "",
            ]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_text, encoding="ascii")
        self.get_logger().info(f"Generated ORB settings file: '{output_path}'")
        return str(output_path)

    def try_resolve_camera_to_base(self) -> bool:
        if self.t_camera_base is not None:
            return True
        if self.calibration is None:
            return False

        base_frame = str(self.get_parameter("base_frame").value)
        camera_frame = self.calibration.camera_frame
        try:
            t_base_camera = self.tf_buffer.lookup_transform(base_frame, camera_frame, Time())
            self.t_camera_base = np.linalg.inv(transform_to_matrix(t_base_camera.transform))
            self.get_logger().info(f"Resolved TF: {base_frame} <- {camera_frame}")
            return True
        except TransformException as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_tf_lookup_fail_ns > int(2e9):
                self.get_logger().warn(
                    f"Waiting for TF {base_frame} <- {camera_frame}: {exc}. "
                    "ORB pose-to-base transform is pending."
                )
                self.last_tf_lookup_fail_ns = now_ns
            return False

    def on_stereo(self, left_msg: Image, right_msg: Image) -> None:
        if self.calibration is None:
            return
        if self.backend is None:
            self.start_backend()
            if self.backend is None:
                return

        if bool(self.get_parameter("require_base_tf").value) and not self.try_resolve_camera_to_base():
            return
        if self.t_camera_base is None:
            self.try_resolve_camera_to_base()

        t_left = float(left_msg.header.stamp.sec) + float(left_msg.header.stamp.nanosec) * 1e-9
        t_right = float(right_msg.header.stamp.sec) + float(right_msg.header.stamp.nanosec) * 1e-9
        sync_delta_ms = abs(t_left - t_right) * 1000.0

        if bool(self.get_parameter("enable_sync_gate").value):
            max_delta_ms = float(self.get_parameter("max_sync_delta_ms").value)
            if sync_delta_ms > max_delta_ms:
                self.rejected_pairs += 1
                return

        if self.last_timestamp is not None and t_left <= self.last_timestamp:
            self.get_logger().warn("Timestamp moved backwards, skipping frame.")
            return
        self.last_timestamp = t_left
        self.accepted_pairs += 1

        left_raw = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        right_raw = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")

        if bool(self.get_parameter("use_rectification").value):
            left_img = cv2.remap(left_raw, self.calibration.left_map1, self.calibration.left_map2, cv2.INTER_LINEAR)
            right_img = cv2.remap(
                right_raw, self.calibration.right_map1, self.calibration.right_map2, cv2.INTER_LINEAR
            )
        else:
            left_img = left_raw
            right_img = right_raw

        pose_t_cw = np.asarray(self.backend.track_stereo(left_img, right_img, t_left), dtype=np.float64)
        if pose_t_cw.shape != (4, 4):
            return
        if hasattr(self.backend, "is_lost") and self.backend.is_lost():
            return

        t_world_camera = np.linalg.inv(pose_t_cw)
        t_world_base = t_world_camera
        if self.t_camera_base is not None:
            t_world_base = t_world_camera @ self.t_camera_base

        t_publish = t_world_base
        if bool(self.get_parameter("align_vo_to_odom_on_start").value):
            if self.t_odom_orbworld is None:
                if self.latest_wheel_odom is None:
                    now_ns = self.get_clock().now().nanoseconds
                    if now_ns - self.last_align_wait_log_ns > int(2e9):
                        self.get_logger().warn("Waiting for wheel odom to compute startup VO->odom alignment.")
                        self.last_align_wait_log_ns = now_ns
                    return
                t_odom_base0 = odom_pose_to_matrix(self.latest_wheel_odom)
                self.t_odom_orbworld = t_odom_base0 @ np.linalg.inv(t_world_base)
                self.get_logger().info("Initialized startup alignment: ORB world -> odom.")
            t_publish = self.t_odom_orbworld @ t_world_base

        translation = t_publish[:3, 3]
        quaternion = rotation_to_quaternion(t_publish[:3, :3])
        self.publish_outputs(left_msg.header, translation, quaternion)

    def publish_outputs(self, header, translation: np.ndarray, quaternion: np.ndarray) -> None:
        world_frame = str(self.get_parameter("world_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = world_frame
        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])

        self.path_msg.header.stamp = header.stamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = base_frame
        odom_msg.pose.pose = pose_msg.pose
        odom_msg.pose.covariance = [
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.08,
        ]
        self.odom_pub.publish(odom_msg)

        if bool(self.get_parameter("publish_vo_tf").value):
            tf_msg = TransformStamped()
            tf_msg.header = odom_msg.header
            tf_msg.child_frame_id = base_frame
            tf_msg.transform.translation.x = odom_msg.pose.pose.position.x
            tf_msg.transform.translation.y = odom_msg.pose.pose.position.y
            tf_msg.transform.translation.z = odom_msg.pose.pose.position.z
            tf_msg.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

    def destroy_node(self) -> bool:
        if self.backend is not None:
            try:
                self.backend.shutdown()
            except Exception:
                pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OrbStereoVONode()
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
