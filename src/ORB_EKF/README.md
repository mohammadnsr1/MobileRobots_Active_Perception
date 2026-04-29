# ORB_EKF — ROS 2 Package (`orb_ekf`)

Stereo Visual Odometry (VO) for ROS 2 using ORB-SLAM3, with a fused path output.

## What this package does

Two nodes run together:

1. **`orb_vo_node`** — subscribes to a stereo camera pair, runs ORB-SLAM3 tracking, and publishes the camera trajectory as odometry and a path in the `odom` frame.
2. **`fused_output_node`** — subscribes to the VO odometry and wheel odometry, computes a naive position average, and publishes it as a path for visualization.

---

## Prerequisites

- ROS 2 (Jazzy or compatible)
- Python 3.12
- System packages: `build-essential`, `cmake`, `libeigen3-dev`, `libopencv-dev`, `libglew-dev`, `libgl1-mesa-dev`

---

## Step 1 — Install ORB-SLAM3 and build the Python backend

Run this **once** from inside the package directory. It clones Pangolin and ORB-SLAM3 under `vendor/`, builds them, and compiles the Python bridge (`orbslam3_backend*.so`).

```bash
cd /path/to/ros2_ws/src/ORB_EKF
./install_orbslam3.sh
```

After this completes, the following files will exist:

| File | Purpose |
|---|---|
| `vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt` | ORB visual vocabulary required by ORB-SLAM3 at startup |
| `orbslam3_backend.cpython-312-x86_64-linux-gnu.so` | Compiled Python/C++ bridge that lets the Python node call ORB-SLAM3 C++ APIs |

> If you already have ORB-SLAM3 built and only need to rebuild the Python bridge, run `./build_orbslam3_backend.sh` directly.

---

## Step 2 — Build the ROS 2 package

Run from the **workspace root** (not from inside `ORB_EKF`):

```bash
cd /path/to/ros2_ws
colcon build --packages-select orb_ekf --symlink-install
source install/setup.bash
```

`--symlink-install` means edits to Python source files take effect immediately without rebuilding.

---

## Step 3 — Run

```bash
ros2 launch orb_ekf orb_ekf.launch.py
```

The node will automatically locate the vocabulary file and camera settings using the workspace install path. If auto-detection fails, pass them explicitly:

```bash
ros2 launch orb_ekf orb_ekf.launch.py \
  vocabulary_file:=/path/to/ros2_ws/src/ORB_EKF/vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  orbslam_backend_library:=/path/to/ros2_ws/src/ORB_EKF/orbslam3_backend.cpython-312-x86_64-linux-gnu.so
```

To override input or output topics at launch time:

```bash
ros2 launch orb_ekf orb_ekf.launch.py \
  left_image_topic:=/my_robot/left/image_raw \
  right_image_topic:=/my_robot/right/image_raw \
  left_camera_info_topic:=/my_robot/left/camera_info \
  right_camera_info_topic:=/my_robot/right/camera_info \
  wheel_odom_topic:=/my_robot/odom
```

---

## Topics

### Inputs (subscribed)

| Topic | Type | Description |
|---|---|---|
| `/robot_10/oakd/left/image_raw` | `sensor_msgs/Image` | Left rectified or raw camera image |
| `/robot_10/oakd/right/image_raw` | `sensor_msgs/Image` | Right rectified or raw camera image |
| `/robot_10/oakd/left/camera_info` | `sensor_msgs/CameraInfo` | Left camera calibration (K, D, R, P) |
| `/robot_10/oakd/right/camera_info` | `sensor_msgs/CameraInfo` | Right camera calibration (K, D, R, P) |
| `/robot_10/odom` | `nav_msgs/Odometry` | Wheel odometry used for startup alignment and fusion |

Default topic names are set in `config/orb_vo.yaml` and can be overridden at launch.

### Outputs (published)

| Topic | Type | Node | Description |
|---|---|---|---|
| `/orb_slam/vo_odom` | `nav_msgs/Odometry` | `orb_stereo_vo_node` | VO pose in the `odom` frame at camera frame rate |
| `/orb_slam/vo_path` | `nav_msgs/Path` | `orb_stereo_vo_node` | Full VO trajectory since startup |
| `/orb_ekf/average_path` | `nav_msgs/Path` | `orb_ekf_fused_output_node` | Naive average of VO position and wheel odom position |

---

## Configuration files

### `config/orb_vo.yaml`

ROS 2 parameter file loaded by `orb_vo_node` at startup. Controls runtime behavior.

| Parameter | Default | Description |
|---|---|---|
| `left_image_topic` | `/robot_10/oakd/left/image_raw` | Left camera image input |
| `right_image_topic` | `/robot_10/oakd/right/image_raw` | Right camera image input |
| `left_camera_info_topic` | `/robot_10/oakd/left/camera_info` | Left camera info input |
| `right_camera_info_topic` | `/robot_10/oakd/right/camera_info` | Right camera info input |
| `wheel_odom_topic` | `/robot_10/odom` | Wheel odometry input |
| `vo_odom_topic` | `/orb_slam/vo_odom` | VO odometry output |
| `vo_path_topic` | `/orb_slam/vo_path` | VO path output |
| `world_frame` | `odom` | Fixed frame for publishing poses |
| `base_frame` | `base_link` | Robot base frame |
| `publish_vo_tf` | `false` | Whether to broadcast `odom → base_link` TF from VO |
| `require_base_tf` | `false` | If true, wait for camera→base_link TF before processing |
| `align_vo_to_odom_on_start` | `false` | Align VO origin to wheel odom pose at first tracked frame |
| `use_rectification` | `true` | Apply undistort+rectify maps to images before tracking |
| `enable_sync_gate` | `true` | Drop stereo pairs whose timestamps differ by more than `max_sync_delta_ms` |
| `max_sync_delta_ms` | `15.0` | Maximum allowed timestamp gap between left and right images (ms) |
| `sync_slop` | `0.05` | ApproximateTimeSynchronizer slop window (seconds) |
| `camera_fps` | `30.0` | Used when auto-generating the ORB-SLAM3 settings file |
| `vocabulary_file` | `""` | Path to `ORBvoc.txt`; auto-located via workspace if empty |
| `settings_file` | `""` | Path to ORB-SLAM3 camera YAML; auto-located if empty |
| `orbslam_backend_library` | `""` | Path to `orbslam3_backend*.so`; searched automatically if empty |
| `generated_settings_file` | `/tmp/orb_ekf_orbslam3_stereo.yaml` | Where to write the auto-generated settings file |

**To change topic names:** edit `config/orb_vo.yaml`. No rebuild needed if built with `--symlink-install`.

---

### `config/orbslam3_stereo.yaml`

ORB-SLAM3 camera and feature extractor settings. Loaded directly by the ORB-SLAM3 C++ library (not by ROS 2 parameter system).

| Parameter | Value | Description |
|---|---|---|
| `Camera.type` | `Rectified` | Tells ORB-SLAM3 images arrive pre-rectified |
| `Camera1.fx / fy` | `457.30` | Focal length in pixels (from left camera projection matrix) |
| `Camera1.cx / cy` | `338.46 / 249.60` | Principal point in pixels (left camera, used for both cameras after rectification fix) |
| `Camera.width / height` | `640 / 480` | Image resolution |
| `Camera.fps` | `30` | Camera frame rate |
| `Stereo.b` | `0.07521` | Stereo baseline in meters |
| `Stereo.ThDepth` | `40.0` | Depth threshold in units of baseline (points beyond this are treated as far) |
| `ORBextractor.nFeatures` | `1500` | Number of ORB features extracted per frame |
| `ORBextractor.scaleFactor` | `1.2` | Scale between pyramid levels |
| `ORBextractor.nLevels` | `8` | Number of image pyramid levels |
| `ORBextractor.iniThFAST` | `20` | Initial FAST corner threshold |
| `ORBextractor.minThFAST` | `7` | Minimum FAST threshold if not enough corners found at initial threshold |

This file is used when `settings_file` is set explicitly. If `settings_file` is left empty in `orb_vo.yaml`, the node generates an equivalent file at runtime from the live `camera_info` messages.

---

### `config/ekf.yaml`

Configuration for `robot_localization`'s EKF node. **Not loaded by the current launch file.** Provided for optional manual use if you want to run a full EKF fusion alongside this package.

It fuses:
- `odom0`: `/robot_10/odom` — wheel odometry (x, y, yaw, vx, vy, vyaw)
- `odom1`: `/orb_slam/vo_odom` — VO odometry (x, y, yaw only, pose-only)

To use it manually:
```bash
ros2 run robot_localization ekf_node --ros-args --params-file config/ekf.yaml
```

---

## `vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt`

The ORB visual vocabulary file. It is a large pre-trained bag-of-words database (~1M visual words) used by ORB-SLAM3 for place recognition and loop closure. ORB-SLAM3 **will not initialize** without it. It is extracted from a compressed archive during `./install_orbslam3.sh` and is not committed to the repository.

---

## Node details

### `orb_stereo_vo_node` (executable: `orb_vo_node`)

- Waits for both `camera_info` messages, then builds rectification maps from the calibration.
- Loads the ORB-SLAM3 backend and vocabulary.
- For each synchronized stereo pair: rectifies both images, passes them to `track_stereo`, converts the resulting camera pose to the `odom` frame, and publishes `/orb_slam/vo_odom` and `/orb_slam/vo_path`.
- On first tracked frame, captures the VO origin and aligns it to the current wheel odometry pose so both paths start at the same point.

### `orb_ekf_fused_output_node` (executable: `fused_output_node`)

- Subscribes to `/orb_slam/vo_odom` and `/robot_10/odom`.
- At each VO frame, takes the latest wheel odom position and averages it with the VO position (x, y only).
- Publishes the averaged positions as `/orb_ekf/average_path` for visualization.
- This is a lightweight placeholder. For proper sensor fusion, use the EKF setup in `config/ekf.yaml` with `robot_localization`.

---

## Visualizing in RViz

Add the following displays in RViz (Fixed Frame: `odom`):

| Display type | Topic | Color suggestion |
|---|---|---|
| Path | `/orb_slam/vo_path` | Purple — VO trajectory |
| Path | `/orb_ekf/average_path` | Green — averaged path |
| Odometry | `/robot_10/odom` | White arrows — wheel odometry |
