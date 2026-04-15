# ORB_EKF (ROS 2 package: `orb_ekf`)

This package wires:

1. ORB-SLAM stereo VO (`orb_vo_node`) from left/right images.
2. EKF fusion (`robot_localization/ekf_node`) using wheel odom + ORB VO odom.
3. Final fused outputs (`fused_output_node`) as path + pose.

## Topics

- Stereo input:
  - `/robot_10/oakd/left/image_raw`
  - `/robot_10/oakd/right/image_raw`
  - `/robot_10/oakd/left/camera_info`
  - `/robot_10/oakd/right/camera_info`
- Wheel odom input:
  - `/robot_10/odom`
- ORB VO output:
  - `/orb_slam/vo_odom`
  - `/orb_slam/vo_path`
- EKF output:
  - `/odometry/filtered`
- Fused output node:
  - `/orb_ekf/fused_pose`
  - `/orb_ekf/fused_path`

## Install ORB-SLAM (required first)

`ORB_EKF` depends on the ORB-SLAM backend (`orbslam3_backend`) and vocabulary file (`ORBvoc.txt`).

Run this once from `ORB_EKF`:

```bash
cd ORB_EKF
./install_orbslam3.sh
./build_orbslam3_backend.sh
```

What this installs/builds:
- Pangolin (under `ORB_EKF/vendor/pangolin_install`)
- ORB-SLAM3 core (under `ORB_EKF/vendor/ORB_SLAM3`)
- Python backend bridge (`orbslam3_backend*.so`)

Typical paths after install:
- Vocabulary: `/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt`
- Backend `.so`: `/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/orbslam3_backend.cpython-312-x86_64-linux-gnu.so`

## Build ORB_EKF

```bash
cd ORB_EKF
colcon build --packages-select orb_ekf --base-paths .
source install/setup.bash
```

Dependencies are declared in [package.xml](/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/package.xml), including `robot_localization`, `launch_ros`, `tf2_ros`, `cv_bridge`, and message packages.

For standalone use, `orbslam3_backend` must be available:
- either importable in your Python environment (`import orbslam3_backend`)
- or passed as a compiled shared library path via launch argument `orbslam_backend_library:=/abs/path/orbslam3_backend*.so`

## What each piece is doing (and why)

- `vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt`: ORB-SLAM visual vocabulary. It is required to perform place recognition and robust feature matching; ORB-SLAM will not initialize without it.
- `orbslam3_backend*.so`: Python/C++ bridge built from `orbslam3_backend.cpp`. It lets the Python ROS 2 node call ORB-SLAM C++ (`StereoSystem`, `track_stereo`) frame-by-frame. 
- `orb_vo_node.py`: ROS 2 bridge node for stereo VO. It subscribes to left/right image + camera_info, rectifies images, calls ORB-SLAM backend, converts camera pose to `base_link` using TF, aligns startup pose to wheel odom, and publishes `/orb_slam/vo_odom`.
- `config/orb_vo.yaml`: runtime behavior for ORB node (topics, sync limits, TF behavior, startup alignment, ORB backend path options).
- `config/ekf.yaml`: `robot_localization` fusion model. It tells EKF which states from wheel odom and VO odom should be fused in 2D (`x, y, yaw`, etc.) and publishes fused `/odometry/filtered`.
- `fused_output_node.py`: convenience publisher that converts EKF odom into easy-to-plot pose/path topics (`/orb_ekf/fused_pose`, `/orb_ekf/fused_path`).
- `launch/orb_ekf.launch.py`: runs the full stack in the correct order with shared parameters.

## Launch execution flow

When you run `ros2 launch orb_ekf orb_ekf.launch.py ...`, the system does this:

1. Starts `orb_vo_node` and waits for camera info + stereo images.
2. Loads `orbslam3_backend` and ORB vocabulary, then runs ORB-SLAM tracking per stereo pair.
3. Converts ORB camera pose to `base_link` using TF (`base_link <- camera_frame`).
4. Initializes ORB-world to odom startup alignment using first wheel odom pose.
5. Publishes VO odom (`/orb_slam/vo_odom`) in odom-aligned frame.
6. Starts `robot_localization` `ekf_node` and fuses `/robot_10/odom` + `/orb_slam/vo_odom`.
7. Publishes fused odom (`/odometry/filtered`) and final `odom -> base_link` TF.
8. Publishes fused path/pose for visualization.

## Run

Use this as the default launch command:
- `cd ORB_EKF`: enters the standalone package root.
- `source install/setup.bash`: loads this package's ROS 2 environment after build.
- `ros2 launch orb_ekf orb_ekf.launch.py`: starts ORB VO node, EKF node, and fused output node together.
- `vocabulary_file:=.../ORBvoc.txt`: points to ORB-SLAM visual vocabulary file (required by ORB-SLAM system initialization).
- `orbslam_backend_library:=.../orbslam3_backend*.so`: points to compiled Python/C++ bridge module used by `orb_vo_node.py` to call ORB-SLAM C++ APIs.

```bash
cd ORB_EKF
source install/setup.bash
ros2 launch orb_ekf orb_ekf.launch.py \
  vocabulary_file:=/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  orbslam_backend_library:=/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/orbslam3_backend.cpython-312-x86_64-linux-gnu.so
```

## TF notes

- `orb_vo_node` resolves TF from camera frame to `base_link` and converts ORB camera pose to base pose before publishing VO odom.
- EKF publishes the final `odom -> base_link` TF.
- If your bag already has camera-to-base TF, use only the default command above.
- If your bag does not provide camera-to-base TF, run the same launch command with extra static TF arguments:
- `use_static_camera_tf:=true`: enables launching a static TF publisher.
- `camera_frame:=oakd_left_camera_optical_frame`: camera frame name that will be connected to `base_link`.
- `static_tf_x/y/z`: translation values for `base_link -> camera_frame` (meters).
- `static_tf_qx/qy/qz/qw`: rotation quaternion for `base_link -> camera_frame`.
- Keep `vocabulary_file` and `orbslam_backend_library` the same as in the default command; only TF-related arguments are added.

```bash
cd ORB_EKF
source install/setup.bash
ros2 launch orb_ekf orb_ekf.launch.py \
  vocabulary_file:=/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/vendor/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  orbslam_backend_library:=/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/orbslam3_backend.cpython-312-x86_64-linux-gnu.so \
  use_static_camera_tf:=true \
  camera_frame:=oakd_left_camera_optical_frame \
  static_tf_x:=0.0 static_tf_y:=0.0 static_tf_z:=0.0 \
  static_tf_qx:=0.0 static_tf_qy:=0.0 static_tf_qz:=0.0 static_tf_qw:=1.0
```
