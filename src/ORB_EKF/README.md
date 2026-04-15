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

## Build

```bash
colcon build --packages-select orb_ekf
source install/setup.bash
```

Dependencies are declared in [package.xml](/home/vikas-narang/RAS598_Assignments/RAS598_Mobile_Robotics/VO/ORB_EKF/package.xml), including `robot_localization`, `launch_ros`, `tf2_ros`, `cv_bridge`, and message packages.

For standalone use, `orbslam3_backend` must also be available:
- either importable in your Python environment (`import orbslam3_backend`)
- or passed as a compiled shared library path via launch argument `orbslam_backend_library:=/abs/path/orbslam3_backend*.so`

## Run

```bash
ros2 launch orb_ekf orb_ekf.launch.py \
  vocabulary_file:=/absolute/path/to/ORBvoc.txt \
  orbslam_backend_library:=/absolute/path/to/orbslam3_backend.cpython-312-x86_64-linux-gnu.so
```

## TF notes

- `orb_vo_node` resolves TF from camera frame to `base_link` and converts ORB camera pose to base pose before publishing VO odom.
- EKF publishes the final `odom -> base_link` TF.
- If your bag does not provide camera-to-base static TF, launch with:

```bash
ros2 launch orb_ekf orb_ekf.launch.py \
  vocabulary_file:=/absolute/path/to/ORBvoc.txt \
  use_static_camera_tf:=true \
  camera_frame:=oakd_left_camera_optical_frame \
  static_tf_x:=0.0 static_tf_y:=0.0 static_tf_z:=0.0 \
  static_tf_qx:=0.0 static_tf_qy:=0.0 static_tf_qz:=0.0 static_tf_qw:=1.0
```
