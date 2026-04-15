from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    orb_vo_params_file = PathJoinSubstitution([FindPackageShare("orb_ekf"), "config", "orb_vo.yaml"])
    ekf_params_file = PathJoinSubstitution([FindPackageShare("orb_ekf"), "config", "ekf.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("vocabulary_file", default_value=""),
            DeclareLaunchArgument("settings_file", default_value=""),
            DeclareLaunchArgument("orbslam_backend_library", default_value=""),
            DeclareLaunchArgument("left_image_topic", default_value="/robot_10/oakd/left/image_raw"),
            DeclareLaunchArgument("right_image_topic", default_value="/robot_10/oakd/right/image_raw"),
            DeclareLaunchArgument("left_camera_info_topic", default_value="/robot_10/oakd/left/camera_info"),
            DeclareLaunchArgument("right_camera_info_topic", default_value="/robot_10/oakd/right/camera_info"),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/robot_10/odom"),
            DeclareLaunchArgument("orb_odom_topic", default_value="/orb_slam/vo_odom"),
            DeclareLaunchArgument("ekf_output_topic", default_value="/odometry/filtered"),
            DeclareLaunchArgument("world_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("publish_vo_tf", default_value="false"),
            DeclareLaunchArgument("use_static_camera_tf", default_value="false"),
            DeclareLaunchArgument("camera_frame", default_value="oakd_left_camera_optical_frame"),
            DeclareLaunchArgument("static_tf_x", default_value="0.0"),
            DeclareLaunchArgument("static_tf_y", default_value="0.0"),
            DeclareLaunchArgument("static_tf_z", default_value="0.0"),
            DeclareLaunchArgument("static_tf_qx", default_value="0.0"),
            DeclareLaunchArgument("static_tf_qy", default_value="0.0"),
            DeclareLaunchArgument("static_tf_qz", default_value="0.0"),
            DeclareLaunchArgument("static_tf_qw", default_value="1.0"),
            Node(
                package="orb_ekf",
                executable="orb_vo_node",
                name="orb_stereo_vo_node",
                output="screen",
                parameters=[
                    orb_vo_params_file,
                    {
                        "vocabulary_file": LaunchConfiguration("vocabulary_file"),
                        "settings_file": LaunchConfiguration("settings_file"),
                        "orbslam_backend_library": LaunchConfiguration("orbslam_backend_library"),
                        "left_image_topic": LaunchConfiguration("left_image_topic"),
                        "right_image_topic": LaunchConfiguration("right_image_topic"),
                        "left_camera_info_topic": LaunchConfiguration("left_camera_info_topic"),
                        "right_camera_info_topic": LaunchConfiguration("right_camera_info_topic"),
                        "world_frame": LaunchConfiguration("world_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                        "publish_vo_tf": LaunchConfiguration("publish_vo_tf"),
                        "vo_odom_topic": LaunchConfiguration("orb_odom_topic"),
                    },
                ],
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[
                    ekf_params_file,
                    {
                        "world_frame": LaunchConfiguration("world_frame"),
                        "base_link_frame": LaunchConfiguration("base_frame"),
                        "odom0": LaunchConfiguration("wheel_odom_topic"),
                        "odom1": LaunchConfiguration("orb_odom_topic"),
                    },
                ],
                remappings=[("odometry/filtered", LaunchConfiguration("ekf_output_topic"))],
            ),
            Node(
                package="orb_ekf",
                executable="fused_output_node",
                name="orb_ekf_fused_output_node",
                output="screen",
                parameters=[{"input_odom_topic": LaunchConfiguration("ekf_output_topic")}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="orb_camera_to_base_static_tf",
                arguments=[
                    LaunchConfiguration("static_tf_x"),
                    LaunchConfiguration("static_tf_y"),
                    LaunchConfiguration("static_tf_z"),
                    LaunchConfiguration("static_tf_qx"),
                    LaunchConfiguration("static_tf_qy"),
                    LaunchConfiguration("static_tf_qz"),
                    LaunchConfiguration("static_tf_qw"),
                    LaunchConfiguration("base_frame"),
                    LaunchConfiguration("camera_frame"),
                ],
                condition=IfCondition(LaunchConfiguration("use_static_camera_tf")),
            ),
        ]
    )
