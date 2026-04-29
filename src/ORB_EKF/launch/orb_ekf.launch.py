from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    orb_vo_params_file = PathJoinSubstitution([FindPackageShare("orb_ekf"), "config", "orb_vo.yaml"])

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
            DeclareLaunchArgument("vo_odom_topic", default_value="/orb_slam/vo_odom"),
            DeclareLaunchArgument("vo_path_topic", default_value="/orb_slam/vo_path"),
            DeclareLaunchArgument("average_path_topic", default_value="/orb_ekf/average_path"),
            DeclareLaunchArgument("world_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("publish_vo_tf", default_value="false"),
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
                        "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                        "world_frame": LaunchConfiguration("world_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "publish_vo_tf": LaunchConfiguration("publish_vo_tf"),
                        "vo_odom_topic": LaunchConfiguration("vo_odom_topic"),
                        "vo_path_topic": LaunchConfiguration("vo_path_topic"),
                    },
                ],
            ),
            Node(
                package="orb_ekf",
                executable="fused_output_node",
                name="orb_ekf_fused_output_node",
                output="screen",
                parameters=[
                    {
                        "vo_odom_topic": LaunchConfiguration("vo_odom_topic"),
                        "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                        "output_path_topic": LaunchConfiguration("average_path_topic"),
                        "world_frame": LaunchConfiguration("world_frame"),
                    }
                ],
            ),
        ]
    )
