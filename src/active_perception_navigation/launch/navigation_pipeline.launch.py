from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_pose_topic", default_value="/active_perception/target_pose"
            ),
            DeclareLaunchArgument(
                "robot_pose_topic", default_value="/odom"
            ),
            DeclareLaunchArgument(
                "confidence_topic",
                default_value="/active_perception/navigation/confidence",
            ),
            DeclareLaunchArgument(
                "nbv_required_topic",
                default_value="/active_perception/navigation/nbv_required",
            ),
            DeclareLaunchArgument(
                "nbv_goal_topic",
                default_value="/active_perception/navigation/nbv_goal",
            ),
            DeclareLaunchArgument(
                "safe_to_navigate_topic",
                default_value="/active_perception/navigation/safe_to_navigate",
            ),
            DeclareLaunchArgument(
                "nav_action_name", default_value="navigate_to_pose"
            ),
            Node(
                package="active_perception_navigation",
                executable="ap_nav_confidence_evaluator",
                name="ap_nav_confidence_evaluator",
                output="screen",
                parameters=[
                    {
                        "input_pose_topic": LaunchConfiguration("target_pose_topic"),
                        "output_confidence_topic": LaunchConfiguration(
                            "confidence_topic"
                        ),
                        "output_nbv_topic": LaunchConfiguration("nbv_required_topic"),
                    }
                ],
            ),
            Node(
                package="active_perception_navigation",
                executable="ap_nav_nbv_planner",
                name="ap_nav_nbv_planner",
                output="screen",
                parameters=[
                    {
                        "target_pose_topic": LaunchConfiguration("target_pose_topic"),
                        "robot_pose_topic": LaunchConfiguration("robot_pose_topic"),
                        "confidence_topic": LaunchConfiguration("confidence_topic"),
                        "nbv_required_topic": LaunchConfiguration(
                            "nbv_required_topic"
                        ),
                        "goal_topic": LaunchConfiguration("nbv_goal_topic"),
                    }
                ],
            ),
            Node(
                package="active_perception_navigation",
                executable="ap_nav_safety_monitor",
                name="ap_nav_safety_monitor",
                output="screen",
                parameters=[
                    {
                        "robot_pose_topic": LaunchConfiguration("robot_pose_topic"),
                        "safe_to_navigate_topic": LaunchConfiguration(
                            "safe_to_navigate_topic"
                        ),
                    }
                ],
            ),
            Node(
                package="active_perception_navigation",
                executable="ap_nav_orchestrator",
                name="ap_nav_orchestrator",
                output="screen",
                parameters=[
                    {
                        "target_pose_topic": LaunchConfiguration("target_pose_topic"),
                        "robot_pose_topic": LaunchConfiguration("robot_pose_topic"),
                        "confidence_topic": LaunchConfiguration("confidence_topic"),
                        "nbv_required_topic": LaunchConfiguration(
                            "nbv_required_topic"
                        ),
                        "nbv_goal_topic": LaunchConfiguration("nbv_goal_topic"),
                        "safe_to_navigate_topic": LaunchConfiguration(
                            "safe_to_navigate_topic"
                        ),
                        "nav_action_name": LaunchConfiguration("nav_action_name"),
                    }
                ],
            ),
        ]
    )
