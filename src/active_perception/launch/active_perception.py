from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    tf_remappings = [
        ("/tf", "/robot_10/tf"),
        ("/tf_static", "/robot_10/tf_static"),
    ]

    return LaunchDescription(
        [
            Node(
                package="active_perception",
                executable="box_finder",
                name="box_finder",
                output="screen",
            ),
            Node(
                package="active_perception",
                executable="pose_estimator",
                name="pose_estimator",
                output="screen",
                remappings=tf_remappings,
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                remappings=tf_remappings,
            ),
            Node(
                package="active_perception",
                executable="nbv_planner",
                name="nbv_planner",
                output="screen",
                remappings=tf_remappings,
            ),
            Node(
                package="active_perception",
                executable="orchestrator",
                name="orchestrator",
                output="screen",
                parameters=[
                    {
                        "robot_pose_topic": "/robot_10/odom",
                    }
                ],
            ),
            Node(
                package="active_perception",
                executable="confidence_evaluator",
                name="confidence_evaluator",
                output="screen",
            ),
        ]
    )
