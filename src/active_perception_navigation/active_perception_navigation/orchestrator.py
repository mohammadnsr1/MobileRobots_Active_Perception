from collections import deque
from typing import Deque, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class OrchestratorNode(Node):
    def __init__(self) -> None:
        super().__init__("ap_navigation_orchestrator")

        self.declare_parameter("target_pose_topic", "/active_perception/target_pose")
        self.declare_parameter(
            "confidence_topic", "/active_perception/navigation/confidence"
        )
        self.declare_parameter(
            "nbv_required_topic", "/active_perception/navigation/nbv_required"
        )
        self.declare_parameter(
            "nbv_goal_topic", "/active_perception/navigation/nbv_goal"
        )
        self.declare_parameter("robot_pose_topic", "/odom")
        self.declare_parameter(
            "status_topic", "/active_perception/navigation/orchestrator_status"
        )
        self.declare_parameter(
            "safe_to_navigate_topic",
            "/active_perception/navigation/safe_to_navigate",
        )
        self.declare_parameter("max_iterations", 6)
        self.declare_parameter("goal_frame", "odom")
        self.declare_parameter("reobserve_delay_sec", 2.0)
        self.declare_parameter("send_initial_goal", False)
        self.declare_parameter("nav_action_name", "navigate_to_pose")

        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.confidence_topic = self.get_parameter("confidence_topic").value
        self.nbv_required_topic = self.get_parameter("nbv_required_topic").value
        self.nbv_goal_topic = self.get_parameter("nbv_goal_topic").value
        self.robot_pose_topic = self.get_parameter("robot_pose_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.safe_to_navigate_topic = self.get_parameter("safe_to_navigate_topic").value
        self.max_iterations = int(self.get_parameter("max_iterations").value)
        self.goal_frame = self.get_parameter("goal_frame").value
        self.reobserve_delay_sec = float(
            self.get_parameter("reobserve_delay_sec").value
        )
        self.send_initial_goal = bool(self.get_parameter("send_initial_goal").value)
        self.nav_action_name = self.get_parameter("nav_action_name").value

        self.latest_target_pose: Optional[PoseStamped] = None
        self.latest_nbv_goal: Optional[PoseStamped] = None
        self.latest_confidence = 0.0
        self.latest_nbv_required = True
        self.safe_to_navigate = True
        self.navigation_in_progress = False
        self.completed_iterations = 0
        self.robot_pose_history: Deque[Odometry] = deque(maxlen=10)
        self.goal_handle = None

        self.target_pose_sub = self.create_subscription(
            PoseStamped, self.target_pose_topic, self.target_pose_callback, 10
        )
        self.confidence_sub = self.create_subscription(
            Float32, self.confidence_topic, self.confidence_callback, 10
        )
        self.nbv_required_sub = self.create_subscription(
            Bool, self.nbv_required_topic, self.nbv_required_callback, 10
        )
        self.nbv_goal_sub = self.create_subscription(
            PoseStamped, self.nbv_goal_topic, self.nbv_goal_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.robot_pose_topic, self.odom_callback, 10
        )
        self.safe_to_navigate_sub = self.create_subscription(
            Bool, self.safe_to_navigate_topic, self.safe_to_navigate_callback, 10
        )

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        self.get_logger().info(
            "Navigation orchestrator ready. Waiting for target pose, confidence, and NBV goals."
        )

    def publish_status(self, message: str) -> None:
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        self.get_logger().info(message)

    def odom_callback(self, msg: Odometry) -> None:
        self.robot_pose_history.append(msg)

    def target_pose_callback(self, msg: PoseStamped) -> None:
        self.latest_target_pose = msg

    def confidence_callback(self, msg: Float32) -> None:
        self.latest_confidence = float(msg.data)
        if self.latest_confidence >= 0.80 and not self.latest_nbv_required:
            self.publish_status(
                f"Active perception complete: confidence={self.latest_confidence:.3f}"
            )

    def nbv_required_callback(self, msg: Bool) -> None:
        self.latest_nbv_required = bool(msg.data)

    def safe_to_navigate_callback(self, msg: Bool) -> None:
        was_safe = self.safe_to_navigate
        self.safe_to_navigate = bool(msg.data)

        if self.safe_to_navigate or was_safe == self.safe_to_navigate:
            return

        self.publish_status(
            "Safety monitor reported SAFE_STOP; canceling current Nav2 goal."
        )
        if self.navigation_in_progress and self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)

    def nbv_goal_callback(self, msg: PoseStamped) -> None:
        self.latest_nbv_goal = msg

        if not self.safe_to_navigate:
            self.publish_status(
                "NBV goal received, but safety monitor is blocking autonomous navigation."
            )
            return

        if self.navigation_in_progress:
            self.publish_status(
                "Navigation already in progress; received NBV goal will be ignored for now."
            )
            return

        if self.completed_iterations >= self.max_iterations:
            self.publish_status(
                "Maximum active perception iterations reached; no further goals will be sent."
            )
            return

        if not self.latest_nbv_required and not self.send_initial_goal:
            self.publish_status(
                "NBV goal received, but current logic says no new viewpoint is required."
            )
            return

        self.send_navigation_goal(msg)

    def send_navigation_goal(self, pose: PoseStamped) -> None:
        if not self.safe_to_navigate:
            self.publish_status("Navigation blocked: robot is not in a safe state.")
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.publish_status(
                "Nav2 action server '%s' is not available." % self.nav_action_name
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        if not goal_msg.pose.header.frame_id:
            goal_msg.pose.header.frame_id = self.goal_frame
        if (
            goal_msg.pose.header.stamp.sec == 0
            and goal_msg.pose.header.stamp.nanosec == 0
        ):
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.navigation_in_progress = True
        self.completed_iterations += 1
        self.publish_status(
            "Sending Nav2 goal #%d to (%.3f, %.3f)"
            % (
                self.completed_iterations,
                goal_msg.pose.pose.position.x,
                goal_msg.pose.pose.position.y,
            )
        )

        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.publish_status(
            "Navigation feedback: remaining distance %.3f"
            % float(feedback.distance_remaining)
        )

    def goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.navigation_in_progress = False
            self.publish_status("Nav2 goal was rejected.")
            return

        self.goal_handle = goal_handle
        self.publish_status("Nav2 goal accepted.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def cancel_done_callback(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            self.publish_status(f"Failed to cancel Nav2 goal cleanly: {exc}")
            return

        self.publish_status("Nav2 goal cancel request sent.")

    def goal_result_callback(self, future) -> None:
        self.navigation_in_progress = False
        self.goal_handle = None
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status(
                "Navigation succeeded. Waiting %.1f s before next observation cycle."
                % self.reobserve_delay_sec
            )
        elif status == GoalStatus.STATUS_ABORTED:
            self.publish_status("Navigation aborted by Nav2.")
        elif status == GoalStatus.STATUS_CANCELED:
            self.publish_status("Navigation canceled.")
        else:
            self.publish_status(f"Navigation finished with status code {status}.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
