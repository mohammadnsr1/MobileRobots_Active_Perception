import math
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class OdomGoalDriver(Node):
    """
    ROS 2 node that accepts a goal in /odom frame and sends it to Nav2.

    Input topic:
      - /active_perception/nav_goal_odom   (geometry_msgs/msg/PoseStamped)

    Output topic:
      - /active_perception/nav_status      (std_msgs/msg/String)

    Behavior:
      - validates the incoming goal frame is 'odom'
      - forwards it to Nav2's navigate_to_pose action server
      - publishes status and feedback messages
      - can cancel an active goal if a newer one arrives
    """

    def __init__(self) -> None:
        super().__init__("odom_goal_driver")

        self.declare_parameter("goal_topic", "/active_perception/nav_goal_odom")
        self.declare_parameter("status_topic", "/active_perception/nav_status")
        self.declare_parameter("expected_frame", "odom")
        self.declare_parameter("action_name", "navigate_to_pose")
        self.declare_parameter("cancel_previous_goal", True)
        self.declare_parameter("server_wait_timeout_sec", 3.0)

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.expected_frame = str(self.get_parameter("expected_frame").value)
        self.action_name = str(self.get_parameter("action_name").value)
        self.cancel_previous_goal = bool(
            self.get_parameter("cancel_previous_goal").value
        )
        self.server_wait_timeout_sec = float(
            self.get_parameter("server_wait_timeout_sec").value
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, 10
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)

        self.current_goal_handle = None
        self.pending_goal: Optional[PoseStamped] = None
        self.navigation_in_progress = False

        self.publish_status(
            f"odom_goal_driver ready. Listening on {self.goal_topic} and forwarding to {self.action_name}."
        )

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def normalize_quaternion(self, q: Quaternion) -> Quaternion:
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if norm <= 1e-12:
            q.x = 0.0
            q.y = 0.0
            q.z = 0.0
            q.w = 1.0
            return q
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm
        return q

    def goal_callback(self, msg: PoseStamped) -> None:
        frame_id = (msg.header.frame_id or "").strip()
        if frame_id != self.expected_frame:
            self.publish_status(
                f"Rejected goal because frame_id='{frame_id}' but expected '{self.expected_frame}'."
            )
            return

        msg.pose.orientation = self.normalize_quaternion(msg.pose.orientation)

        if self.navigation_in_progress and self.cancel_previous_goal:
            self.publish_status("New goal received. Canceling active goal first.")
            self.pending_goal = msg
            if self.current_goal_handle is not None:
                cancel_future = self.current_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
            else:
                self.navigation_in_progress = False
                pending = self.pending_goal
                self.pending_goal = None
                if pending is not None:
                    self.send_goal(pending)
            return

        self.send_goal(msg)

    def cancel_done_callback(self, future) -> None:
        try:
            _ = future.result()
            self.publish_status("Previous Nav2 goal cancel request completed.")
        except Exception as exc:
            self.publish_status(f"Error while canceling previous goal: {exc}")

        self.navigation_in_progress = False
        self.current_goal_handle = None

        if self.pending_goal is not None:
            goal = self.pending_goal
            self.pending_goal = None
            self.send_goal(goal)

    def send_goal(self, pose_msg: PoseStamped) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=self.server_wait_timeout_sec):
            self.publish_status(
                f"Nav2 action server '{self.action_name}' is not available."
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg

        if (
            goal_msg.pose.header.stamp.sec == 0
            and goal_msg.pose.header.stamp.nanosec == 0
        ):
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.publish_status(
            "Sending goal in /odom to Nav2: x=%.3f, y=%.3f"
            % (
                goal_msg.pose.pose.position.x,
                goal_msg.pose.pose.position.y,
            )
        )

        self.navigation_in_progress = True
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.publish_status(
            "Nav2 feedback: remaining=%.3f m, recoveries=%d"
            % (
                float(feedback.distance_remaining),
                int(feedback.number_of_recoveries),
            )
        )

    def goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.navigation_in_progress = False
            self.publish_status(f"Failed to send goal to Nav2: {exc}")
            return

        if goal_handle is None or not goal_handle.accepted:
            self.navigation_in_progress = False
            self.publish_status("Nav2 rejected the goal.")
            return

        self.current_goal_handle = goal_handle
        self.publish_status("Nav2 accepted the goal.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future) -> None:
        self.navigation_in_progress = False
        self.current_goal_handle = None

        try:
            result_msg = future.result()
        except Exception as exc:
            self.publish_status(f"Error while waiting for Nav2 result: {exc}")
            return

        status = result_msg.status
        result = result_msg.result

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status("Nav2 goal succeeded.")
            return

        error_code = getattr(result, "error_code", None)
        error_msg = getattr(result, "error_msg", "")
        self.publish_status(
            f"Nav2 goal finished with status={status}, error_code={error_code}, error_msg='{error_msg}'"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomGoalDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()