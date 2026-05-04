import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class OdomController(Node):
    def __init__(self) -> None:
        super().__init__("odom_controller")

        self.declare_parameter("goal_topic", "/robot_10/active_perception/nav_goal_odom")
        self.declare_parameter("status_topic", "/robot_10/active_perception/nav_status")
        self.declare_parameter("pose_feedback_topic", "/robot_10/odom")
        self.declare_parameter("cmd_vel_topic", "/robot_10/cmd_vel")
        self.declare_parameter("expected_frame", "odom")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_linear_speed", 0.25)
        self.declare_parameter("max_angular_speed", 1.2)
        self.declare_parameter("kp_linear", 0.8)
        self.declare_parameter("kp_angular", 2.5)
        self.declare_parameter("kd_angular", 0.2)
        self.declare_parameter("goal_position_tolerance", 0.08)
        self.declare_parameter("goal_yaw_tolerance", 0.1)
        self.declare_parameter("rotate_in_place_angle_threshold", 0.35)
        self.declare_parameter("slowdown_radius", 0.5)

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.pose_feedback_topic = str(
            self.get_parameter("pose_feedback_topic").value
        )
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.expected_frame = str(self.get_parameter("expected_frame").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.kp_linear = float(self.get_parameter("kp_linear").value)
        self.kp_angular = float(self.get_parameter("kp_angular").value)
        self.kd_angular = float(self.get_parameter("kd_angular").value)
        self.goal_position_tolerance = float(
            self.get_parameter("goal_position_tolerance").value
        )
        self.goal_yaw_tolerance = float(
            self.get_parameter("goal_yaw_tolerance").value
        )
        self.rotate_in_place_angle_threshold = float(
            self.get_parameter("rotate_in_place_angle_threshold").value
        )
        self.slowdown_radius = float(self.get_parameter("slowdown_radius").value)

        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.pose_feedback_topic, self.odom_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, self.cmd_vel_topic, 10
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.current_goal: Optional[PoseStamped] = None
        self.latest_odom: Optional[Odometry] = None
        self.last_heading_error = 0.0
        self.last_control_time = None
        self.goal_reached_announced = False

        timer_period = 1.0 / max(self.control_rate_hz, 1.0)
        self.control_timer = self.create_timer(timer_period, self.control_loop)

        self.publish_status(
            "odom_controller ready. goal_topic=%s pose_feedback_topic=%s "
            "cmd_vel_topic=%s expected_frame=%s"
            % (
                self.goal_topic,
                self.pose_feedback_topic,
                self.cmd_vel_topic,
                self.expected_frame,
            )
        )

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.linear.y = 0.0
        cmd.twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    def quaternion_to_yaw(self, orientation) -> float:
        qx = float(orientation.x)
        qy = float(orientation.y)
        qz = float(orientation.z)
        qw = float(orientation.w)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg: Odometry) -> None:
        frame_id = (msg.header.frame_id or "").strip()
        if frame_id and frame_id != self.expected_frame:
            self.get_logger().warn(
                "Pose feedback frame_id='%s' does not match expected '%s'."
                % (frame_id, self.expected_frame)
            )
        self.latest_odom = msg

    def goal_callback(self, msg: PoseStamped) -> None:
        frame_id = (msg.header.frame_id or "").strip()
        if frame_id != self.expected_frame:
            self.publish_status(
                "Rejected goal because frame_id='%s' but expected '%s'."
                % (frame_id, self.expected_frame)
            )
            return

        self.current_goal = msg
        self.last_heading_error = 0.0
        self.last_control_time = None
        self.goal_reached_announced = False
        self.publish_status(
            "Accepted odom goal: x=%.3f, y=%.3f"
            % (msg.pose.position.x, msg.pose.position.y)
        )

    def control_loop(self) -> None:
        if self.current_goal is None:
            return

        if self.latest_odom is None:
            self.stop_robot()
            return

        now = self.get_clock().now()
        current_time = now.nanoseconds / 1e9
        dt = 0.0
        if self.last_control_time is not None:
            dt = max(current_time - self.last_control_time, 1e-6)
        self.last_control_time = current_time

        robot_pose = self.latest_odom.pose.pose
        robot_x = float(robot_pose.position.x)
        robot_y = float(robot_pose.position.y)
        robot_yaw = self.quaternion_to_yaw(robot_pose.orientation)

        goal_pose = self.current_goal.pose
        goal_x = float(goal_pose.position.x)
        goal_y = float(goal_pose.position.y)
        goal_yaw = self.quaternion_to_yaw(goal_pose.orientation)

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance_error = math.hypot(dx, dy)
        heading_to_goal = math.atan2(dy, dx)
        heading_error = self.wrap_angle(heading_to_goal - robot_yaw)
        final_yaw_error = self.wrap_angle(goal_yaw - robot_yaw)

        if distance_error <= self.goal_position_tolerance:
            if abs(final_yaw_error) <= self.goal_yaw_tolerance:
                self.stop_robot()
                if not self.goal_reached_announced:
                    self.goal_reached_announced = True
                    self.publish_status("Odom goal reached.")
                self.current_goal = None
                return

            angular_cmd = self.clamp(
                self.kp_angular * final_yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
            self.publish_cmd(0.0, angular_cmd)
            return

        angular_derivative = 0.0
        if dt > 0.0:
            angular_derivative = (heading_error - self.last_heading_error) / dt
        self.last_heading_error = heading_error

        angular_cmd = (
            self.kp_angular * heading_error + self.kd_angular * angular_derivative
        )
        angular_cmd = self.clamp(
            angular_cmd, -self.max_angular_speed, self.max_angular_speed
        )

        linear_cmd = self.kp_linear * distance_error
        if distance_error < self.slowdown_radius:
            linear_cmd *= distance_error / max(self.slowdown_radius, 1e-6)

        if abs(heading_error) > self.rotate_in_place_angle_threshold:
            linear_cmd = 0.0

        linear_cmd = self.clamp(linear_cmd, 0.0, self.max_linear_speed)

        self.publish_cmd(linear_cmd, angular_cmd)

    def destroy_node(self):
        self.stop_robot()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
