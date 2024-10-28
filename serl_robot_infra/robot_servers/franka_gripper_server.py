import rospy
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from sensor_msgs.msg import JointState
import numpy as np

from robot_servers.gripper_server import GripperServer


class FrankaGripperServer(GripperServer):
    def __init__(self):
        super().__init__()
        self.grippermovepub = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self.grippergrasppub = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.gripper_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self._update_gripper
        )
        self.binary_gripper_pose = 0

    def open(self):
        if self.binary_gripper_pose == 0:
            return
        msg = MoveActionGoal()
        # msg.goal.width = 0.025
        msg.goal.width = 0.09
        msg.goal.speed = 0.3
        self.grippermovepub.publish(msg)
        self.binary_gripper_pose = 0

    def close(self):
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.3
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)
        self.binary_gripper_pose = 1

    def close_slow(self):
        if self.binary_gripper_pose == 1:
            return
        msg = GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.1
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)
        self.binary_gripper_pose = 1

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 255]"""
        msg = MoveActionGoal()
        msg.goal.width = float(position / (255 * 10))  # width in [0, 0.1]m
        msg.goal.speed = 0.3
        self.grippermovepub.publish(msg)

    def _update_gripper(self, msg):
        """internal callback to get the latest gripper position."""
        self.gripper_pos = np.sum(msg.position) / 0.08
