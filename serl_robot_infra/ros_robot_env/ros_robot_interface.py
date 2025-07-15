import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from functools import partial
from cv_bridge import CvBridge

import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist, WrenchStamped, Vector3
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import Header
from colorama import Fore, Style
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                time.sleep(0.01)
                continue

            frame = np.concatenate(
                [
                    cv2.resize(v, (128, 128))
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    ARM = "left"
    CAMERAS: Dict = {
        "wrist": "127122270572",
    }
    DEPTH_CAMERAS: Dict = {}
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class RobotRosInterface:
    def __init__(self, config):
        self.config = config
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.gripper_open = config.gripper_open
        self.gripper_close = config.gripper_close
        self.topic_prefix = config.ARM
        self.grip_status = 0
        self.curr_joint = []
        self.curr_end_pos = np.full(7, 0.0, dtype=np.float32)
        self.curr_end_pos[6] = 1.0
        self.curr_vel = np.full(6, 0.0, dtype=np.float32)
        self.curr_force = [0.0] * 6
        self.curr_gripper_pos = [0.0]
        self.wrist_img = {}
        self.cv_bridge = CvBridge()
        self.rgb_cam_dict = config.CAMERAS
        self.depth_cam_dict = config.DEPTH_CAMERAS

        self.init_pub_sub_node()
        print("Initialized Agilex")

    def init_pub_sub_node(self):
        print("Init node")
        if rospy.get_node_uri():
            print("Node has been initialzed already")
        else:
            rospy.init_node(f"hilserl_robot_interface", anonymous=True)
        self.joint_sub = rospy.Subscriber(
            self.topic_prefix + "/joint_states", JointState, self.update_curr_joint
        )
        self.end_pose_euler_sub = rospy.Subscriber(
            self.topic_prefix + "/end_pose_quat", Pose, self.update_curr_end_pos
        )
        self.end_pose_vel_sub = rospy.Subscriber(
            self.topic_prefix + "/end_pose_vel", Twist, self.update_curr_end_vel
        )
        self.end_force_sub = rospy.Subscriber(
            self.topic_prefix + "/end_force", WrenchStamped, self.update_curr_end_force
        )
        self.gripper_state_sub = rospy.Subscriber(
            f"{self.topic_prefix}/gripper_angle", Float64, self.update_gripper_state
        )
        self.gripper_grip_status = rospy.Subscriber(
            f"{self.topic_prefix}/gripper_grip_status", Int32, self.update_grip_status
        )
        self.cameras_subscrib = {}
        for key, values in self.rgb_cam_dict.items():
            self.cameras_subscrib[key] = rospy.Subscriber(
                values,
                Image,
                partial(self.img_callback, key=key),
                queue_size=1,
                tcp_nodelay=True,
            )

        for key, values in self.depth_cam_dict.items():
            print(f"sub cam {key}, {values}")
            self.cameras_subscrib[key] = rospy.Subscriber(
                values,
                Image,
                partial(self.depth_img_callback, key=key),
                queue_size=1,
                tcp_nodelay=True,
            )
        self.end_pose_pub = rospy.Publisher(
            self.topic_prefix + "/control_pose", Pose, queue_size=1
        )
        self.gripper_pub = rospy.Publisher(
            f"{self.topic_prefix}/control_gripper", Float64, queue_size=1
        )
        self.joint_pub = rospy.Publisher(
            self.topic_prefix + "/control_joint", JointState, queue_size=1
        )

    def update_curr_joint(self, joint_state):
        self.curr_joint = list(joint_state.position)
        # print(f"joint state: {self.curr_joint}")

    def update_curr_end_pos(self, pose_rot):
        end_pose = []
        end_pose.append(pose_rot.position.x)
        end_pose.append(pose_rot.position.y)
        end_pose.append(pose_rot.position.z)
        end_pose.append(pose_rot.orientation.x)
        end_pose.append(pose_rot.orientation.y)
        end_pose.append(pose_rot.orientation.z)
        end_pose.append(pose_rot.orientation.w)
        self.curr_end_pos = np.array(end_pose)
        # joint_angle = self.compute_ik(end_pose)
        # print(f"pos: {end_pose}\n")

    def update_curr_end_vel(self, tcp_vel):
        self.curr_vel[:3] = tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z
        self.curr_vel[3:] = tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z

    def update_curr_end_force(self, tcp_force):
        self.curr_force = [
            tcp_force.wrench.force.x,
            tcp_force.wrench.force.y,
            tcp_force.wrench.force.z,
            tcp_force.wrench.torque.x,
            tcp_force.wrench.torque.y,
            tcp_force.wrench.torque.z,
        ]

    def update_gripper_state(self, gripper_state):
        # print(f"get gripper val {gripper_state}")
        factor = 0.01
        range = max(self.gripper_open - self.gripper_close, 1)
        self.curr_gripper_pos = [
            np.clip((gripper_state.data - self.gripper_close) / range, 0.0, 1.0)
        ]

    def update_grip_status(self, status):
        self.grip_status = status.data

    def img_callback(self, image, key):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        cropped_rgb = (
            self.config.IMAGE_CROP[key](cv_image)
            if key in self.config.IMAGE_CROP
            else cv_image
        )
        #TODO(Chao): resize in robot env
        resized = cv2.resize(cropped_rgb, (128, 128))
        self.wrist_img[key] = resized

    def depth_img_callback(self, image, key):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        cropped_rgb = (
            self.config.IMAGE_CROP[key](cv_image)
            if key in self.config.IMAGE_CROP
            else cv_image
        )
        resized = cv2.resize(cropped_rgb, (128, 128))
        scaled_depth = cv2.convertScaleAbs(resized, alpha=0.03)  # alpha控制对比度
        pseudo_rgb = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)  #
        # resized = cv2.cvtColor(pseudo_rgb, cv2.COLOR_GRAY2BGR)
        self.wrist_img[key] = pseudo_rgb

    def send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = pos[:3]
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = pos[3:]
        self.end_pose_pub.publish(pose)

    def send_joint_command(self, joint: np.ndarray):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
        ]
        joint_state_msg.position = joint
        self.joint_pub.publish(joint_state_msg)

    def send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (
                (pos <= -0.5)
                # and (self.curr_gripper_pos[0] > 0.85)
                # and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):  # close gripper
                # specified for nong fu shan quan bottole
                self.gripper_pub.publish(self.gripper_close)
                self.last_gripper_act = time.time()
            elif (
                pos
                >= 0.5
                # and (self.curr_gripper_pos[0] < 0.85)
            ):  # open gripper
                self.gripper_pub.publish(self.gripper_open)
                self.last_gripper_act = time.time()
            else:
                return
        else:
            self.gripper_pub.publish(pos)

    def get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.curr_end_pos,
            "tcp_vel": self.curr_vel,
            "gripper_pose": self.curr_gripper_pos,
            # "tcp_force": self.currforce,
            # "tcp_torque": self.currtorque,
            "tcp_force": self.curr_force[:3],
            "tcp_torque": self.curr_force[3:],
            "grip_status": self.grip_status,
        }
        return copy.deepcopy(dict(images=self.wrist_img, state=state_observation))
