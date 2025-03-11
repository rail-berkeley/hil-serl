"""Gym Interface for Franka"""

import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
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
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64
from std_msgs.msg import Header

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
                break

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


class AgilexEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.topic_prefix = config.ARM
        self.curr_joint = []
        self.curr_end_pos = np.full(7, 0.0, dtype=np.float32)
        self.curr_end_pos[6] = 1.0
        self.curr_vel = np.full(6, 0.0, dtype=np.float32)
        self.curr_gripper_pos = [0.0]
        self.wrist_img = {}
        self.cv_bridge = CvBridge()
        self.control_arm = False
        if not fake_env:
            self.control_arm = True
            self.init_pub_sub_node()
            self.ik_solver = IK("base_link", "link6", urdf_string=open("/home/agilex/ChaseWang/ws/Piper_ros/src/piper_description/urdf/piper_description.urdf", 'r').read())
            self.pinco_ik_solver = Arm_IK(urdf_path="")
        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = (
            config.JOINT_RESET_PERIOD
        )  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
                        for key in config.CAMERAS
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        # if not fake_env:
        #    from pynput import keyboard
        #    self.terminate = False
        #    def on_press(key):
        #        if key == keyboard.Key.esc:
        #            self.terminate = True
        #    self.listener = keyboard.Listener(on_press=on_press)
        #    self.listener.start()

        print("Initialized Agilex")

    def init_pub_sub_node(self):
        print("Init node")
        if rospy.get_node_uri():
            print("Node has been initialzed already")
        else:
            rospy.init_node(f"hilserl_{self.topic_prefix}_agilex_env", anonymous=True)
        self.joint_sub = rospy.Subscriber(
            self.topic_prefix + "/joint_states", JointState, self.update_curr_joint
        )
        self.end_pose_euler_sub = rospy.Subscriber(
            self.topic_prefix + "/end_pose_quat", Pose, self.update_curr_end_pos
        )
        self.gripper_state_sub = rospy.Subscriber(
            f"{self.topic_prefix}/gripper_angle", Float64, self.update_gripper_state
        )
        self.cameras_subscrib = {}
        for key, values in self.config.CAMERAS.items():
            self.cameras_subscrib[key] = rospy.Subscriber(
                values,
                Image,
                partial(self.img_callback, key=key),
                queue_size=1000,
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
        #print(f"joint state: {self.curr_joint}")

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
        #joint_angle = self.compute_ik(end_pose)
        #print(f"arm {self.config.ARM} pos: {end_pose}\njoint_recv: {self.curr_joint} \njoint_com: {joint_angle}")

    def update_gripper_state(self, gripper_state):
        #print(f"get gripper val {gripper_state}")
        self.curr_gripper_pos = [gripper_state.data]

    def img_callback(self, image, key):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        #print(f"image shape {key} {cv_image.shape}")
        self.wrist_img[key] = cv_image
        #if key not in self.wrist_img:
        #    self.wrist_img[key] = [cv_image]
        #else:
        #    self.wrist_img[key].append(cv_image)

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.curr_end_pos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.curr_end_pos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]
        if self.control_arm:
            self._send_gripper_command(gripper_action)
            cliped_pos = self.clip_safety_box(self.nextpos)
            print(f"send step action: {cliped_pos}")
            self._send_pos_command(cliped_pos)

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        print(f"actual pos: {self.curr_end_pos}")
        self._update_curr_end_pos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length or reward or self.terminate
        )
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(
            np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler])
        )
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        pass

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        path = np.linspace(self.curr_end_pos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._send_pos_command(self.curr_end_pos)
        time.sleep(0.3)
        #requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            #requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        #requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        # TODO: update param
        #requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if (
            self.joint_reset_cycle != 0
            and self.cycle_count % self.joint_reset_cycle == 0
        ):
            self.cycle_count = 0
            joint_reset = True

        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0

        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists("./videos"):
                    os.makedirs("./videos")

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f"./videos/left_{camera_key}_{timestamp}.mp4"
                    else:
                        video_path = f"./videos/right_{camera_key}_{timestamp}.mp4"

                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        pass

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        if not self.control_arm:
            return
        #print(f"arm {self.config.ARM} send pose command {arr}")
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
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        joint_state_msg.position = joint
        self.joint_pub.publish(joint_state_msg)

    def _send_gripper_command(self, pos: float):
        """Internal function to send gripper command to the robot."""
        gripper_msg = pos
        self.gripper_pub.publish(gripper_msg)

    def _update_curr_end_pos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        # FIXME: resolve data race problem
        pass

    def _get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.curr_end_pos,
            "tcp_vel": self.curr_vel,
            "gripper_pose": self.curr_gripper_pos,
            # "tcp_force": self.currforce,
            # "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=self.wrist_img, state=state_observation))

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
