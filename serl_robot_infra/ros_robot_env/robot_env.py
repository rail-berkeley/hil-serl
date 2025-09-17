"""Gym Interface for Franka"""

import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Slerp
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict
from colorama import Fore, Style
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from franka_env.utils.transformations import calculate_tool_tip_with_matrix
# 添加ROS相关导入
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ros_robot_env.ros_robot_interface import RobotRosInterface
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d



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
                    cv2.resize(v, (256, 256))
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


class RobotRosEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.robot = RobotRosInterface(config)
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.control_arm = False
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
                        "grip_status": gym.spaces.Discrete(4), 
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in {**config.CAMERAS, **config.DEPTH_CAMERAS}
                    }
                ),
            }
        )
        self.cycle_count = 0
        if not fake_env:
            self.img_queue = queue.Queue(maxsize=1)
            self.dis_images = {}
            if self.display_image:
                self.displayer = ImageDisplayer(self.img_queue, "visualizer")
                self.displayer.start()
            self.control_arm = True
        if fake_env:
            return
        self.display_img_id = 0
        print("Initialized Agilex")

        # 初始化用于新相机的变量
        self.new_camera_image = None
        self.cv_bridge = CvBridge()
        self.new_camera_subscriber = None

        # 只在非fake环境中初始化相机订阅
        if not fake_env:
            self._init_new_camera_subscriber()

        # self.robot.reset_robot_force_config()

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        # print(
        #    Fore.RED
        #    + f" {self.xyz_bounding_box.low} "
        #    + Fore.BLUE
        #    + f" {pose[:3]} "
        #    + Fore.RED
        #    + f" {self.xyz_bounding_box.high}"
        #    + Style.RESET_ALL
        # )
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )

        
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        if any(np.abs(euler) > np.pi):
           print(f"big euler {euler}")
        # print(
        #    Fore.RED
        #    + f" {self.rpy_bounding_box.low} "
        #    + Fore.BLUE
        #    + f" {euler} "
        #    + Fore.RED
        #    + f" {self.rpy_bounding_box.high}"
        #    + Style.RESET_ALL
        # )
        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )
        euler[1] = np.clip(
            euler[1], self.rpy_bounding_box.low[1], self.rpy_bounding_box.high[1]
        )
        sign2 = np.sign(euler[2])
        euler[2] = sign2 * (
            np.clip(
                np.abs(euler[2]),
                self.rpy_bounding_box.low[2],
                self.rpy_bounding_box.high[2],
            )
        )
        # print(f"cliped euler: {euler}")
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.nextpos = self.robot.curr_end_pos.copy()
        # print(f"current pos in step: {self.nextpos}")
        # self.nextpos[:3] = actioin[:3]
        # self.nextpos[3:] = euler_2_quat(action[3:6])
        
        try:
            action = np.clip(action, -1.0, 1.0)
            xyz_delta = action[:3]
            self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
            ## GET ORIENTATION FROM ACTION
            # euler_zxy = np.array([action[5], action[3], action[4]])
            self.nextpos[3:] = (
                R.from_euler("xyz", action[3:6] * self.action_scale[1])
                * R.from_quat(self.nextpos[3:])
            ).as_quat()

            gripper_action = action[6] * self.action_scale[2]
            gripper_action = -1.0
            # print(f"pos before clip: {self.nextpos}")
            if self.control_arm:
                self.robot.send_gripper_command(gripper_action)
                cliped_pos = self.clip_safety_box(self.nextpos)
                # print(f"send step action: {cliped_pos}")
                self.robot.send_pos_command(cliped_pos)
        except Exception as e:
            print(f"calculate action error {action}, exception {e}")
        # print("action", xyz_delta)
        # print("current pos", self.robot.curr_end_pos)
        # print("goal pos", cliped_pos)
        self.curr_path_length += 1
        dt = time.time() - start_time
        
        time.sleep(max(0, (1.0 / self.hz) - dt-0.03))
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = (
            self.curr_path_length >= self.max_episode_length or self.terminate
        )
        # print(ob,'*'*200000)
        return ob, int(reward), False, done, {"succeed": reward}

    def compute_reward(self, obs):
        #tool_tip_pose = calculate_tool_tip_with_matrix(obs["state"]["tcp_pose"])
        ## # if tool_tip_pose[1] < 0.02308:
        ## #     return 1
        #print(f"tool tip pose: {tool_tip_pose}")
        #if np.abs(tool_tip_pose[0] - 0.6705) < 0.01:
        #    return 1
        return 0



    # def interpolate_move(self, goal: np.ndarray, timeout: float):
    #     """Move the robot to the goal position with linear interpolation."""
    #     if goal.shape == (6,):
    #         goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
    #         print("pos+quat",goal)
    #     key_times = [0, 1]
    #     key_rots = R.from_quat([self.robot.curr_end_pos[3:], goal[3:]])
    #     slerp = Slerp(key_times, key_rots)
    #     steps = int(timeout * self.hz)
    #     interpolate_times = np.linspace(0, 1, steps)
    #     path_slerp = slerp(interpolate_times)
    #     path_slerp = [rot.as_quat() for rot in path_slerp]
    #     path = np.linspace(self.robot.curr_end_pos, goal, steps)
    #     for idx, p in enumerate(path):
    #         p[3:] = path_slerp[idx]
    #         # print(f"reset p {path_slerp[idx]}")
    #         self.robot.send_pos_command(p)
    #         time.sleep(1 / self.hz)
    #     self.nextpos = p


    # def go_to_reset(self, joint_reset=False):
    #     """
    #     The concrete steps to perform reset should be
    #     implemented each subclass for the specific task.
    #     Should override this method if custom reset procedure is needed.
    #     """
    #     # self.robot.send_gripper_command(1)
    #     self.robot.send_gripper_command(-1)
    #     # Perform joint reset if needed
    #     if joint_reset:
    #         print("JOINT RESET")
    #         JOINT_RESET = [0, 0, 1.570, 0, 1.57, 0]
    #         self.robot.send_joint_command(JOINT_RESET)
    #         time.sleep(1.5)
    #         return

    #     current_pos = self.robot.curr_end_pos.copy()
    #     safe_height_pose = current_pos.copy()
    #     alpha=0.5
    #     safe_height_pose[2] = (1-alpha)*self.resetpos[2]+alpha*current_pos[2]
    #     safe_height_pose[4] = self.resetpos[4]
    #     print(f"First lifting to safe height: {safe_height_pose[:3]}")
    #     self.interpolate_move(safe_height_pose, timeout=0.4)
    #     # Perform Carteasian reset
    #     if self.randomreset:  # randomize reset position in xy plane
    #         reset_pose = self.resetpos.copy()
    #         reset_pose[:2] += np.random.uniform(
    #             -self.random_xy_range, self.random_xy_range, (2,)
    #         )
    #         euler_random = self._RESET_POSE[3:].copy()
    #         euler_random[-1] += np.random.uniform(
    #             -self.random_rz_range, self.random_rz_range
    #         )
    #         reset_pose[3:] = euler_2_quat(euler_random)
    #         self.interpolate_move(reset_pose, timeout=1)
    #     else:
    #         reset_pose = self.resetpos.copy()
    #         self.interpolate_move(reset_pose, timeout=2)
    #     time.sleep(0.2)

        # Change to compliance mode
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

# 四元数姿态调整辅助函数
    def adjust_quaternion_pitch(self,quat, target_pitch,target_roll):
        """调整四元数到目标pitch角，保持roll和yaw不变"""
        r = R.from_quat(quat)
        __annotations__, _, yaw = r.as_euler('xyz')
        return R.from_euler('xyz', [target_roll, target_pitch, yaw]).as_quat()

    def interpolate_move(self, waypoints, timeout=2.0):
        """多关键点平滑插值运动，支持多个旋转关键点的球面插值"""
        waypoints = np.array(waypoints)
        if waypoints.shape[1] != 7:  # 确保是 [x,y,z,qx,qy,qz,qw] 格式
            raise ValueError("姿态必须是7维数组 [x,y,z,qx,qy,qz,qw]")

        steps = int(timeout * self.hz)
        t = np.linspace(0, 1, steps)
        
        # 位置插值
        pos_traj = interp1d(np.linspace(0, 1, len(waypoints)), waypoints[:, :3], kind='cubic', axis=0)(t)
        
        # 旋转插值 - 使用正确的scipy R方法
        Rs = R.from_quat(waypoints[:, 3:])
        
        for i in range(steps):
            pos = pos_traj[i]
            
            # 使用R类的Slerp方法
            if len(Rs) == 1:
                rot = Rs[0]
            else:
                # 创建Slerp对象进行插值
                key_times = np.linspace(0, 1, len(Rs))
                slerps = Slerp(key_times, Rs)
                rot = slerps(t[i])
            
            self.robot.send_pos_command(np.concatenate([pos, rot.as_quat()]))
            time.sleep(1/self.hz)
    def go_to_reset(self, joint_reset=False):
        """优先重置逻辑实现：先恢复pitch角，然后恢复z坐标，最后完整reset"""
        if joint_reset:
            # 关节空间重置
            self.robot.send_joint_command(self.RESET_JOINT_POSITIONS)
            time.sleep(2.0)
            return
        
        current_pose = self.robot.curr_end_pos.copy()
        print("current_pose",current_pose)
        reset_pose = self._RESET_POSE
        target_roll=self._RESET_POSE[3]
        target_pitch=self._RESET_POSE[4]
        reset_pose_quat = R.from_euler('xyz', reset_pose[3:]).as_quat()  # 7维目标姿态
        reset_pose=np.concatenate([reset_pose[:3],reset_pose_quat])
        
        
        # 定义关键路径点 - 体现优先级
        waypoints = [
             # 第一步：优先调整pitch角（保持x,y,z不变）
            np.concatenate([
                [current_pose[0], current_pose[1], current_pose[2]], 
                self.adjust_quaternion_pitch(current_pose[3:], target_pitch,target_roll)
            ]),
            np.concatenate([
                [current_pose[0], current_pose[1]+0.01, current_pose[2]],
                self.adjust_quaternion_pitch(current_pose[3:], target_pitch,target_roll)
            ]),
              
            # 第二步：调整z坐标（保持x,y和调整后的姿态不变）
            np.concatenate([
                [current_pose[0], current_pose[1], reset_pose[2]],
                self.adjust_quaternion_pitch(current_pose[3:], target_pitch,target_roll)
            ]),
            
            # 第三步：完整调整到reset姿态
            reset_pose
        ]
        
        self.interpolate_move(waypoints, timeout=2.0)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        # TODO: update param
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
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
        self.robot.reset_robot_force_config()
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

    def _get_obs(self) -> dict:
        obs = self.robot.get_obs()
        if self.display_image:
            display_key = ['wrist', 'wrist2','side']
            dis_images = {}
            for key in display_key:
                if key in obs['images'].keys():
                    dis_images[key] = obs['images'][key]
            if not self.img_queue.empty():
                try:
                    self.img_queue.get_nowait()
                except queue.Empty:
                    pass
            self.img_queue.put(dis_images)
        return obs

    def _init_new_camera_subscriber(self):
        """初始化新相机的订阅者"""
        try:
            self.new_camera_subscriber = rospy.Subscriber(
                "/wrist/color/image_raw",
                Image,
                self._new_camera_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            print("Successfully subscribed to /wrist/color/image_raw")
        except Exception as e:
            print(f"Failed to subscribe to new camera: {e}")

    def _new_camera_callback(self, msg):
        """新相机的回调函数"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.new_camera_image = cv_image
        except Exception as e:
            print(f"Error processing new camera image: {e}")

    def close(self):
        # 关闭新相机订阅
        if self.new_camera_subscriber is not None:
            self.new_camera_subscriber.unregister()

        if hasattr(self, "listener"):
            self.listener.stop()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
