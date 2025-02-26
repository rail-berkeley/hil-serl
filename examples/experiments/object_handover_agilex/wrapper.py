import copy
import threading
import time
from typing import OrderedDict

#from franka_env.utils.rotations import euler_2_quat
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from agilex_env.agilex_env import AgilexEnv
from agilex_env.agilex_env import DefaultEnvConfig

class LeftEnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "128422272758",
            "dim": (1280, 720),
            "exposure": 10000,
        },
        "side": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 10000,
        },
        "side_classifier": {}
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 250:],
        "side": lambda img: img[100:500, 150:1100],
        "side_classifier": lambda img: img[230:358, 800:1056],
    }
    RESET_POSE = np.array([0.6, -0.02, 0.42, np.pi, 0, 0])
    ABS_POSE_LIMIT_LOW = np.array([0.4, -0.03, 0.3, np.pi-0.01, -0.3, 0])
    ABS_POSE_LIMIT_HIGH = np.array([0.67, -0.01, 0.42, np.pi+0.01, 0.0, 0.01])
    RANDOM_RESET = False
    ACTION_SCALE = (0.04, 0.2, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 200
    GRIPPER_SLEEP = 1.0
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 120,
        "rotational_damping": 5,
        "translational_Ki": 0,
        "translational_clip_x": 0.0085,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.006,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 120,
        "rotational_damping": 5,
        "translational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.04,
        "rotational_clip_y": 0.04,
        "rotational_clip_z": 0.04,
        "rotational_clip_neg_x": 0.04,
        "rotational_clip_neg_y": 0.04,
        "rotational_clip_neg_z": 0.04,
        "rotational_Ki": 0,
    }


class RightEnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 10000,
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[::-1, ::-1][:, :900],
    }
    RESET_POSE = np.array([0.4, 0.0, 0.5, np.pi, 0, np.pi])
    ABS_POSE_LIMIT_LOW = np.array([0.32, -0.01, 0.36, np.pi-0.01, 0.0, np.pi-0.01])
    ABS_POSE_LIMIT_HIGH = np.array([0.67, 0.01, 0.5, np.pi+0.01, 0.3, np.pi+0.01])
    RANDOM_RESET = False
    ACTION_SCALE = (0.04, 0.2, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 200

    GRIPPER_SLEEP = 1.0
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 120,
        "rotational_damping": 5,
        "translational_Ki": 0,
        "translational_clip_x": 0.0085,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.007,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.006,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 120,
        "rotational_damping": 5,
        "translational_Ki": 0,
        "translational_clip_x": 0.02,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.02,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.04,
        "rotational_clip_y": 0.04,
        "rotational_clip_z": 0.04,
        "rotational_clip_neg_x": 0.04,
        "rotational_clip_neg_y": 0.04,
        "rotational_clip_neg_z": 0.04,
        "rotational_Ki": 0,
    }


class DualAgilexEnv(AgilexEnv):
    def __init__(
        self,
        fake_env = False,
        save_video = False
    ):

        self.env_left = AgilexEnv(is_left_arm = True, config=LeftEnvConfig(), fake_env=fake_env, save_video=save_video)
        self.env_right = AgilexEnv(is_left_arm = False, config=LeftEnvConfig(), fake_env = fake_env, save_video = save_video)

        # Action/Observation Space
        action_dim = len(self.env_left.action_space.low) + len(self.env_right.action_space.low)
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
        image_dict = ({f"left/{key}": self.env_left.observation_space["images"][key] for key in self.env_left.observation_space["images"].keys()} | 
                        {f"right/{key}": self.env_right.observation_space["images"][key] for key in self.env_right.observation_space["images"].keys()})

        state_dict = ({f"left/{key}": self.env_left.observation_space["state"][key] for key in self.env_left.observation_space["state"].keys()} |
                        {f"right/{key}": self.env_right.observation_space["state"][key] for key in self.env_right.observation_space["state"].keys()})
        
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(image_dict)
            }
        )
        self.display_images = False
        if self.display_images:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()

    def step(self, action: np.ndarray) -> tuple:
        action_left = action[:len(action)//2]
        action_right = action[len(action)//2:]
        def step_env_left():
            global ob_left, reward_left, done_left
            ob_left, reward_left, done_left, _, _ = self.env_left.step(action_left)

        def step_env_right():
            global ob_right, reward_right, done_right
            ob_right, reward_right, done_right, _, _ = self.env_right.step(action_right)

        # Create threads for each function
        thread_left = threading.Thread(target=step_env_left)
        thread_right = threading.Thread(target=step_env_right)

        # Start the threads
        thread_left.start()
        thread_right.start()

        # Wait for both threads to complete
        thread_left.join()
        thread_right.join()
        ob = self.combine_obs(ob_left, ob_right)
        if self.display_images:
            self.img_queue.put(ob['images'])
        return ob, int(reward_left and reward_right), done_left or done_right, False, {}
        

    def reset(self, **kwargs):
        def reset_env_left():
            global ob_left
            ob_left, _ = self.env_left.reset(**kwargs)

        def reset_env_right():
            global ob_right
            ob_right, _ = self.env_right.reset(**kwargs)

        thread_left = threading.Thread(target=reset_env_left)
        thread_right = threading.Thread(target=reset_env_right)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        ob = self.combine_obs(ob_left, ob_right)
        return ob, {}
    
    def combine_obs(self, ob_left, ob_right):
        left_images = {f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()}
        right_images = {f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()}
        left_state = {f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()}
        right_state = {f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()}
        ob = {
                "state": left_state | right_state,
                "images": left_images | right_images
            }
        return ob

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_left_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_left_gripper_pos = obs["state"][0, 0]
        self.last_right_gripper_pos = obs["state"][0, 19]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        info["grasp_penalty"] = 0.0

        if (action[6] < -0.5 and self.last_left_gripper_pos > 0.85) or (
            action[6] > 0.5 and self.last_left_gripper_pos < 0.85
        ):
            info["grasp_penalty"] += self.penalty
            print("left grasp penalty")
            
        if (action[13] < -0.5 and self.last_right_gripper_pos > 0.85) or (
            action[13] > 0.5 and self.last_right_gripper_pos < 0.85
        ):
            info["grasp_penalty"] += self.penalty
            print("right grasp penalty")

        self.last_left_gripper_pos = observation["state"][0, 0]
        self.last_right_gripper_pos = observation["state"][0, 13]
        
        return observation, reward, terminated, truncated, info
