import os
import gymnasium as gym
import jax
import jax.numpy as jnp

import numpy as np
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper ,RemoveImageKeyWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from ros_robot_env.robot_env import DefaultEnvConfig
from ros_robot_env.robot_env import RobotRosEnv
from experiments.plugusb.wrapper import GripperPenaltyWrapper, ExpertIntervWrapper, TimePenaltyWrapper, ForcePenaltyWrapper, JitterPenaltyWrapper
from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    SpacemouseIntervention,

)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper

USE_FORCE_PENALTY =False
USE_TIME_PENALTY = True
USE_JITTER_PENALTY = True

class RightEnvConfig(DefaultEnvConfig):
    ARM = "dobot"
    CAMERAS = {
        "wrist": "/1/camera/color/image_raw", # "/camera_r/color/image_raw",
        "wrist2": "/2/camera/color/image_raw", # "/camera_l/color/image_raw",
        "side": "/wrist/color/image_raw",
    }
    DEPTH_CAMERAS = {
        "wrist_depth": "/1/camera/aligned_depth_to_color/image_raw", # "/camera_r/depth/image_rect_raw",
        "wrist2_depth": "/2/camera/aligned_depth_to_color/image_raw", # "/camera_l/depth/image_rect_raw",
        # "side_depth": "/camera_s/depth/image_rect_raw",
    }
    IMAGE_CROP = {
        # "side": lambda img: img[0:480, 120:600],
        #"wrist": lambda img: img[0:400, 130:530],
        "wrist": lambda img: img[100:350, 350:750],
        "wrist2": lambda img: img[100:350, 350:750],
        "side": lambda img: img[565:725,276:426],
        "wrist_depth": lambda img: img[100:350, 350:750],
        "wrist2_depth": lambda img: img[100:350, 350:750],
    }
    # cls_keys = ["wrist2"]
    # CLS_CAMERAS = {
    #     "wrist2cls": "/2/camera/color/image_raw", 
    # }
    RESET_POSE = np.array([0.670, 0.222, 0.258, -np.pi, 0.0, 0.5*np.pi])
    ABS_POSE_LIMIT_LOW = np.array([0.640, 0.202, 0.200, np.pi-0.1, -0.05, 0.5*np.pi-0.1])
    ABS_POSE_LIMIT_HIGH = np.array([0.700, 0.242, 0.288, np.pi, 0.05, 0.5*np.pi+0.1])
    gripper_open = 600
    gripper_close = 120

    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0
    ACTION_SCALE = (0.01, 0.05, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 150


    FORCE_PARAM = {
      'fc_mode': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      'fc_mass' : [20, 20, 20, 10, 20, 20],
      'fc_damping' : [600, 600, 600, 100, 100, 100],
      'fc_stiffness' : [400, 400, 400, 1, 1, 1],
      'fc_speed' : [200, 200, 200, 3, 3, 3],
      'gripper_force': 60
    }
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


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist2","wrist2_depth", "wrist_depth",  "wrist"]
    classifier_keys = ["side","wrist2","wrist2_depth", "wrist_depth",  "wrist"]
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "gripper_pose",
        "tcp_force",
        "tcp_torque",
        "grip_status"
    ]
    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=True,sample=False):
        env = RobotRosEnv(
            hz=5, config=RightEnvConfig(), fake_env=fake_env, save_video=save_video
        )
        if USE_JITTER_PENALTY:
            env = JitterPenaltyWrapper(env)
        if not fake_env:
            env = SpacemouseIntervention(env)

        
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier and self.classifier_keys is not None:
            env = MultiCameraBinaryRewardClassifierWrapper(
                env=env,
                model_path="/home/amigo/jiaqi/yolov/runs/classify/train2/weights/best.pt",
                target_hz=5
            )
        if not fake_env:
            env = ExpertIntervWrapper(env)
        if not sample:
           env = RemoveImageKeyWrapper(env, keys_to_remove=self.classifier_keys[0])
        # env = GripperPenaltyWrapper(env, penalty=-0.02)
        env = GripperPenaltyWrapper(env, penalty=0)
        if USE_FORCE_PENALTY:
            env = ForcePenaltyWrapper(env)
        if USE_TIME_PENALTY:
            env = TimePenaltyWrapper(env)
        return env
