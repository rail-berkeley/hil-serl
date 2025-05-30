import os
import gymnasium as gym
import jax
import jax.numpy as jnp

import numpy as np
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from ros_robot_env.robot_env import DefaultEnvConfig
from ros_robot_env.robot_env import RobotRosEnv
from experiments.grab_earphone.wrapper import GripperPenaltyWrapper, ExpertIntervWrapper #TimePenaltyWrapper, ForcePenaltyWrapper, JitterPenaltyWrapper
from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    SpacemouseIntervention,

)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper

USE_FORCE_PENALTY = False
USE_TIME_PENALTY = False
USE_JITTER_PENALTY = False

class RightEnvConfig(DefaultEnvConfig):
    ARM = "dobot"
    CAMERAS = {
        "wrist": "/1/camera/color/image_raw", # "/camera_r/color/image_raw",
        "wrist2": "/2/camera/color/image_raw", # "/camera_l/color/image_raw",
        # "side": "/camera_s/color/image_raw",
    }
    DEPTH_CAMERAS = {
        "wrist_depth": "/1/camera/aligned_depth_to_color/image_raw", # "/camera_r/depth/image_rect_raw",
        "wrist2_depth": "/2/camera/aligned_depth_to_color/image_raw", # "/camera_l/depth/image_rect_raw",
        # "side_depth": "/camera_s/depth/image_rect_raw",
    }
    IMAGE_CROP = {
        # "side": lambda img: img[0:480, 120:600],
        #"wrist": lambda img: img[0:400, 130:530],
        "wrist": lambda img: img[0:450, 150:750],
        "wrist2": lambda img: img[0:450, 150:750],
        "wrist_depth": lambda img: img[0:480, 150:750],
        "wrist2_depth": lambda img: img[0:450, 150:750],
    }

    Z_OFFSET = -0.035
    RESET_POSE = np.array([0.452, 0.051, 0.100 + Z_OFFSET, 3.14, 0.0, 00])
    ABS_POSE_LIMIT_LOW = np.array([0.42, 0.01, 0.055 + Z_OFFSET, 3.12, -0.10, -1.57])
    ABS_POSE_LIMIT_HIGH = np.array([0.484, 0.086, 0.102 + Z_OFFSET, 3.15, 0.10, 1.57])
    gripper_open = 330
    gripper_close = 30

    RANDOM_RESET = False
    ACTION_SCALE = (0.015, 0.15, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100


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
    classifier_keys = None
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

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = RobotRosEnv(
            hz=5, config=RightEnvConfig(), fake_env=fake_env, save_video=save_video
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
            env = ExpertIntervWrapper(env)
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        env = RelativeFrame(env)

        env = Quat2EulerWrapper(env)

        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier and self.classifier_keys is not None:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                p = sigmoid(classifier(obs))
                #print(f"reward: {p}")
                # return int(p > 0.75 and obs['state'][0, 0] > 0.5)
                return 0# int(p[0] > 0.75)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        # env = GripperPenaltyWrapper(env, penalty=0)
        if USE_FORCE_PENALTY:
            env = ForcePenaltyWrapper(env)
        if USE_TIME_PENALTY:
            env = TimePenaltyWrapper(env)
        if USE_JITTER_PENALTY:
            env = JitterPenaltyWrapper(env)
        return env
