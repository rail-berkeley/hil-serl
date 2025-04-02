import os
import gymnasium as gym
import jax
import jax.numpy as jnp

import numpy as np
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from agilex_env.agilex_env import DefaultEnvConfig
from agilex_env.agilex_env import AgilexEnv
from experiments.grab_bottole.wrapper import GripperPenaltyWrapper, ExpertIntervWrapper
from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    SpacemouseIntervention,

)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import Quat2EulerWrapper


class RightEnvConfig(DefaultEnvConfig):
    ARM = "right"
    CAMERAS = {
        "wrist": "/camera_r/color/image_raw",
        #"front": "/camera_f/color/image_raw",
        "side": "/camera_s/color/image_raw",
    }
    IMAGE_CROP = {
        "side": lambda img: img[20:430, 30:600],
        "wrist": lambda img: img[150:480, 0:600],
        "front": lambda img: img[50:400, 50:600],
    }
    RESET_POSE = np.array([0.10, 0.00, 0.206, 0, 1.483, 0])
    ABS_POSE_LIMIT_LOW = np.array([0.08, -0.20, 0.12, -1.04, 1.43, -0.24])
    ABS_POSE_LIMIT_HIGH = np.array([0.5, 0.20, 0.3, 0.04, 1.74, 0.24])

    RANDOM_RESET = False
    ACTION_SCALE = (0.04, 0.2, 1)
    DISPLAY_IMAGE = True
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


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "side"]
    classifier_keys = ["side"]
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "gripper_pose",
    ]
    buffer_period = 1000
    checkpoint_period = 2000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = AgilexEnv(
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
                print(f"reward: {p}")
                # return int(p > 0.75 and obs['state'][0, 0] > 0.5)
                return 0# int(p[0] > 0.75)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
