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
from experiments.object_handover_agilex.wrapper import (
    DualAgilexEnv,
    GripperPenaltyWrapper,
    DualRelativeFrame,
)


class LeftEnvConfig(DefaultEnvConfig):
    ARM = "left"
    SERVER_URL = "http://127.0.0.1:5000/"
    CAMERAS = {
        "wrist": "/camera_l/color/image_raw",
        "front": "/camera_f/color/image_raw",
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 250:],
        "front": lambda img: img[100:500, 150:1100],
    }
    RESET_POSE = np.array([0.10, 0.00, 0.206, 0, 1.483, 0])
    ABS_POSE_LIMIT_LOW = np.array([0.08, -0.10, 0.20, np.pi - 0.01, -0.3, 0])
    ABS_POSE_LIMIT_HIGH = np.array([0.4, 0.28, 0.40, np.pi + 0.01, 0.0, 0.01])
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
    ARM = "right"
    CAMERAS = {
        "wrist": "/camera_r/color/image_raw",
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[::-1, ::-1][:, :900],
    }
    RESET_POSE = np.array([0.10, 0.00, 0.206, 0, 1.483, 0])
    ABS_POSE_LIMIT_LOW = np.array([0.08, -0.31, 0.20, np.pi - 0.01, 0.0, np.pi - 0.01])
    ABS_POSE_LIMIT_HIGH = np.array([0.4, 0.11, 0.4, np.pi + 0.01, 0.3, np.pi + 0.01])
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


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left/wrist", "right/wrist", "left/front"]
    classifier_keys = ["left/side_classifier"]
    proprio_keys = [
        "left/tcp_pose",
        "left/tcp_vel",
        "left/gripper_pose",
        "right/tcp_pose",
        "right/tcp_vel",
        "right/gripper_pose",
    ]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "dual-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = DualAgilexEnv(
            left_env_config=LeftEnvConfig(),
            right_env_config=RightEnvConfig(),
            fake_env=fake_env,
            save_video=save_video,
        )
        # if not fake_env:
        #    env = DualSpacemouseIntervention(env, gripper_enabled=True)

        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        print("create relative env")
        env = DualRelativeFrame(env)

        # env = DualQuat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # if classifier and self.classifier_keys is not None:
        #    classifier = load_classifier_func(
        #        key=jax.random.PRNGKey(0),
        #        sample=env.observation_space.sample(),
        #        image_keys=self.classifier_keys,
        #        checkpoint_path=os.path.abspath("classifier_ckpt/"),
        #    )

        #    def reward_func(obs):
        #        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        #        p = sigmoid(classifier(obs))
        #        return int(p > 0.75 and obs['state'][0, 0] > 0.5)

        #    env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        # env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
