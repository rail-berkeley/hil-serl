import os
import gymnasium as gym
import jax
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    DualSpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import DualRelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from franka_env.envs.dual_franka_env import DualFrankaEnv
import numpy as np
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.object_handover.wrapper import HandOffEnv, GripperPenaltyWrapper

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


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left/wrist", "right/wrist", "left/side"]
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
        left_env = HandOffEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=LeftEnvConfig,
        )

        right_env = HandOffEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=RightEnvConfig,
        )

        env = DualFrankaEnv(left_env, right_env)
        if not fake_env:
            env = DualSpacemouseIntervention(env, gripper_enabled=True)
        env = DualRelativeFrame(env)

        env = DualQuat2EulerWrapper(env)
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
                return int(p > 0.75 and obs['state'][0, 0] > 0.5)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
