import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.cube_reach2.wrapper import CubeReach2Env
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://localhost:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "123622270810",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_1": {
            "serial_number": "032522250211",
            "dim": (1280, 720),
            "exposure": 13000,
        }
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[50:550, 150:1200],
        # "wrist_2": lambda img: img[100:500, 400:900],
        # "side_1": lambda img: img[440:700, 440:900],
        "side_1": lambda img: img[320:700, 600:950],
    }
    # (440, 440) -> (900, 700)
    # TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    # GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])

    # Step 1: add reset pose
    # RESET_POSE = np.array([0.7, -0.05, 0.225,np.pi, 0, -np.pi/2])
    RESET_POSE = np.array([0.62, 0.099, 0.15, np.pi, 0, np.pi / 2])

    # Step 2: add bounding boxes
    ABS_POSE_LIMIT_LOW  = np.array([0.58, -0.056, 0.09, np.pi - 0.1, -0.1, np.pi / 2 - 0.1])
    ABS_POSE_LIMIT_HIGH = np.array([0.66,  0.267, 0.18, np.pi + 0.1,  0.1, np.pi / 2 + 0.1])

    # Step 3: set reset randomization
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 6
    ACTION_SCALE = np.array([0.1, 0.3, 1]) # Testing x
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 300


    ## Step 4: Copy the same control parameters -- keep it the same as below : TODO: figure out how to tune these+
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "side_1"]
    classifier_keys = ["wrist_1", "side_1"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    checkpoint_period = 4000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = CubeReach2Env(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                prob = sigmoid(classifier(obs))[0]
                print(prob)
                return int(prob > 0.85) # and obs['state'][0, 3] < 0.12
                # return 0.0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env