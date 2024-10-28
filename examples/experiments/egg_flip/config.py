import gymnasium as gym
import jax 

from franka_env.envs.wrappers import Quat2EulerWrapper
from franka_env.envs.franka_wrench_env import FrankaWrenchEnv, DefaultWrenchEnvConfig
from experiments.egg_flip.wrapper import (
    EggFlipActionWrapper,
    EggFlipSpacemouseIntervention,
    EggClassifierWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig

class EggFlipEnvConfig(DefaultWrenchEnvConfig):
    IMAGE_CROP = {
        "wrist_1": lambda image: image[200:, :, :],
        "side": lambda image: image[100:650, 550:900, :],
    }

class TrainConfig(DefaultTrainingConfig):
    image_keys = ['wrist_1', "side"]
    proprio_keys = ['tcp_pose', 'tcp_vel', 'q', 'dq']
    classifier_keys = ['wrist_1']
    discount = 0.985
    buffer_period = 1000
    checkpoint_period = 5000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"
    
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = FrankaWrenchEnv(
                            fake_env=fake_env,
                            save_video=save_video,
                            config=EggFlipEnvConfig(),
                    )
        
        env = EggFlipActionWrapper(env)
        env = EggFlipSpacemouseIntervention(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if classifier and self.classifier_keys is not None:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path="classifier_ckpt/",
                n_way=3,
            )
            env = EggClassifierWrapper(env, classifier_func)

        return env