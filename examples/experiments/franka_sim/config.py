import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    SimSpacemouseIntervention,
    Quat2EulerWrapper,
    Useless4To7Wrapper,
    BruhImageResizingWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, AWrapperThatFlattensState

from experiments.config import DefaultTrainingConfig
from experiments.franka_sim.wrapper import FrankaSimEnv
from serl_launcher.wrappers.chunking import ChunkingWrapper
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper
from franka_sim.mujoco_gym_env import GymRenderingSpec


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 10 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = False
    checkpoint_period = 1000
    cta_ratio = 5
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env: bool, save_video: bool, classifier: bool, state_based: bool = False):
        if state_based:
            self.image_keys = [] # hacky but works
        from franka_sim.envs import PandaPickCubeGymEnv
        env = PandaPickCubeGymEnv(
            action_scale=(0.1, 1),
            render_mode="rgb_array" if fake_env else "human",
            image_obs=not state_based,
            control_dt=0.1,
            time_limit=30.0,
            show_viewer=not fake_env,
            render_spec=GymRenderingSpec(height=512, width=512),
            fixed_block_position=False,
        )
        env = FrankaSimEnv(env, action_scale=[0.1, 0.1, 0.1], show_viewer=False, fake_env=fake_env)
        if not state_based:
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = SimSpacemouseIntervention(env, fake_env=fake_env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = Useless4To7Wrapper(env)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        env = BruhImageResizingWrapper(env, (128, 128))
        if state_based:
            env = AWrapperThatFlattensState(env)
        
        return env
