
from absl import app, flags
import time
import numpy as np
import os
import pickle
import imageio
import cv2
import queue
from pynput import keyboard
import threading
from flax.training import checkpoints
import jax
import jax.numpy as jnp

# from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)

checkpoint_path = "/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/franka_sim/debug_rlif_2"

from experiments.config import DefaultTrainingConfig
class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 1 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = False
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

exp_name = "franka_sim"
config = TrainConfig()
# env = config.get_environment(fake_env=True,save_video=False,classifier=True)

intervene_steps = 0  # Default number of steps between pre and post intervention states
constraint_eps = 0.1  # Default constraint epsilon

# obs_key_shapes = [('front', (1, 128, 128, 3)), ('state', (1, 7)), ('wrist', (1, 128, 128, 3))]
obs_sample = {
    'front': np.zeros((1, 128, 128, 3), dtype=np.uint8),
    'state': np.zeros((1, 7), dtype=np.float32),
    'wrist': np.zeros((1, 128, 128, 3), dtype=np.uint8),
}
action_sample = np.zeros(7, dtype=np.float32)

agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
    seed=0,
    sample_obs=obs_sample,
    sample_action=action_sample,
    image_keys=config.image_keys,
    encoder_type=config.encoder_type,
    discount=config.discount,
    enable_cl=False,
    soft_cl = False,
    intervene_steps=intervene_steps,
    constraint_eps=constraint_eps,
)

ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(checkpoint_path),
        agent.state,
        step='40000',
    )
agent = agent.replace(state=ckpt)


preference_buffer_base_path="experiments/franka_sim/debug_rlif_2/interventions/transitions"
preference_buffer_paths = [f"{preference_buffer_base_path}_{i}.pkl" for i in range(1000, 14000, 1000)]

preference_buffer = []

for preference_buffer_path in preference_buffer_paths:
    if not os.path.exists(preference_buffer_path):
        print(f"Preference buffer path {preference_buffer_path} does not exist.")
        continue

    # Load the preference buffer
    with open(preference_buffer_path, 'rb') as f:
        preference_buffer_part = pickle.load(f)
        preference_buffer.extend(preference_buffer_part)

rng = jax.random.PRNGKey(0)

def get_action(obs, rng):
    rng, key = jax.random.split(rng)
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=True,
        seed=key
    )
    return actions, rng

pre_intervention_obs = [p['observations'][0] for p in preference_buffer]
intervene_actions = [p['actions'][0] for p in preference_buffer]
policy_actions = [p['policy_actions'][0] for p in preference_buffer]
post_intervention_obs = [p['observations'][-1] for p in preference_buffer]

pre_intervention_obs = {
    'front': np.array([obs['front'] for obs in pre_intervention_obs]),
    'state': np.array([obs['state'] for obs in pre_intervention_obs]),
    'wrist': np.array([obs['wrist'] for obs in pre_intervention_obs]),
}
pre_intervention_expert_action, rng = get_action(pre_intervention_obs, rng)


post_intervention_obs = {
    'front': np.array([obs['front'] for obs in post_intervention_obs]),
    'state': np.array([obs['state'] for obs in post_intervention_obs]),
    'wrist': np.array([obs['wrist'] for obs in post_intervention_obs]),
}
post_intervention_expert_action, rng = get_action(post_intervention_obs, rng)

policy_actions = np.array(policy_actions)
intervene_actions = np.array(intervene_actions)

key, rng =  jax.random.split(rng)
q_pre_expert = agent.forward_critic(pre_intervention_obs, pre_intervention_expert_action[:, :6], key)
key, rng =  jax.random.split(rng)
q_post_expert = agent.forward_critic(post_intervention_obs, post_intervention_expert_action[:, :6], key)
key, rng =  jax.random.split(rng)
q_pre_policy = agent.forward_critic(pre_intervention_obs, policy_actions[:, :6], key)
key, rng =  jax.random.split(rng)
q_pre_intervene = agent.forward_critic(pre_intervention_obs, intervene_actions[:, :6], key)
# q_post_expert = agent.q_network.apply(agent.state.params, post_intervention_obs, post_intervention_expert_action)

# q_pre_policy = agent.q_network.apply(agent.state.params, pre_intervention_obs, policy_actions)
# q_post_policy = agent.q_network.apply(agent.state.params, pre_intervention_obs, policy_actions)
constraint1_acc = ((q_pre_expert - q_post_expert) < 0).mean()
qvalue_based_learning_intervene = ((q_pre_policy - q_pre_intervene) < 0).mean()
qvalue_based_learning_expert = ((q_pre_policy - q_pre_expert) < 0).mean()
constraint2_acc = ((q_pre_intervene - q_post_expert) < 0).mean()

print(f"Constraint 1 accuracy: {constraint1_acc}")
print(f"Q-value based learning intervene accuracy: {qvalue_based_learning_intervene}")
print(f"Q-value based learning expert accuracy: {qvalue_based_learning_expert}")
print(f"Constraint 2 accuracy: {constraint2_acc}")
breakpoint()