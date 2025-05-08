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

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.tools import ImageDisplayer, q_image

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("method", "rlif", "valid values: rlif, cl, hil, soft_cl")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_string("save_to_txt", None, "Where to save the results to.")
flags.DEFINE_boolean("show_q_values", False, "")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_cyan(x):
    return print("\033[96m {}\033[00m".format(x))


def main(_):
    print_green(FLAGS)
    exp_name = FLAGS.exp_name
    enable_cl = FLAGS.method in ["cl", "soft_cl"]

    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=True,
        state_based=False,
    )
    rng = jax.random.PRNGKey(FLAGS.seed)

    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        cl_config = {
            "enabled": enable_cl,
            "soft": FLAGS.method == "soft_cl",
            "enable_margin_constraint": True,
            "enable_action_constraint": True,
            "constraint_coeff": 1.0,
            "constraint_eps": 0.0,
            "reward_coeff": 1.0,
        }
        print_green(f"Using CL Config: {cl_config}")

        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            enable_cl=enable_cl,
            cl_config=cl_config,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")
    
    success_counter = 0
    time_list = []

    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)
    print_green(f"Overriding agent with checkpoint at {FLAGS.eval_checkpoint_step}.")

    if FLAGS.show_q_values:
        q_queue = queue.Queue()
        q_display = ImageDisplayer(q_queue, "q_display")
        q_display.start()
    for episode in range(FLAGS.eval_n_trajs):
        print("reset start")
        ### receive signal from learner and then reset
        obs, _ = env.reset()
        print("reset end")
        done = False
        start_time = time.time()
        while not done:
            rng, key = jax.random.split(rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                argmax=True,
                seed=key
            )
            if FLAGS.show_q_values:
                q_value = np.asarray(jax.device_get(agent.forward_critic(jax.device_put(obs), actions[:-1], rng=key).min()))
                q_value_grasp = np.asarray(jax.device_get(agent.forward_grasp_critic(jax.device_put(obs), rng=key)))
                q_value_grasp_index = int(actions[-1] + 1)
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs

            if 'intervene_action' in info:
                print(info['intervene_action'][-1])

            if FLAGS.show_q_values:
                q_queue.put({'q_image': q_image(q_value, q_value_grasp, q_value_grasp_index, 'intervene_action' in info)})

            if done or truncated:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)

                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")

    if FLAGS.show_q_values:
        q_queue.put(None)
        cv2.destroyAllWindows()
        q_display.join()

    with open(FLAGS.save_to_txt, "a") as f:
        f.write(f"{FLAGS.eval_checkpoint_step} - {success_counter / FLAGS.eval_n_trajs} ({FLAGS.eval_n_trajs})\n")

if __name__ == "__main__":
    app.run(main)
