#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from pynput import keyboard
import requests
import cv2
import threading
import queue

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.tools import ImageDisplayer, q_image

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore, PreferenceBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("method", "rlif", "Valid values: rlif, cl, soft_cl, hgdagger")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("use_bc_loss", False, "Whether to add a crude bc max likelihood loss on the policy.")

flags.DEFINE_float("optimism", 0.0, "Whether or not to add a small amount of bonus to the post-state of interventions.")
flags.DEFINE_boolean("optimism_done_mask", False, "The done to be set for the optimism transition.")
flags.DEFINE_boolean("show_q_values", False, "Whether or not to open another window that shows the live Q values.")
flags.DEFINE_boolean("state_based", False, "Whether or not to use states instead of image + encoder.")
flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_cyan(x):
    return print("\033[96m {}\033[00m".format(x))

def print_cyan(x):
    return print("\033[96m {}\033[00m".format(x))


##############################################################################


failure_key = False
checkpoint_key = False
pause_key = False
def on_press(key):
    global failure_key
    global checkpoint_key
    global pause_key
    try:
        if str(key) == "'f'":
            failure_key = True
        elif str(key) == "'c'":
            checkpoint_key = True
        elif str(key) == "'r'":
            print_yellow("reset")
            requests.post("http://localhost:5000/reset_gripper")
        elif str(key) == "'t'":
            print_yellow("close gripper for reset")
            requests.post("http://localhost:5000/close_gripper")
        elif str(key) == "'p'":
            print_yellow("pause")
            pause_key = not pause_key
    except AttributeError:
        print("error")
        pass

def actor(agent, data_store, intvn_data_store, env, sampling_rng, pref_data_store = None, bc_data_store = None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global failure_key
    global checkpoint_key
    global pause_key

    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        print_green(f"Overriding agent with checkpoint at {FLAGS.eval_checkpoint_step}.")

        for episode in range(FLAGS.eval_n_trajs):
            print("reset start")
            ### receive signal from learner and then reset
            obs, _ = env.reset()
            time.sleep(7.0)
            obs, _ = env.reset()
            print("reset end")
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                if failure_key:
                    print("failure detected")
                    failure_key = False
                    done = True
                obs = next_obs

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
        return  # after done eval, return and exit

    buffers = (
        natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else []
    )
    start_step = 1 if len(buffers) == 0 else int(os.path.basename(buffers[-1])[12:-4]) + 1

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    if pref_data_store is not None:
        datastore_dict["actor_env_pref"] = pref_data_store
    if FLAGS.use_bc_loss:
        assert bc_data_store is not None
        datastore_dict["actor_env_bc"] = bc_data_store

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    learner_step = start_step
    def update_params(params):
        if isinstance(params, dict) and set(params.keys()) == set(["step"]):
            nonlocal learner_step
            learner_step = params['step']
        else:
            nonlocal agent
            agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    transitions_full_trajs = []
    demo_transitions = []
    demo_transitions_full_trajs = []
    interventions = []
    this_intervention = None
    preference_datas = []
    bc_transitions = []

    print("reset start")
    obs, _ = env.reset()
    time.sleep(5.0)
    obs, _ = env.reset()
    print("reset end")
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    total_interventions = 0

    pre_int_obs = None
    post_int_obs = None
    a_int_pi = None
    a_int_exp = None

    if FLAGS.show_q_values:
        q_queue = queue.Queue()
        q_display = ImageDisplayer(q_queue, "q_display")
        q_display.start()

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    print(config.buffer_period)
    from_time = time.time()
    cur_steps = 0
    for step in pbar:
        while pause_key:
            time.sleep(0.5)
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        if FLAGS.show_q_values:
            q_value = float(np.asarray(jax.device_get(agent.forward_critic(jax.device_put(obs), actions[:-1], rng=key).min())))
            q_value_grasp = float(np.asarray(jax.device_get(agent.forward_grasp_critic(jax.device_put(obs), rng=key)[int(actions[-1] + 1)])))
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            reward *= config.reward_scale
            cur_steps += 1
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")
            if failure_key:
                print("failure detected")
                failure_key = False
                truncated = True
            if FLAGS.show_q_values:
                info['q_value_grasp'] = q_value_grasp
                info['q_value'] = q_value
                q_queue.put({'q_image': q_image(q_value, q_value_grasp, 'intervene_action' in info)})

            # override the action with the intervention action
            if "intervene_action" in info:
                policy_actions = actions
                info["policy_action"] = actions
                actions = info["intervene_action"]
                info["policy_action"] = actions
                actions = info["intervene_action"]
                intervention_steps += 1

                post_int_obs = next_obs

                if not already_intervened:
                    print_cyan("Started intervention.")
                    intervention_count += 1

                    pre_int_obs = obs
                    a_int_exp = actions
                    a_int_pi = policy_actions

                    this_intervention = dict(
                        t0=cur_steps,
                        t1=cur_steps+1,
                        observations=[obs, next_obs],
                        actions=[actions],
                        policy_actions=[policy_actions],
                        dones=[done],
                        truncateds=[truncated],
                    )
                    if config.rlif_minus_one:
                        print(f"Detected intervention; RLIF replaced reward {reward} with {-1}.")
                        reward = -1
                else:
                    this_intervention['observations'].append(next_obs)
                    this_intervention['actions'].append(actions)
                    this_intervention['policy_actions'].append(policy_actions)
                    this_intervention['dones'].append(done)
                    this_intervention['truncateds'].append(truncated)
                    this_intervention['t1'] = cur_steps+1
                already_intervened = True
            else:
                if already_intervened:
                    print_cyan(f"Ended intervention of {this_intervention['t1'] - this_intervention['t0']} steps.")
                    if this_intervention is None:
                        print("Error: Should not be None")
                    interventions.append(this_intervention)
                    # add to preference buffer
                    if FLAGS.method in ["cl", "soft_cl"]:
                        pref_datapoint = dict(
                            pre_obs=pre_int_obs,
                            post_obs=post_int_obs,
                            a_pi=a_int_pi,
                            a_exp=a_int_exp,
                            t=np.array([this_intervention['t1'] - this_intervention['t0']]),
                        )
                        pref_data_store.insert(pref_datapoint)
                        preference_datas.append(pref_datapoint)
                    this_intervention = None

                    if abs(FLAGS.optimism) > 1e-9:
                        print_cyan(f"Adding optimism transition with reward={FLAGS.optimism} and done={FLAGS.optimism_done_mask}.")
                        transition = dict(
                            observations=obs,
                            actions=actions,
                            next_observations=next_obs,
                            rewards=FLAGS.optimism,
                            masks=1.0 - FLAGS.optimism_done_mask, # Used in training, denoting whether or not we're at the end of a trajectory.
                            dones=FLAGS.optimism_done_mask, # Not actually used in training.
                        )
                        if 'grasp_penalty' in info:
                            transition['grasp_penalty'] = 0
                        data_store.insert(transition)
                        transitions.append(copy.deepcopy(transition) | {'info': info | {'optimism': True}})
                already_intervened = False

            if (done or truncated) and this_intervention is not None:
                print_cyan(f"Ended intervention of {this_intervention['t1'] - this_intervention['t0']} steps.")
                interventions.append(this_intervention)
                # add to preference buffer
                if FLAGS.method in ["cl", "soft_cl"]:
                    pref_datapoint = dict(
                        pre_obs=pre_int_obs,
                        post_obs=post_int_obs,
                        a_pi=a_int_pi,
                        a_exp=a_int_exp,
                        t=np.array([this_intervention['t1'] - this_intervention['t0']]),
                    )
                    pref_data_store.insert(pref_datapoint)
                    preference_datas.append(pref_datapoint)
                this_intervention = None

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # if checkpoint_key:
            #     breakpoint()
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition) | {'info': info})
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition) | {'info': info})
            if FLAGS.use_bc_loss and "intervene_action" in info:
                bc_data_store.insert(transition)
                bc_transitions.append(transition)

            obs = next_obs
            if done or truncated:
                failure_key = False
                if "episode" not in info:
                    info["episode"] = {}
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps

                total_interventions += intervention_count

                info["episode"]["total_interventions"] = total_interventions
                info["episode"]["intervention_rate"] = total_interventions / (step + 1)
                info["episode"]["current_intervention_rate"] = intervention_steps / cur_steps
                info["episode"]["episode_duration"] = time.time() - from_time
                info["episode"]["success_rate"] = running_return
                info["episode"]["episode_steps"] = cur_steps
                info["episode"]["environment_step"] = step
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                cur_steps = 0

                already_intervened = False
                client.update()
                print("reset start")
                obs, _ = env.reset()
                transitions_full_trajs = transitions
                demo_transitions_full_trajs = demo_transitions
                # input("Waiting for input to proceed...")
                time.sleep(7.0)
                obs, _ = env.reset()
                # For synchronizing learner and actor...
                if step > learner_step * 1 + 200:
                    print("Stopped actor")
                    while step > learner_step * 1 + 200:
                        time.sleep(0.5)
                    print("Released actor")
                print("reset end")
                from_time = time.time()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            interventions_buffer_path = os.path.join(FLAGS.checkpoint_path, "interventions")
            preference_buffer_path = os.path.join(FLAGS.checkpoint_path, "preference_buffer")
            bc_buffer_path = os.path.join(FLAGS.checkpoint_path, "bc_buffer")

            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            if not os.path.exists(interventions_buffer_path):
                os.makedirs(interventions_buffer_path)
            if not os.path.exists(preference_buffer_path):
                os.makedirs(preference_buffer_path)
            if not os.path.exists(bc_buffer_path):
                os.makedirs(bc_buffer_path)
            
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(transitions_full_trajs)} transitions out of {len(transitions)} to {fp} !!!")
                pkl.dump(transitions_full_trajs, f)
                transitions = transitions[len(transitions_full_trajs):]
                transitions_full_trajs = []
            with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(demo_buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(demo_transitions_full_trajs)} expert transitions out of {len(demo_transitions)} to {fp} !!!")
                pkl.dump(demo_transitions_full_trajs, f)
                demo_transitions = demo_transitions[len(demo_transitions_full_trajs):]
                demo_transitions_full_trajs = []
            with open(os.path.join(interventions_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(interventions_buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(interventions)} interventions to {fp}")
                pkl.dump(interventions, f)
                interventions = []
            with open(os.path.join(bc_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(bc_buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(bc_transitions)} interventions to {fp}")
                pkl.dump(bc_transitions, f)
                bc_transitions = []
        
        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)
    
    if FLAGS.show_q_values:
        q_queue.put(None)
        cv2.destroyAllWindows()
        q_display.join()


##############################################################################


def learner(rng, agent: SACAgentHybridSingleArm, replay_buffer, demo_buffer, preference_buffer = None, bc_buffer = None, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    if FLAGS.method == "cl" and FLAGS.method != "soft_cl":
        assert "modules_log_alpha_state" in agent.state.params
        assert "modules_log_alpha_gripper_state" in agent.state.params
        train_critic_networks_to_update = frozenset(train_critic_networks_to_update | {"log_alpha_state", "log_alpha_grasp_state"})
        train_networks_to_update = frozenset(train_networks_to_update | {"log_alpha_state", "log_alpha_grasp_state"})


    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step) # + config.pretraining_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    if preference_buffer is not None:
        server.register_data_store("actor_env_pref", preference_buffer)
    server.start(threaded=True)

    if FLAGS.use_bc_loss and config.pretraining_steps > 0:
        pbar = tqdm.tqdm(range(config.pretraining_steps))
        for step in pbar:
            bc_batch = bc_buffer.sample(config.batch_size)
            agent, update_info = agent.update_bc(bc_batch)
            pbar.set_description(f"bc_loss = {round(update_info['actor']['bc_loss'], 3)}")
            wandb_logger.log(update_info, step=step)
        step = config.pretraining_steps
        agent = jax.block_until_ready(agent)
        server.publish_network(agent.state.params)
        checkpoints.save_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path), agent.state, step=1, keep=100
        )

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    if FLAGS.method in ["cl", "soft_cl"]:
        assert preference_buffer is not None
        preference_iterator = preference_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size,
            },
            device=sharding.replicate(),
        )
    if FLAGS.use_bc_loss:
        assert bc_buffer is not None
        bc_iterator = bc_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size,
            },
            device=sharding.replicate(),
        )

    # wait till the replay buffer is filled with enough data
    timer = Timer()


    '''
    update_steps = 0
    current_envs_steps = 0
    while update_steps < max_steps:
        new_steps = getfromactor - current_env_steps
        num_updates = new_steps * utd
        for _ in range(num)
            update###
        send signal to actor to get next trajectory
        
    
    
    '''
    for step in tqdm.tqdm(
        range(start_step + config.pretraining_steps, config.max_steps + config.pretraining_steps), dynamic_ncols=True, desc="learner"
    ):
        if step - config.pretraining_steps > len(replay_buffer) * 1 + 1 + 300:
            while step - config.pretraining_steps > len(replay_buffer) * 1 + 1 + 300:
                time.sleep(0.5)
            print(f"Training for another {(len(replay_buffer) * 1 + 1 + 300) - (step - config.pretraining_steps) + 1} steps...")
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        if FLAGS.method != "hgdagger":
            for critic_step in range(config.cta_ratio - 1):
                with timer.context("sample_replay_buffer"):
                    batch = next(replay_iterator)
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)
                    pref_batch = next(preference_iterator) if preference_buffer is not None else None
                    bc_batch = next(bc_iterator) if bc_buffer is not None else None

                with timer.context("train_critics"):
                    agent, critics_info = agent.update(
                        batch,
                        networks_to_update=train_critic_networks_to_update,
                        pref_batch=pref_batch,
                        bc_batch=bc_batch,
                    )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            pref_batch = next(preference_iterator) if preference_buffer is not None else None
            bc_batch = next(bc_iterator) if bc_buffer is not None else None

            if FLAGS.method == "hgdagger":
                agent, update_info = agent.update_bc(bc_batch)
            else:
                agent, update_info = agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
                    pref_batch=pref_batch,
                    bc_batch=bc_batch,
                )

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
            server.publish_network({'step': step - config.pretraining_steps})

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step) # + config.pretraining_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step) # + config.pretraining_steps)


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    enable_cl = FLAGS.method in ["cl", "soft_cl"]
    enable_cl = FLAGS.method in ["cl", "soft_cl"]

    if config.rlif_minus_one:
        print_green("Using RLIF.")
    if enable_cl:
        print_green("Using CL.")
    if FLAGS.method == "hgdagger":
        print_green("Using HG-DAgger.")
        assert FLAGS.use_bc_loss

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

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
            has_image=not FLAGS.state_based,
            use_bc_loss=FLAGS.use_bc_loss
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

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl-cubereach2",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    def create_demo_buffer():
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        return demo_buffer

    def create_preference_buffer():
        if FLAGS.method in ["rlif", "hgdagger", "hil"]:
            return None
        preference_buffer = PreferenceBufferDataStore(
            env.observation_space,
            env.observation_space,
            env.action_space,
            env.action_space,
            config.replay_buffer_capacity,
        )
        return preference_buffer

    def create_bc_buffer():
        if not FLAGS.use_bc_loss:
            return None
        bc_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        return bc_buffer

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = create_demo_buffer()
        preference_buffer = create_preference_buffer()
        bc_buffer = create_bc_buffer()

        prev_checkpoint_exist: bool = FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path)

        if prev_checkpoint_exist:
            assert os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if prev_checkpoint_exist and preference_buffer is not None:
            assert os.path.exists(os.path.join(FLAGS.checkpoint_path, "preference_buffer"))
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "preference_buffer/*.pkl")):
                with open(file, "rb") as f:
                    preferences = pkl.load(f)
                    for preference in preferences:
                        preference_buffer.insert(preference)
            print_green(
                f"Loaded previous preference buffer data. Preference buffer size: {len(preference_buffer)}"
            )

        if prev_checkpoint_exist:
            assert os.path.exists(os.path.join(FLAGS.checkpoint_path, "demo_buffer"))
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )
        
        if prev_checkpoint_exist and bc_buffer is not None:
            assert os.path.exists(os.path.join(FLAGS.checkpoint_path, "bc_buffer"))
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "bc_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        bc_buffer.insert(transition)
            print_green(
                f"Loaded previous bc buffer data. Bc buffer size: {len(bc_buffer)}"
            )

        assert FLAGS.demo_path is not None
        if len(demo_buffer) == 0:
            num_demos = 0
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        # Transforming the data
                        for k in set(transition['observations'].keys()) - set(config.image_keys + ['state']):
                            del transition['observations'][k]
                            del transition['next_observations'][k]
                        for k in config.image_keys:
                            img = transition['observations'][k]
                            if img.ndim == 4 and img.shape[0] == 1:
                                img = img[0]
                            transition['observations'][k] = cv2.resize(img, (128, 128))
                            img = transition['next_observations'][k]
                            if img.ndim == 4 and img.shape[0] == 1:
                                img = img[0]
                            transition['next_observations'][k] = cv2.resize(img, (128, 128))
                        if transition['actions'].shape == (4,):
                            transition['actions'] = np.concatenate([transition['actions'][:3], np.zeros((3,)), transition['actions'][3:]], axis=0)
                        transition['grasp_penalty'] = 0
                        assert transition['rewards'] < 1 + 1e-6 and transition['rewards'] > -1e-6, f"{transition['rewards']}"

                        num_demos += transition['rewards']
                        transition['rewards'] *= config.reward_scale
                        demo_buffer.insert(transition)
                        if bc_buffer is not None:
                            bc_buffer.insert(transition)
            
            print_green(f"demo buffer size: {len(demo_buffer)}")
            print_green(f"demo count: {num_demos}")
            if bc_buffer is not None:
                print_green(f"bc buffer size: {len(bc_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            preference_buffer=preference_buffer,
            bc_buffer=bc_buffer,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(10000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(10000)
        pref_data_store = QueuedDataStore(10000) if FLAGS.method in ["cl", "soft_cl"] else None
        bc_data_store = QueuedDataStore(10000) if FLAGS.use_bc_loss else None

        if FLAGS.method in ["cl", "soft_cl"]:
            pref_data_store = QueuedDataStore(10000)
        else:
            pref_data_store = None
        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            pref_data_store=pref_data_store,
            bc_data_store=bc_data_store
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
