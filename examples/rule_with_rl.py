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
import sys
import signal
import copy
import pickle as pkl
import threading
from pynput import keyboard
from colorama import Fore, Style

from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.utils.timer_utils import Timer
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent_hybrid_single_arm,
)

from experiments.mappings import CONFIG_MAPPING
from ros_robot_env.ros_robot_interface import RobotRosInterface

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def has_checkpoint_folder(directory):
    if not os.path.exists(directory):
        return False
    # 获取目录下的所有文件和文件夹
    items = os.listdir(directory)
    
    # 遍历所有项，检查是否存在以 "checkpoint_" 开头的文件夹
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint_"):
            return True
    return False


##############################################################################

import cv2
import numpy as np

def detect_white_headphone(image, center_tolerance_ratio=0.2, min_contour_area=1000):
    """
    检测图像中是否存在白色耳机，且是否位于图像中心附近。
    """
    # 转为 HSV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 白色的 HSV 范围（需视照明微调）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    center_tolerance = center_tolerance_ratio * np.array([width, height])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue

        # 得到轮廓中心
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = np.array([cx, cy])

        # 判断是否在中心容忍范围内
        if np.all(np.abs(center - image_center) <= center_tolerance):
            return True  # 有一个居中白色大物体 -> 认为是耳机

    return False


def is_white_headphone_centered(image1, image2):
    """
    输入两个图像，检测是否任一图像中存在居中的白色耳机。
    """
    return detect_white_headphone(image1) or detect_white_headphone(image2)

def close_to_target(robot_interface):
    if robot_interface.curr_end_pos[2] > 0.1: #rl reset pose
        return False
    obs_imgs = robot_interface.wrist_img
    return is_white_headphone_centered(obs_imgs['wrist_l'], obs_imgs['wrist_r'])

def rule_based_policy(robot_interface, mouse_expert):
    action, buttons = self.expert.get_action()
    gripper_action = 0
    if buttons[0]:
        gripper_action = -1
    elif buttons[-1]:
        gripper_action = 1
    # print(f"pos before clip: {curr_end_pos}")
    if np.linalg.norm(action) > 0.01:
        try:
            current_end_pos = robot_interface.curr_end_pos
            xyz_delta = action[:3]
            target_end_pos[:3] = curr_end_pos[:3] + xyz_delta * self.action_scale[0]
            ## GET ORIENTATION FROM ACTION
            #euler_zxy = np.array([action[5], action[3], action[4]])
            curr_euler = Rotation.from_quat(self.curr_end_pos[3:]).as_euler("xyz")
            print(f"curr_euler: {curr_euler}")
            target_end_pos[3:] = (
                    Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
                * Rotation.from_quat(self.curr_end_pos[3:])
            ).as_quat()
        except Exception as e:
            print(f"calculate action error {action}, exception {e}")

        return gripper_action, target_end_pos



def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global gl_stop_state
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        print_green(f"Eval ckpt {FLAGS.eval_checkpoint_step}")
        agent = agent.replace(state=ckpt)
        use_rl_to_finish_task = False
        robot_interface = env.robot
        expert = SpaceMouseExpert()
        rate = rospy.Rate(10)
        while True:
            while gl_stop_state.value:
                time.sleep(0.5)
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            sampling_rng, key = jax.random.split(sampling_rng)

            # 判断是否接近目标
            if use_rl_to_finish_task:  # 假设您有这个判断函数
                # 使用RL策略
                actions = agent.sample_actions(
                    observations=jax.device_put(obs), argmax=False, seed=key
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
            else:
                # 使用规则策略
                if close_to_target(robot_interface):
                    use_rl_to_finish_task = True
                    actions = np.asarray(jax.device_get([0.0]*6))
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs
                else:
                    gripper, pos = rule_based_policy(robot_interface, expert)  # 假设这是您的规则策略
                    robot_interface.send_pos_command(pos)
                    robot_interface.send_gripper_command(gripper)
            rate.sleep()

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit


import posix_ipc
import mmap
import ctypes

# 定义共享内存和信号量名称（需唯一）
SHM_NAME = "/shared_state"
SEM_NAME = "/pause_state"

def init_shared_state():
    """初始化共享内存和信号量"""
    # 创建共享内存（存储一个布尔值）
    shm = posix_ipc.SharedMemory(SHM_NAME, flags=posix_ipc.O_CREAT, size=ctypes.sizeof(ctypes.c_bool))
    # 映射共享内存
    shared_data = mmap.mmap(shm.fd, shm.size)
    os.close(shm.fd)  # 关闭文件描述符，mmap仍有效

    # 初始化信号量（初始值为1，表示未锁定）
    sem = posix_ipc.Semaphore(SEM_NAME, flags=posix_ipc.O_CREAT, initial_value=1)
    return shared_data, sem

def cleanup_shared_state(shared_data, sem):
    """清理资源"""
    try:
        # 尝试关闭资源
        shared_data.close()
        sem.close()
    finally:
        try:
            posix_ipc.unlink_shared_memory(SHM_NAME)
            posix_ipc.unlink_semaphore(SEM_NAME)
        except posix_ipc.ExistentialError:
            pass  # 忽略已释放的错误

gl_stop_state = None
gl_sem = None
gl_stop_learner = True
gl_listener_thread = None
def start_key_listen_thread():
    global gl_stop_state, gl_sem, gl_listener_thread
    def on_press(key):
        global gl_stop_state, gl_sem
        try:
            if key == keyboard.Key.space:
                gl_sem.acquire()
                if gl_stop_state.value:
                    print(Fore.GREEN + "Resume training" + Style.RESET_ALL)
                    gl_stop_state.value = False
                else:
                    print(Fore.RED + "Stop training" + Style.RESET_ALL)
                    gl_stop_state.value = True
                gl_sem.release()
        except AttributeError:
            pass

    def start_keyboard_listener():
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    gl_listener_thread = threading.Thread(
        target=start_keyboard_listener, daemon=True
    )
    gl_listener_thread.start()


def handler(sig, frame):
    global gl_stop_state, gl_sem, shared_data
    gl_stop_state = False
    cleanup_shared_state(shared_data, gl_sem)
    sys.exit(0)

def main(_):
    signal.signal(signal.SIGINT, handler)
    global gl_stop_state, gl_sem, shared_data
    shared_data, gl_sem = init_shared_state()
    gl_stop_state = ctypes.c_bool.from_buffer(shared_data)

    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)
    robot_interface = RobotRosInterface
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        robot_interface,
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier= False,
    )
    print(f"env observe space {env.observation_space}")
    print(f"env action space {env.action_space}")
    rng, sampling_rng = jax.random.split(rng)

    agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and has_checkpoint_folder(FLAGS.checkpoint_path):
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

   start_key_listen_thread()
   actor(
       agent,
       data_store,
       intvn_data_store,
       env,
       sampling_rng,
   )

if __name__ == "__main__":
    app.run(main)
