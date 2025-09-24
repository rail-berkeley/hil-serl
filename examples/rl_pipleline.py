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
from collections import deque
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.utils.timer_utils import Timer
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from experiments.plugusb.config import RightEnvConfig as envcfg
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent_hybrid_single_arm,
)

from experiments.mappings import CONFIG_MAPPING
from ros_robot_env.ros_robot_interface import RobotRosInterface
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from franka_env.utils.transformations import calculate_offset_toolcoordinate
flags.DEFINE_multi_string("exp_name", None, "Names of experiments corresponding to folders.")
flags.DEFINE_multi_string("checkpoint_path", None, "Paths to save checkpoints.")
flags.DEFINE_multi_string("demo_path", None, "Paths to the demo data.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_multi_integer("eval_checkpoint_step", None, "Steps to evaluate the checkpoint.")  # Changed to multi integer
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # Debug mode will disable wandb logging
flags.DEFINE_integer("seed", 40, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
FLAGS = flags.FLAGS


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
def ensure_same_hemisphere(waypoints: np.ndarray) -> np.ndarray:
    """
    输入: waypoints (N,7)，每行为 [x,y,z,qx,qy,qz,qw]
    功能: 归一化四元数，并确保相邻四元数处于同一半球（相邻点点积 >= 0；若 <0 则将后者取反）
    输出: 处理后的新数组（不修改原数组）
    """
    wps = np.array(waypoints, dtype=float, copy=True)
    if wps.ndim != 2 or wps.shape[1] != 7:
        raise ValueError("waypoints 必须是形状 (N,7) 的数组 [x,y,z,qx,qy,qz,qw]")

    quats = wps[:, 3:]  # (N,4) -> (x,y,z,w)

    # 归一化
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("检测到零长度四元数")
    quats /= norms

    # 同半球处理：让相邻点积 >= 0
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    wps[:, 3:] = quats
    return wps
def move_orthogonal(env,timeout=2.0,x_offset=0,y_offset=0,z_offset=0):
    """
    正交运动
    """
    curr_ep=env.robot.curr_ep
    goal=calculate_offset_toolcoordinate(curr_ep,x_offset=x_offset,y_offset=y_offset,z_offset=z_offset)
    timeout=cal_timeout(curr=curr_ep,goal=goal)
    steps=int(timeout*env.hz)
    path = np.linspace(curr_ep[:3], goal[:3], steps)
    for i in range(steps):
        pos = path[i]
        env.robot.send_pos_command(np.concatenate([pos, curr_ep[3:]]))
        time.sleep(0.2)
    time.sleep(0.5)

def cal_timeout(curr, goal):
        # 提取四元数
        q_curr = np.array(curr[3:])
        q_goal = np.array(goal[3:])
        # 计算点积
        dot_original = np.dot(q_curr, q_goal)
        # 调整：如果点积为负，将q_goal取反
        if dot_original < 0:
            q_goal = -q_goal
            dot_adjusted = -dot_original
        else:
            dot_adjusted = dot_original

        goal[3:] = q_goal.tolist()   
        # 计算角度差
        dot_adjusted = np.clip(dot_adjusted, -1.0, 1.0)   # 确保在[-1,1]内
        theta = 2 * np.arccos(dot_adjusted)
        delta=np.linalg.norm(curr[:3]-goal[:3])
        print(f"theta: {theta}, delta: {delta}")
        timeout = theta*3+delta*15
        print(f"timeout: {timeout}")
        return np.max((timeout,2),axis=0)
def interpolate_move(env, waypoints, timeout=2.0,min_segment_time=0.5):
        """
        多关键点平滑插值运动（纯归一化时间轴 + 权重分配）
        :param waypoints: 关键帧列表 [[x,y,z,qx,qy,qz,qw], ...]
        :param timeout: 总运动时间 (秒)
        """
        waypoints = np.array(waypoints)
        waypoints = ensure_same_hemisphere(waypoints)
        print("wwwwwww",waypoints[:,:3])
        if waypoints.shape[1] != 7:
            raise ValueError("姿态必须是7维数组 [x,y,z,qx,qy,qz,qw]")
        timeout=cal_timeout(waypoints[0],waypoints[-1])
        # 1. 计算每段运动的平移距离和旋转角度
        segment_distances = []
        segment_angles = []
        for i in range(len(waypoints) - 1):
            # 平移距离 (米)
            delta_pos = np.linalg.norm(waypoints[i+1, :3] - waypoints[i, :3])
            segment_distances.append(delta_pos)
            
            # 旋转角度 (弧度)
            dot = np.dot(waypoints[i, 3:], waypoints[i+1, 3:])
            dot = np.clip(dot, -1.0, 1.0)
            theta = 2 * np.arccos(abs(dot))
            segment_angles.append(theta)

        # 2. 计算归一化的动态时间轴（基于position×10 + rotation×2）
        segment_time = [
        max((d * 15) + (a * 3), min_segment_time)
        for d, a in zip(segment_distances, segment_angles)]
        total_time= sum(segment_time)
        # 生成归一化的时间轴 [0, ..., 1]
        key_times = np.cumsum([0] + segment_time) / total_time
        # 3. 位置插值 (动态时间轴 + 三次样条)
        pos_interpolator = interp1d(
            key_times,
            waypoints[:, :3],
            kind="linear",
            axis=0
        )
        # 4. 姿态插值 (Slerp)
        Rs = R.from_quat(waypoints[:, 3:])
        slerp = Slerp(key_times, Rs)
        # 5. 生成插值点并发送指令
        steps = int(timeout * 10)
        t_eval = np.linspace(0, 1, steps)  # 归一化的插值时间轴
        for t in t_eval:
            pos = pos_interpolator(t)    # 位置插值  
            rot = slerp(t)    
            print(f"t: {t}, pos: {pos}, euler: {rot.as_euler('xyz')}")            
            env.robot.send_pos_command(np.concatenate([pos, rot.as_quat()]))
            time.sleep(0.1)
        return env.robot.curr_ep
def release_lift(env, inverse=False, z_offset=-0.03):
    """
    释放夹爪
    """
    # 如果 inverse 为 True，先进行正交移动
    if inverse and z_offset != 0:
        move_orthogonal(env, z_offset=z_offset)

    # 发送夹爪命令
    env.robot.send_gripper_command(1.0)
    
    # 等待夹爪释放
    while env.robot.curr_gripper_pos[0] < 0.5:
        time.sleep(0.1)
    
    # 如果 inverse 为 False，才进行正交移动
    if not inverse and z_offset != 0:
        move_orthogonal(env, z_offset=z_offset)

    # 返回当前末端位置
    return env.robot.curr_ep

def grasp_decrease(env, inverse=False, z_offset=0.03):
    """
    夹紧夹爪
    """
    # 如果 inverse 为 True，先进行正交移动
    if inverse and z_offset != 0:
        move_orthogonal(env, z_offset=z_offset)

    # 发送夹爪命令，进行夹紧
    env.robot.send_gripper_command(-1)

    # 等待夹爪完全夹紧
    time.sleep(1)

    # 如果 inverse 为 False，才进行正交移动
    if not inverse and z_offset != 0:
        move_orthogonal(env, z_offset=z_offset)

    # 返回当前末端位置
    return env.robot.curr_ep
def gen_waypoints(start_pose,end_pose, movement_config):
    """
    使用部分关键字匹配来检查动作类型
    start quat
    end euler
    """
    waypoints = [start_pose.copy()]
    current_pose = start_pose.copy()
    print(f"start pose: {current_pose}")
    goal_pose=end_pose.copy()
    euler=R.from_quat(goal_pose[3:]).as_euler('xyz')
    for action in movement_config:
        # 检查所有键，使用部分匹配
        for key in action.keys():
            if 'connect2end' in key:
                current_pose[:3] = (current_pose[:3] +goal_pose[:3])/2
            elif 'move_x' in key:
                current_pose[0] += action[key]
            
            elif 'move_y' in key:
                current_pose[1] += action[key]
            
            elif 'move_z' in key:
                current_pose[2] += action[key]
            elif 'roll_reset' in key:
                temp=R.from_quat(current_pose[3:]).as_euler('xyz')
                temp[0]=euler[0]
                current_pose=np.concatenate([current_pose[:3], R.from_euler('xyz', temp).as_quat()])
            elif 'pitch_reset' in key:
                temp=R.from_quat(current_pose[3:]).as_euler('xyz')
                temp[1]=euler[1]
                current_pose=np.concatenate([current_pose[:3], R.from_euler('xyz', temp).as_quat()])
            elif 'yaw_reset' in key:
                temp=R.from_quat(current_pose[3:]).as_euler('xyz')
                temp[2]=euler[2]
                current_pose=np.concatenate([current_pose[:3], R.from_euler('xyz', temp).as_quat()])
            # 处理 tool_x, tool_y, tool_z
            elif 'tool_x' in key:
                current_pose[0] = calculate_offset_toolcoordinate(current_pose, x_offset=action[key])[0]

            elif 'tool_y' in key:
                current_pose[1] = calculate_offset_toolcoordinate(current_pose, y_offset=action[key])[1]

            elif 'tool_z' in key:
                current_pose[2] = calculate_offset_toolcoordinate(current_pose, z_offset=action[key])[2]
        waypoints.append(current_pose.copy())
    waypoints.append(goal_pose)
    print(f"waypoints: {waypoints}")
    return waypoints

def actor(agent, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global goaldeque
    global gl_stop_state
    if FLAGS.eval_checkpoint_step:
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        print_green(f"Eval ckpt {FLAGS.eval_checkpoint_step}")
        agent = agent.replace(state=ckpt)
        env.robot.reset_robot_force_config()
        time.sleep(0.2)
        done = False
        while reward<=0:
            while gl_stop_state.value:
                time.sleep(0.5)
            done = False
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, _ = agent.sample_actions(
                    observations=jax.device_put(obs), argmax=True, seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                finished_pos=obs["state"]["tcp_pose"]
            obs, _ = env.reset()
        goaldeque.append(finished_pos)   
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

movement_config=[{   'tool_z': -0.02},
                {   'connect2end': True, 
                       'tool_z': -0.03,
                       'roll_reset': True
                }]
movement_config2=[{'pitch_reset': True},
    {   'tool_x': 0.015,'tool_z':-0.005},
    {   'tool_z':-0.02},
                 {   'connect2end': True, 
                       'tool_z': -0.02,
                       'roll_reset': True,
                       'pitch_reset': True
                 }]
goaldeque=deque(maxlen=10)
gl_stop_state = None
gl_sem = None
gl_stop_learner = True
gl_listener_thread = None
init_pos=[[0.4397,-0.2374,0.2259,0,1,0,0],[0.4307, -0.15356, 0.17261, 0.0, 1.0, 0.0, 0.0]]
resetpose=[]
def main(_):
    signal.signal(signal.SIGINT, handler)
    global gl_stop_state, gl_sem, shared_data
    shared_data, gl_sem = init_shared_state()
    gl_stop_state = ctypes.c_bool.from_buffer(shared_data)

    exp_names = FLAGS.exp_name  # List of experiment names
    checkpoint_paths = FLAGS.checkpoint_path  # List of checkpoint paths
    eval_steps = FLAGS.eval_checkpoint_step  # List of eval steps

    # Prepare the experiments and configurations
    configs = []
    for exp_name in exp_names:
        config = CONFIG_MAPPING[exp_name]()  
        configs.append(config)
        resetpose.append(np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        ))

    # 初始化 RNG
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # 创建环境（只创建第一个实验的环境）
    config = configs[0]  # 使用第一个实验的配置来创建环境
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier= True,
    )
    robot = RobotRosInterface(env.config)
    env.set_robot(robot)
    print(f"env observe space {env.observation_space}")
    print(f"env action space {env.action_space}")

    # 创建智能体列表（两个智能体都使用同一个环境）
    agents = []
    # for i, exp_name in enumerate(exp_names):
    #     # 获取当前实验的配置
    #     current_config = configs[i]
        
    #     # 创建 SAC agent（每个智能体使用同一个环境）
    #     agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
    #         seed=FLAGS.seed,
    #         sample_obs=env.observation_space.sample(),
    #         sample_action=env.action_space.sample(),
    #         image_keys=current_config.image_keys,
    #         encoder_type=current_config.encoder_type,
    #         discount=current_config.discount,
    #     )
    #     # replicate agent across devices
    #     agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
        
    #     # 处理 checkpoint
    #     checkpoint_path = checkpoint_paths[i]
    #     if checkpoint_path is not None and has_checkpoint_folder(checkpoint_path):
    #         input("Checkpoint path already exists. Press Enter to resume training.")
    #         ckpt = checkpoints.restore_checkpoint(
    #             os.path.abspath(checkpoint_path),
    #             agent.state,
    #         )
    #         agent = agent.replace(state=ckpt)
    #         ckpt_number = os.path.basename(
    #             checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    #         )[11:]
    #         print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
    #     agents.append(agent)
    start_key_listen_thread()
    #run pipleline
    curr_pos=env.robot.curr_ep
    print(f"start at {curr_pos}")
    curr_pos=release_lift(env)
    waypoints=gen_waypoints(curr_pos,init_pos[0], movement_config)
    curr_pos=interpolate_move(env,waypoints)
    time.sleep(0.5)
    curr_pos=grasp_decrease(env,inverse=True)
    #grasped cable to plug it in
    waypoints=gen_waypoints(curr_pos,resetpose[0], movement_config)
    curr_pos=interpolate_move(env,waypoints)
    print(f"move to platte to grasp cable {curr_pos}")
    time.sleep(0.5)
    #move platte
    # actor(
    #    agents[0],
    #    env,
    #    sampling_rng,
    # )
    print(f"after plugcable finished lift to {curr_pos}")
    waypoints=gen_waypoints(curr_pos,init_pos[0], movement_config)
    curr_pos=interpolate_move(env,waypoints)
    print(f"move to cablebase to release cable {curr_pos}")
    time.sleep(0.5)
    curr_pos=grasp_decrease(env)
    print(f"cable fixed at {curr_pos}")
    #move cablebase
    curr_pos=release_lift(env)
    print(f"moved up to safety{curr_pos}")
    waypoints=gen_waypoints(curr_pos,init_pos[1], movement_config)
    curr_pos=interpolate_move(env,waypoints)
    print(f"move to usbbase to grasp usb {curr_pos}")
    # env.update_config(envcfg)
    curr_pos=grasp_decrease(env,z_offset=-0.05)
    waypoints=gen_waypoints(curr_pos,resetpose[1], movement_config)
    curr_pos=interpolate_move(env,waypoints)
    time.sleep(0.5)
    print(f"move to platte to plug usb {curr_pos}")
    rng, sampling_rng = jax.random.split(rng)
    time.sleep(0.5)
    #move platte
    # actor(
    #    agents[1],
    #    env,
    #    sampling_rng,
    # )
    temp=init_pos[1]
    temp[2]+=0.03
    waypoints=gen_waypoints(curr_pos,temp, movement_config2)
    curr_pos=interpolate_move(env,waypoints)
    time.sleep(0.5)
    curr_pos=release_lift(env,z_offset=0.03,inverse=True)
    # #move usbbase
    # # org=goaldeque.popleft
    # # org[2]+=0.03
    # # waypoints=gen_waypoints(curr_pos,org, movement_config)
    # # curr_pos=interpolate_move(env,waypoints)
    # # print(f"move to platte to grasp cable {curr_pos}")
    # # curr_pos=grasp_decrease(env,z_offset=0.03,inverse=True)
    # # print(f"grasp cable at{curr_pos}")
    # # waypoints=gen_waypoints(curr_pos,org, movement_config)
    # # curr_pos=interpolate_move(env,waypoints)
    # print(f"finished at {curr_pos}")
if __name__ == "__main__":
    app.run(main)
