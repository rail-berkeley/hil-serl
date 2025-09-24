

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import os
import time
import copy
import threading

from flax.training import checkpoints
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class  RLTask:
    def __init__(self, config):
        self.config = config
        # TODO: Use param
        self.name = config['task_name']
        self.rng = jax.random.PRNGKey(40)
        self.rl_config = CONFIG_MAPPING[self.config['task_name']]()
        self.env = self.rl_config.get_environment(
            fake_env=False,
            save_video=False,
            classifier= True,
        )
        self.robot = None
        self.build_agent(config)
        self.finish_task_pos = None
        self.finish_rule_based_movement = False

    def build_agent(self, config):
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=40,
            sample_obs=self.env.observation_space.sample(),
            sample_action=self.env.action_space.sample(),
            image_keys=self.rl_config.image_keys,
            encoder_type=self.rl_config.encoder_type,
            discount=self.rl_config.discount,
        )
        # replicate agent across devices
        devices = jax.local_devices()
        sharding = jax.sharding.PositionalSharding(devices)
        self.agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(self.config['checkpoint_path']),
            self.agent.state,
            step=self.config['checkpoint_step'],
        )
        print_green(f"Use ckpt {self.config['checkpoint_path']} + 'step' {self.config['checkpoint_step']}")
        self.agent = self.agent.replace(state=ckpt)

        # warm up agent
        obs = self.env.observation_space.sample()
        actions, _ = self.agent.sample_actions(
            observations=jax.device_put(obs), argmax=True, seed=self.rng
        )

    def run(self, robot):
        self.robot = robot
        self.robot.update_config(self.env.config)
        self.env.set_robot(robot)
        self.env.enable_movement(False)
        rl_thread = threading.Thread(target=self.finish_target_with_rl)
        rl_thread.start()
        self.robot.set_force_config()
        self.rule_based_approach_target()
        time.sleep(1)
        self.env.enable_movement(True)
        self.env.reset()
        time.sleep(1)
        rl_thread.join()
        return True

    def rule_based_approach_target(self):
        # 1. Grab target object
        # 2. Move to target position
        #
        self.robot.send_gripper_command(1.0)
        waypoints = []
        curr_pos = self.robot.curr_ep
        move_config = self.config['rule_based_move_config']
        waypoints.append(curr_pos)
        waypoints.append(move_config['point_above_object'])
        waypoints.append(move_config['point_object'])
        self.interpolate_move(waypoints)

        # Grasp object
        self.robot.send_gripper_command(-1.0)
        time.sleep(1.0)
        waypoints.clear()
        curr_pos = self.robot.curr_ep
        waypoints.append(curr_pos)
        waypoints.append(move_config['point_above_object'])
        waypoints.append(move_config['point_above_target'])
        self.interpolate_move(waypoints)

    def interpolate_move(self, waypoints, timeout=2.0,min_segment_time=0.5):
            """
            多关键点平滑插值运动（纯归一化时间轴 + 权重分配）
            :param waypoints: 关键帧列表 [[x,y,z,qx,qy,qz,qw], ...]
            :param timeout: 总运动时间 (秒)
            """
            waypoints = np.array(waypoints)
            waypoints = self.transto_same_hemisphere(waypoints)
            if waypoints.shape[1] != 7:
                raise ValueError("姿态必须是7维数组 [x,y,z,qx,qy,qz,qw]")
            timeout = self.cal_timeout(waypoints[0],waypoints[-1])
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
            steps = int(timeout*2)
            t_eval = np.linspace(0, 1, steps)  # 归一化的插值时间轴
            for t in t_eval:
                pos = pos_interpolator(t)    # 位置插值
                rot = slerp(t)
                print("pos",pos)
                self.robot.movel(np.concatenate([pos, rot.as_quat()]))
                time.sleep(0.7)
            time.sleep(1)
            # self.robot.reset_robot_force_config()

    def transto_same_hemisphere(self, waypoints: np.ndarray) -> np.ndarray:
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

    def cal_timeout(self,curr, goal):
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

    def finish_target_with_rl(self):
        #run_time = 5
        #while run_time > 0:
        #    rand_action = np.random.rand(7)*0.02
        #    self.env.step(rand_action)
        #    print("Simulate rl............")
        #    run_time -= 1
        #    time.sleep(0.7)
        #print("Finish RL part")
        #time.sleep(5)
        #return True
        obs, reward, done, truncated, info = self.env.step([0.0]*7)
        self.rng, sampling_rng = jax.random.split(self.rng)
        while reward <= 0:
            done=False
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, _ = self.agent.sample_actions(
                    observations=jax.device_put(obs), argmax=True, seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, rew, done, truncated, info = self.env.step(actions)
                obs = next_obs
            reward=rew
            if reward<=0:
                self.env.reset()
        

    def log_current_pos(self):
        self.finish_task_pos = self.robot.curr_ep

    def move_arm_to_safe_pos(self):
        self.robot.send_gripper_command(1.0)
        time.sleep(0.5)
        target_pos = copy.copy(self.robot.curr_ep)
        for _ in range(2):
            target_pos[2] += 0.025
            self.robot.movel(target_pos)
            time.sleep(0.5)
        

    def reset_object(self):
        move_config = self.config['rule_based_move_config']
        waypoints = []
        curr_pos = self.robot.curr_ep
        # if curr far from finish task pos
        self.robot.send_gripper_command(1.0)
        waypoints.append(curr_pos)
        pose_above_finish_target = copy.copy(self.finish_task_pos)
        pose_above_finish_target[2] += 0.05
        waypoints.append(pose_above_finish_target)
        waypoints.append(self.finish_task_pos)
        self.interpolate_move(waypoints)
        self.robot.send_gripper_command(-1.0)
        time.sleep(0.5)

        #Unplug the object
        unplug_pos = copy.copy(self.finish_task_pos)
        unplug_pos[0] += move_config['unplug_offset'][0]
        unplug_pos[1] += move_config['unplug_offset'][1]
        unplug_pos[2] += move_config['unplug_offset'][2]
        self.robot.movel(unplug_pos)

        waypoints = []
        waypoints.append(unplug_pos)
        above_unplug_pos = copy.copy(unplug_pos)
        above_unplug_pos[2] += 0.05
        waypoints.append(above_unplug_pos)
        waypoints.append(move_config['point_above_object'])
        waypoints.append(move_config['point_object'])
        self.interpolate_move(waypoints)
        self.robot.send_gripper_command(1.0)
        self.move_arm_to_safe_pos()
