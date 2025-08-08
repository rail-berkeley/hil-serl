import copy
import threading
import time
from typing import OrderedDict
from collections import deque
from colorama import Fore, Style
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64
from std_msgs.msg import Header

from tf.transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换
from tf.transformations import euler_from_quaternion  # 用于欧拉角到四元数的转换

# from franka_env.utils.rotations import euler_2_quat
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from agilex_env.agilex_env import AgilexEnv, DefaultEnvConfig

from gym import Env
from experiments.object_handover_agilex.transformers import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)

import ctypes
import posix_ipc
import mmap


class DualAgilexEnv(AgilexEnv):
    def __init__(
        self,
        left_env_config: DefaultEnvConfig = None,
        right_env_config: DefaultEnvConfig = None,
        fake_env=False,
        save_video=False,
    ):

        self.env_left = AgilexEnv(
            config=left_env_config, fake_env=fake_env, save_video=save_video
        )

        # Action/Observation Space
        action_dim = len(self.env_left.action_space.low) + len(
            self.env_right.action_space.low
        )
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
        image_dict = {
            f"left/{key}": self.env_left.observation_space["images"][key]
            for key in self.env_left.observation_space["images"].keys()
        } | {
            f"right/{key}": self.env_right.observation_space["images"][key]
            for key in self.env_right.observation_space["images"].keys()
        }

        state_dict = {
            f"left/{key}": self.env_left.observation_space["state"][key]
            for key in self.env_left.observation_space["state"].keys()
        } | {
            f"right/{key}": self.env_right.observation_space["state"][key]
            for key in self.env_right.observation_space["state"].keys()
        }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(image_dict),
            }
        )
        self.display_images = False
        if self.display_images:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()

    def step(self, action: np.ndarray) -> tuple:
        action_left = action[: len(action) // 2]
        action_right = action[len(action) // 2 :]

        def step_env_left():
            global ob_left, reward_left, done_left
            ob_left, reward_left, done_left, _, _ = self.env_left.step(action_left)

        def step_env_right():
            global ob_right, reward_right, done_right
            ob_right, reward_right, done_right, _, _ = self.env_right.step(action_right)

        # Create threads for each function
        thread_left = threading.Thread(target=step_env_left)
        thread_right = threading.Thread(target=step_env_right)

        # Start the threads
        thread_left.start()
        thread_right.start()

        # Wait for both threads to complete
        thread_left.join()
        thread_right.join()
        ob = self.combine_obs(ob_left, ob_right)
        if self.display_images:
            self.img_queue.put(ob["images"])
        return ob, int(reward_left and reward_right), done_left or done_right, False, {}

    def reset(self, **kwargs):
        def reset_env_left():
            global ob_left
            ob_left, _ = self.env_left.reset(**kwargs)

        def reset_env_right():
            global ob_right
            ob_right, _ = self.env_right.reset(**kwargs)

        thread_left = threading.Thread(target=reset_env_left)
        thread_right = threading.Thread(target=reset_env_right)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        ob = self.combine_obs(ob_left, ob_right)
        return ob, {}

    def combine_obs(self, ob_left, ob_right):
        left_images = {
            f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()
        }
        right_images = {
            f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()
        }
        left_state = {
            f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()
        }
        right_state = {
            f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()
        }
        ob = {"state": left_state | right_state, "images": left_images | right_images}
        return ob

class ForcePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, thresh=10, penalty_coef=1e-3, max_penalty=3e-3):
        '''
        - force_threshold: float, acceptable force magnitude threshold
        - penalty_coef: float, penalty scaling factor (lambda)
        '''
        super().__init__(env)
        self.thresh = thresh
        self.penalty_coef = penalty_coef    
        self.max_penalty = max_penalty

    def compute_penalty(self, force_vector):
        """
        Compute reward based on force feedback and task completion status.

        Parameters:
        - force_vector: np.array of shape (3,), the force readings [Fx, Fy, Fz]

        Returns:
        - reward: float, final reward value
        """
        # Calculate L2 norm of the force vector
        force_norm = np.linalg.norm(force_vector)
        
        # Calculate penalty if force exceeds threshold
        excess_force = max(0.0, force_norm - self.thresh)
        penalty = self.penalty_coef * excess_force
        penalty = min(self.max_penalty, penalty)
        print(f"force norm: {force_norm}")
        print(f"force penalty: {penalty}")
        return -penalty

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        force_vector = self.env.curr_force[:3]
        penalty = self.compute_penalty(force_vector)
        # reward *= 100
        reward += penalty
        print(f"reward: {reward}")
        return observation, reward, terminated, truncated, info
    
class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        '''
        - force_threshold: float, acceptable force magnitude threshold
        - penalty_coef: float, penalty scaling factor (lambda)
        '''
        super().__init__(env)
        self.start_time = time.time()
    
    def reset(self, **kwargs):
        """重置时记录开始时间"""
        self.start_time = time.time()
        return self.env.reset(**kwargs)

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            time_used = min(20, time.time() - self.start_time)  # max 20 s
            reward += 1/time_used
            print(f'time used: {time_used} s')
            print(f"reward: {reward}")
        return observation, reward, terminated, truncated, info

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_left_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_left_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        state = observation["state"]
        # print("state shape:", state.shape)

        current_grip_status  = int(state[0, -1])

        info["grasp_penalty"] = 0.0
        # if (action[6] < -0.5 and self.last_left_gripper_pos > 0.85 and (current_grip_status in (0, 3))) or (
        #     action[6] > 0.5 and self.last_left_gripper_pos < 0.85 and current_grip_status in (2, 1)
        # ):
        if (action[6] <= 0 and current_grip_status == 3) or (
            action[6] > 0.5 and current_grip_status == 2
        ):
            info["grasp_penalty"] += self.penalty
            print("left grasp penalty")

        self.last_left_gripper_pos = observation["state"][0, 0]

        return observation, reward, terminated, truncated, info

class JitterPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_coef=0.1, max_penalty=1, history_length=5):
        """
        抖动惩罚包装器
        
        Parameters:
        - penalty_coef: float, 抖动惩罚系数
        - max_penalty: float, 最大惩罚值
        - history_length: int, 用于计算抖动的历史动作长度
        """
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.max_penalty = max_penalty
        self.history_length = history_length
        
        # 存储历史动作和位置信息
        self.action_history = deque(maxlen=history_length)
        self.tcp_pos_history = deque(maxlen=history_length)
        self.tcp_vel_history = deque(maxlen=history_length)

    def compute_jitter_penalty(self, current_action, current_tcp_pos, current_tcp_vel):
        """
        计算抖动惩罚
        
        Returns:
        - penalty: float, 抖动惩罚值（负数）
        """
        if len(self.action_history) < 2:
            return 0.0
        
        penalty = 0.0
        
        # 1. 动作变化率惩罚（检测动作的突然变化）
        if len(self.action_history) >= 2:
            action_diff1 = np.array(current_action) - np.array(self.action_history[-1])
            action_diff2 = np.array(self.action_history[-1]) - np.array(self.action_history[-2])
            
            # 计算动作加速度（二阶导数）
            action_acceleration = action_diff1 - action_diff2
            action_jerk = np.linalg.norm(action_acceleration[:6])  # 只考虑位置和旋转，不包括夹爪
            
            # 动作抖动惩罚
            if action_jerk > 0.05:  # 阈值可以调整
                penalty += self.penalty_coef * action_jerk
        
        # 2. 位置抖动惩罚
        if len(self.tcp_pos_history) >= 2:
            pos_current = np.array(current_tcp_pos[:3])  # 只取位置部分
            pos_last = np.array(self.tcp_pos_history[-1][:3])
            
            if len(self.tcp_pos_history) >= 3:
                pos_prev = np.array(self.tcp_pos_history[-2][:3])
                
                # 计算位置的二阶导数（加速度）
                vel_current = pos_current - pos_last
                vel_last = pos_last - pos_prev
                pos_acceleration = vel_current - vel_last
                pos_jerk = np.linalg.norm(pos_acceleration)
                
                # 位置抖动惩罚
                if pos_jerk > 0.05:  # 阈值可以调整
                    penalty += self.penalty_coef * pos_jerk 
        
        # 3. 速度突变惩罚
        if len(self.tcp_vel_history) >= 2:
            vel_current = np.array(current_tcp_vel[:3])  # 线性速度
            vel_last = np.array(self.tcp_vel_history[-1][:3])
            
            vel_change = np.linalg.norm(vel_current - vel_last)
            
            # 速度突变惩罚
            if vel_change > 0.05:  # 阈值可以调整
                penalty += self.penalty_coef * vel_change
        
        # 限制最大惩罚
        penalty = min(self.max_penalty, penalty)*(-1)
        
        if penalty < 0:
            print(f"Jitter penalty: {penalty:.4f}")
        
        return penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取当前状态信息
        if hasattr(self.env, 'curr_end_pos'):
            current_tcp_pos = self.env.curr_end_pos
        else:
            # 从observation中获取TCP位置
            if "state" in observation:
                if "tcp_pose" in observation["state"]:
                    current_tcp_pos = observation["state"]["tcp_pose"]
                elif "left/tcp_pose" in observation["state"]:
                    current_tcp_pos = observation["state"]["left/tcp_pose"]
                else:
                    current_tcp_pos = np.zeros(7)
            else:
                current_tcp_pos = np.zeros(7)
        
        # 获取TCP速度
        if hasattr(self.env, 'curr_end_vel'):
            current_tcp_vel = self.env.curr_end_vel
        else:
            if "state" in observation:
                if "tcp_vel" in observation["state"]:
                    current_tcp_vel = observation["state"]["tcp_vel"]
                elif "left/tcp_vel" in observation["state"]:
                    current_tcp_vel = observation["state"]["left/tcp_vel"]
                else:
                    current_tcp_vel = np.zeros(6)
            else:
                current_tcp_vel = np.zeros(6)
        
        # 计算抖动惩罚
        jitter_penalty = self.compute_jitter_penalty(action, current_tcp_pos, current_tcp_vel)
        
        # 更新历史记录
        self.action_history.append(action.copy())
        self.tcp_pos_history.append(current_tcp_pos.copy())
        self.tcp_vel_history.append(current_tcp_vel.copy())
        
        # 应用惩罚
        reward += jitter_penalty
        
        # 添加调试信息
        if "jitter_penalty" not in info:
            info["jitter_penalty"] = jitter_penalty
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """重置时清空历史记录"""
        self.action_history.clear()
        self.tcp_pos_history.clear()
        self.tcp_vel_history.clear()
        return self.env.reset(**kwargs)



import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class DualRelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
            }
        ),
        ......
    }, and at least 12 DoF action space
    """

    def __init__(self, env: Env, include_relative_pose=True):
        print("init relative")
        super().__init__(env)
        self.left_adjoint_matrix = np.zeros((6, 6))
        self.right_adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        # print(f"transformed_action: {transformed_action}")
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update adjoint matrix
        self.left_adjoint_matrix = construct_adjoint_matrix(
            obs["state"]["left/tcp_pose"]
        )
        self.right_adjoint_matrix = construct_adjoint_matrix(
            obs["state"]["right/tcp_pose"]
        )

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix
        self.left_adjoint_matrix = construct_adjoint_matrix(
            obs["state"]["left/tcp_pose"]
        )
        self.right_adjoint_matrix = construct_adjoint_matrix(
            obs["state"]["right/tcp_pose"]
        )

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            )
            self.right_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        left_adjoint_inv = np.linalg.inv(self.left_adjoint_matrix)
        obs["state"]["left/tcp_vel"] = left_adjoint_inv @ obs["state"]["left/tcp_vel"]

        right_adjoint_inv = np.linalg.inv(self.right_adjoint_matrix)
        obs["state"]["right/tcp_vel"] = (
            right_adjoint_inv @ obs["state"]["right/tcp_vel"]
        )

        if self.include_relative_pose:
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

            # Reconstruct transformed tcp_pose vector
            left_p_b_r = left_T_b_r[:3, 3]
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o

            # Reconstruct transformed tcp_pose vector
            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right/tcp_pose"] = np.concatenate(
                (right_p_b_r, right_theta_b_r)
            )

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        if len(action) == 12:
            action[:6] = self.left_adjoint_matrix @ action[:6]
            action[6:] = self.right_adjoint_matrix @ action[6:]
        elif len(action) == 14:
            action[:6] = self.left_adjoint_matrix @ action[:6]
            action[7:13] = self.right_adjoint_matrix @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        if len(action) == 12:
            action[:6] = np.linalg.inv(self.left_adjoint_matrix) @ action[:6]
            action[6:] = np.linalg.inv(self.right_adjoint_matrix) @ action[6:]
        elif len(action) == 14:
            action[:6] = np.linalg.inv(self.left_adjoint_matrix) @ action[:6]
            action[7:13] = np.linalg.inv(self.right_adjoint_matrix) @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action


class DualExpertIntervWrapper(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        if self.AgilexMasterControlMode:
            left_action = self.env_left.curr_end_pos
            euler = euler_from_quaternion(action[3:])
            left_action[3:6] = euler
            left_action[6] = self.env_left.curr_gripper_pos
            right_action = self.env_right.curr_end_pos
            euler = euler_from_quaternion(action[3:])
            right_action[3:6] = euler
            normalized_pos = self.env_right.curr_gripper_pos*13.3
            if normalized_pos > 0.85:
                right_action[6] = 1 
            elif normalized_pos < 0.4:
                right_action[6] = -1
            else:
                right_action[6] = 0
            print(right_action[6], " ", normalized_pos)
            action = np.concatenate(left_action, right_action)
            return action, True

    def step(self, action):
        self.env.control_arm = True
        new_action, replaced = self.action(action)
        if replaced:
            self.env.control_arm = False

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


from pynput import keyboard


class ExpertIntervWrapper(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)
        # Subscribe master node seem it as expert
        if rospy.get_node_uri():
            print("Node has been initialzed already")
        else:
            rospy.init_node(f"hilserl_agilex_env_wrapper", anonymous=True)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
        self.action_indices = action_indices

        self.expert_pos_deque = deque()
        self.expert_gripper_deque = deque()

        self.expert_pos_sub = rospy.Subscriber(
            "/left/end_pose_quat",
            Pose,
            self.update_expert_pos,
            queue_size=100,
            tcp_nodelay=True,
        )
        self.expert_gripper_sub = rospy.Subscriber(
            "/left/gripper_angle",
            Float64,
            self.update_expert_gripper,
            queue_size=100,
            tcp_nodelay=True,
        )
        self.AgilexMasterControlMode = False
        self.has_align_expert = False
        self.listener_thread = threading.Thread(
            target=self.start_keyboard_listener, daemon=True
        )
        self.has_success = False
        self.has_done = False
        self.listener_thread.start()

    def on_press(self, key):
        try:
            #if key == keyboard.Key.space:
            #    if self.AgilexMasterControlMode:
            #        print(Fore.RED + "Exit interve mode" + Style.RESET_ALL)
            #        self.AgilexMasterControlMode = False
            #        self.has_align_expert = False
            #        self.expert_gripper_deque.clear()
            #        self.expert_pos_deque.clear()
            #    else:
            #        print(Fore.GREEN + "Enter interve mode" + Style.RESET_ALL)
            #        self.AgilexMasterControlMode = True
            if key.char == 'd':
                self.has_done = True
            if key.char == 's':
                print(f"set success {self.has_success}")
                self.has_success = True 
            if key.char == 'o':
                print("open gripper")
                self._lift_and_open_gripper()
            if key.char == 'p':
                print("close gripper")
                self._close_gripper_and_reset()
        except AttributeError:
            print(f"特殊按键: {key}")

    def _lift_and_open_gripper(self):
        """机械臂上升5厘米并张开夹爪"""
        try:
            # 获取当前位置
            current_pos = self.env.robot.curr_end_pos.copy()
            # 上升5厘米
            lift_pos = current_pos.copy()
            lift_pos[2] += 0.05  # 上升5厘米
            
            # 发送位置命令
            self.env.robot.send_pos_command(lift_pos)
            time.sleep(0.5)  # 等待移动完成
            
            # 张开夹爪
            self.env.robot.send_gripper_command(1.0)  # 1.0表示张开
            time.sleep(0.5)  # 等待夹爪动作完成
            
            # print(f"机械臂已上升至: {lift_pos[:3]}, 夹爪已张开")
        except Exception as e:
            print(f"执行上升和张开夹爪操作时出错: {e}")

    def _close_gripper_and_reset(self):
        """闭合夹爪并回到reset位置"""
        try:
            # 先闭合夹爪
            self.env.robot.send_gripper_command(-1.0)  # -1.0表示闭合
            time.sleep(1.0)  # 等待夹爪闭合
            
            # 回到reset位置
            reset_pos = self.env.resetpos.copy()
            self.env.interpolate_move(reset_pos, timeout=2.0)
            
            # print(f"夹爪已闭合，机械臂已回到reset位置: {reset_pos[:3]}")
        except Exception as e:
            print(f"执行闭合夹爪和回到reset位置操作时出错: {e}")

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def reset(self, **kwargs):
        self.has_done = False
        self.has_success = False
        print("reset has done")
        return self.env.reset(**kwargs)

    def update_expert_pos(self, msg):
        if len(self.expert_pos_deque) >= 100:
            self.expert_pos_deque.popleft()
        pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        self.expert_pos_deque.append(pose)

    def update_expert_gripper(self, msg):
        if len(self.expert_gripper_deque):
            self.expert_gripper_deque.popleft()
        self.expert_gripper_deque.append(msg.data)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        if self.AgilexMasterControlMode:
            # Wait for expert arm align to action arm
            print(Fore.GREEN + "ready sleep" + Style.RESET_ALL)
            wait_start = time.time()
            while True:
                action_arm_pos = np.array(self.env.curr_end_pos)
                #if len(self.expert_pos_deque) == 0:
                #    return action, False
                expert_arm_pos = [] #self.expert_pos_deque[-1]
                print(f"wraper action {self.has_align_expert}")
                print(f"wraper expert {expert_arm_pos}")
                if not self.has_align_expert and not self.is_pos_close(
                    action_arm_pos, expert_arm_pos
                ):
                    time.sleep(0.5)
                else:
                    self.has_align_expert = True
                    break
                if not self.AgilexMasterControlMode or time.time() - wait_start > 20.0:
                    return action, False

            # action_arm_pos = np.array(self.expert_pos_deque[-(min(40, len(self.expert_pos_deque)))])
            rot_expert = R.from_quat(expert_arm_pos[3:])
            rot_curr = R.from_quat(action_arm_pos[3:])
            euler = (rot_expert*rot_curr.inv()).as_euler('xyz')
            euler = euler/self.env.action_scale[1]
            expert_action = expert_arm_pos
            expert_action[:3] = (expert_action[:3] - action_arm_pos[:3])/self.env.action_scale[0]
            expert_action[3:6] = euler #[euler[1], euler[2], euler[0]]
            if len(self.expert_gripper_deque) > 0:
                normalized_pos = self.expert_gripper_deque[-1]*13.3
                if normalized_pos > 0.85:
                    expert_action[6] = 1 
                elif normalized_pos < 0.4:
                    expert_action[6] = -1
                else:
                    expert_action[6] = 0
            else:
                expert_action[6] = action[6]

            return expert_action, True
        else:
            return action, False

    def is_pos_close(self, q1, q2):
        return False
        """
        计算两个四元数之间的旋转角度误差（弧度）。
        q1, q2: [qx, qy, qz, qw] 形式的四元数
        """
        quat_1 = np.array(q1[3:])
        quat_2 = np.array(q2[3:])
        dot_product = np.dot(quat_1, quat_2)  # 计算四元数点积
        dot_product = np.clip(dot_product, -1.0, 1.0)  # 避免浮点误差导致 acos 出错
        rot_error = 2 * np.arccos(abs(dot_product))  # 计算旋转角度（弧度）

        """
        判断两个位姿是否接近
        pose1, pose2: [x, y, z, qx, qy, qz, qw] 形式的位姿数据
        pos_threshold: 位置接近的阈值（米）
        rot_threshold: 旋转角度接近的阈值（度）
        """
        # 计算位置误差
        pos1, pos2 = np.array(q1[:3]), np.array(q2[:3])
        pos_error = np.linalg.norm(pos1 - pos2)
        print(f"pos error { pos_error}, rot error {rot_error}")
        # 判断是否接近
        is_close = pos_error < 0.1 and rot_error < 0.3
        return is_close

    def step(self, action):
        self.env.control_arm = True
        new_action, replaced = self.action(action)
        if replaced:
            # self.env.control_arm = False
            print(f"get interve action {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        print(f"has_interved {replaced}")
        rew = max(rew, 1 if self.has_success else 0)
        done = done or self.has_done or rew

        return obs, rew, done, truncated, info
