import copy
import threading
import time
from typing import OrderedDict

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


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_left_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_left_gripper_pos = obs["state"][0, 0]
        self.last_right_gripper_pos = obs["state"][0, 19]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        info["grasp_penalty"] = 0.0

        if (action[6] < -0.5 and self.last_left_gripper_pos > 0.85) or (
            action[6] > 0.5 and self.last_left_gripper_pos < 0.85
        ):
            info["grasp_penalty"] += self.penalty
            print("left grasp penalty")

        if (action[13] < -0.5 and self.last_right_gripper_pos > 0.85) or (
            action[13] > 0.5 and self.last_right_gripper_pos < 0.85
        ):
            info["grasp_penalty"] += self.penalty
            print("right grasp penalty")

        self.last_left_gripper_pos = observation["state"][0, 0]
        self.last_right_gripper_pos = observation["state"][0, 13]

        return observation, reward, terminated, truncated, info


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
        AgilexMasterControlMode = True
        if AgilexMasterControlMode:
            left_action = self.env_left.curr_end_pos
            euler = euler_from_quaternion(action[3:])
            left_action[3:6] = euler
            left_action[6] = self.env_left.curr_gripper_pos
            right_action = self.env_right.curr_end_pos
            euler = euler_from_quaternion(action[3:])
            right_action[3:6] = euler
            right_action[6] = self.env_right.curr_gripper_pos
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
