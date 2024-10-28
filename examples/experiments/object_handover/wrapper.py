import copy
import threading
import time
from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv

class HandOffEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resetpos = np.concatenate([self.config.RESET_POSE[:3], R.from_euler("xyz", self.config.RESET_POSE[3:]).as_quat()])

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side"]
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = R.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign_x = np.sign(euler[0])
        euler[0] = sign_x * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        sign_z = np.sign(euler[2])
        euler[2] = sign_z * (
            np.clip(
                np.abs(euler[2]),
                self.rpy_bounding_box.low[2],
                self.rpy_bounding_box.high[2],
            )
        )

        euler[1] = np.clip(
            euler[1], self.rpy_bounding_box.low[1], self.rpy_bounding_box.high[1]
        )
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose
    
    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > 2.0):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > 2.0):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.1)

        # Move up before resetting
        current_pose = self.currpos.copy()
        current_pose[2] = max(current_pose[2], self.config.RESET_POSE[2])
        self.interpolate_move(current_pose, timeout=1)
        requests.post(self.url + "open_gripper")
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            raise NotImplementedError("Joint reset is not implemented")

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=0.5)
            time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

        input("Press Enter to continue...")


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
