import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard
import gymnasium as gym
from franka_sim.mujoco_gym_env import MujocoGymEnv

import mujoco
import mujoco.viewer


class FrankaSimEnv(gym.Wrapper):
    show_viewer: bool

    def __init__(
            self,
            env,
            action_scale: list,
            show_viewer: bool = True,
            fake_env: bool = False,
        ):
        assert isinstance(env, MujocoGymEnv)
        super().__init__(env)
        self.fake_env = fake_env
        m = env.model
        d = env.data
        self.action_scale = np.array(action_scale)
        self.show_viewer = show_viewer
        if self.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(m, d)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)


    def step(self, action):
        assert action.shape == (4,)
        action[:3] *= self.action_scale
        step_start = time.time()
        vals = super().step(action)
        if self.show_viewer:
            self.viewer.sync()
        time_until_next_step = self.env.control_dt - (time.time() - step_start)
        if time_until_next_step > 0 and not self.fake_env:
            time.sleep(time_until_next_step)
        return vals