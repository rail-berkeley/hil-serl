import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard

from franka_env.envs.franka_env import FrankaEnv

class RAMEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # perform Cartesian reset
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
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


    def regrasp(self):
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place RAM in holder and press enter to grasp...")
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05
        top_pose[0] += np.random.uniform(-0.005, 0.005)
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.interpolate_move(grasp_pose, timeout=0.5)

        requests.post(self.url + "close_gripper_slow")
        self.last_gripper_act = time.time()
        time.sleep(2)

        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)

        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)


    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if True:
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}