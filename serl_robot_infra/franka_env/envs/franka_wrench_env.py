"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.multi_video_capture import MultiVideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (300, 300)) for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultWrenchEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": {"serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 7000,
            },
        "side": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 7000,
        },
    }
    ACTION_SCALE = np.array([13, 0, 12, 0, 3.5, 0, 0])
    RANDOM_RESET = False

    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.0
    MAX_EPISODE_LENGTH: int = 100
    WAIT_FOR_RESET: bool = False
    IMAGE_CROP: dict[str, callable] = {}

##############################################################################


class FrankaWrenchEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultWrenchEnvConfig = DefaultWrenchEnvConfig(),
    ):
        self.action_scale = config.ACTION_SCALE
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        # TODO: Pass reset pose from config to franka server and avoid hardcoding 
        # self.resetpos = config.RESET_POSE

        self._update_currpos()
        
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.wait_for_reset = config.WAIT_FOR_RESET
        self.hz = hz

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.episode_num = 0
        self.recording_frames = []

        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        state_space_dict = {
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),  # xyz + quat
            "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            "q": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
            "dq": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
            "gripper_pose": gym.spaces.Box(0, 1, shape=(1,)),
        }

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    state_space_dict
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in config.REALSENSE_CAMERAS}
                ),
            }
        )

        if not fake_env:
            self.cap = None
            self.init_cameras(config.REALSENSE_CAMERAS)
            if self.display_image:
                self.img_queue = queue.Queue()
                self.displayer = ImageDisplayer(self.img_queue, self.url)
                self.displayer.start()

            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            print("Initialized Franka")

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action * self.action_scale 
        self._send_wrench_command(action[:6])

        self.curr_path_length += 1
        # TODO: Make sleep time changeable for dynamic tasks
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        t = time.time()
        self._update_currpos()
        ob = self._get_obs()
        # print(f"Time to update currpos and get obs: {time.time() - t}")
        reward = 0
        # TODO: End trajectory early if unsafe velocity etc.?
        safety_exceeded = False 
        done = self.curr_path_length >= self.max_episode_length or reward or safety_exceeded or self.terminate
        return ob, int(reward), done, False, {}

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        try:
            all_frames = self.cap.read()
            for key, rgb in all_frames.items():
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
        except queue.Empty:
            input(
                "Cameras frozen. Check connections, then press enter to relaunch..."
            )
            self.cap.close()
            self.init_cameras(self.config.REALSENSE_CAMERAS)
            return self.get_im()

        # Store full resolution cropped images separately
        self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def reset(self, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()
        self.episode_num += 1

        if self.randomreset:
            # TODO: Enable random reset for joint angles
            raise NotImplementedError
        else:
            requests.post(self.url + "reset")

        if self.wait_for_reset:
            input("Press enter to continue...")

        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        caps = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            caps[cam_name] = RSCapture(name=cam_name, **kwargs)

        self.cap = MultiVideoCapture(caps)

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def _send_wrench_command(self, wrench: np.ndarray):
        """Internal function to send wrench command to the robot."""
        data = {"arr": wrench.astype(np.float32).tolist()}
        requests.post(self.url + "wrench", json=data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos >= -1) and (pos <= -0.5) and (self.curr_gripper_pos > 0.95) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(0.6)
            elif (pos >= 0.5) and (pos <= 1) and (self.curr_gripper_pos < 0.95) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(0.6)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos = np.array(ps["pose"])
        self.currvel = np.array(ps["vel"])

        self.currforce = np.array(ps["force"])
        self.currtorque = np.array(ps["torque"])
        self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

        self.q = np.array(ps["q"])
        self.dq = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
            "q": self.q,
            "dq": self.dq,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))
