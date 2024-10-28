import queue
import threading
import gymnasium as gym
from gymnasium.core import Env
from copy import deepcopy
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
import cv2

class FrontCameraWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        front_obs_space = {
            k: space for k, space in self.observation_space.items() if "wrist" not in k
        }

        self.front_observation_space = gym.spaces.Dict(front_obs_space)
        # self.observation_space = gym.spaces.Dict(new_obs_space)
        self.front_obs = None

    def observation(self, observation):
        # cache a copy of observation with only the front camera image
        new_obs = deepcopy(observation)
        new_obs.pop("wrist_1")
        self.front_obs = new_obs

        return observation

    def get_front_cam_obs(self):
        return self.front_obs

class NewFrontCameraWrapper(gym.ObservationWrapper):
    """
    This wrapper is using front camera to train classifier only. The wrapped env
    should have a front camera as part of the observation space. The resultant env
    should not have a front camera as part of the observation space. The front camera
    image should be saved and retrieved by get_front_cam_obs method.
    """
    def __init__(self, env: Env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Dict({
        #     k: space for k, space in self.observation_space.items() if "side" not in k
        # })
        self.front_obs = None
        self.cap = VideoCapture(
                RSCapture(name="side", serial_number="128422272758", depth=False)
            )
        # self.img_queue = queue.Queue()
        # self.displayer = ImageDisplayer(self.img_queue, "reward_image")
        # self.displayer.start()

    def observation(self, observation):
        # cache a copy of observation with only the front camera image
        image = self.cap.read()
        image = image[180:300, 170:280]
        image = cv2.resize(image, (128, 128))[None, ...]
        self.front_obs = deepcopy(image)
        # observation.pop("side")
        return observation

    def get_front_cam_obs(self):
        return self.front_obs


# class ImageDisplayer(threading.Thread):
#     def __init__(self, queue, name):
#         threading.Thread.__init__(self)
#         self.queue = queue
#         self.daemon = True  # make this a daemon thread
#         self.name = name

#     def run(self):
#         while True:
#             img_array = self.queue.get()  # retrieve an image from the queue
#             if img_array is None:  # None is our signal to exit
#                 break
#             print(img_array.sum())
#             cv2.imshow(self.name, img_array)
#             cv2.waitKey(1)