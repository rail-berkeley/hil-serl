"""Gym Interface for Franka"""
import queue
import threading
import time
import numpy as np
import gymnasium as gym
import cv2
class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            left_frame = np.concatenate(
                [cv2.resize(v, (256, 256)) for k, v in img_array.items() if "left" in k], axis=1
            )
            right_frame = np.concatenate(
                [cv2.resize(v, (256, 256)) for k, v in img_array.items() if "right" in k], axis=1
            )
            frame = np.concatenate([left_frame, right_frame], axis=1)

            cv2.imshow('Image', frame[..., ::-1])
            cv2.waitKey(1)


class DualFrankaEnv(gym.Env):
    def __init__(
        self,
        env_left,
        env_right,
        display_images=True,
    ):

        self.env_left = env_left
        self.env_right = env_right

        # Action/Observation Space
        action_dim = len(self.env_left.action_space.low) + len(self.env_right.action_space.low)
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
        image_dict = ({f"left/{key}": self.env_left.observation_space["images"][key] for key in self.env_left.observation_space["images"].keys()} | 
                        {f"right/{key}": self.env_right.observation_space["images"][key] for key in self.env_right.observation_space["images"].keys()})

        state_dict = ({f"left/{key}": self.env_left.observation_space["state"][key] for key in self.env_left.observation_space["state"].keys()} |
                        {f"right/{key}": self.env_right.observation_space["state"][key] for key in self.env_right.observation_space["state"].keys()})
        
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                "images": gym.spaces.Dict(image_dict)
            }
        )
        self.display_images = display_images
        if self.display_images:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()

    def step(self, action: np.ndarray) -> tuple:
        action_left = action[:len(action)//2]
        action_right = action[len(action)//2:]
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
            self.img_queue.put(ob['images'])
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
        left_images = {f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()}
        right_images = {f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()}
        left_state = {f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()}
        right_state = {f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()}
        ob = {
                "state": left_state | right_state,
                "images": left_images | right_images
            }
        return ob