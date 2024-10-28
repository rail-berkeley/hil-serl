from collections import OrderedDict
import gym
import numpy as np


class VideoWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
    ):
        super().__init__(env)
        self._name = name
        self._video = OrderedDict()
        self.image_keys = [k for k in self.observation_space.keys() if k != "state"]

    def get_obs_frames(self, keys=None):
        if keys is None:
            video = {k: np.array(v) for k, v in self._video.items()}
        else:
            video = {k: np.array(v) for k, v in self._video.items() if k in keys}
        return video

    def get_rendered_video(self):
        frames = []
        for i in range(len(self._video[self.image_keys[0]])):
            frame = []
            for k in self.image_keys:
                frame.append(self._video[k][i])
            frames.append(np.concatenate(frame, axis=1))
        return np.concatenate(frames, axis=0)

    def _add_frame(self, obs):
        img = []
        for k in self.image_keys:
            if k in obs:
                if k in self._video:
                    self._video[k].append(obs[k])
                else:
                    self._video[k] = [obs[k]]

    def reset(self, **kwargs):
        self._video.clear()
        obs, info = super().reset(**kwargs)
        self._add_frame(obs)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, done, truncate, info = super().step(action)
        self._add_frame(obs)
        return obs, reward, done, truncate, info
