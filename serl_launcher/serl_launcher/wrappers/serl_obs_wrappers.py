import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        # print(self.proprio_keys)
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        # print(f"obs: {obs}")
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs
class RemoveImageKeyWrapper(gym.ObservationWrapper):
    """
    Remove specific image keys from flattened observation
    """
    def __init__(self, env, keys_to_remove=["wrist2cls"]):
        super().__init__(env)
        self.keys_to_remove = keys_to_remove
        
        # 重新定义observation_space，去掉不要的键
        self.observation_space = gym.spaces.Dict({
            key: space for key, space in env.observation_space.spaces.items()
            if key not in self.keys_to_remove
        })
    
    def observation(self, obs):
        # 直接过滤掉不要的键
        return {k: v for k, v in obs.items() if k not in self.keys_to_remove}
