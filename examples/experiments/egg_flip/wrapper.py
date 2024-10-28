import copy
import gymnasium as gym
import jax
import numpy as np

from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time

class EggClassifierWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_classifier_func: callable, confidence_threshold: float = 0.9):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.confidence_threshold = confidence_threshold
        new_space = copy.deepcopy(self.env.observation_space)
        new_shape = list(new_space["state"].shape)
        new_shape[-1] += 1
        new_space["state"] = gym.spaces.Box(-np.inf, np.inf, new_shape)
        self.observation_space = new_space

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.egg_initial, _ = self.compute_reward(obs)
        while self.egg_initial == 2:
            input("We lost the egg!!! Put egg back and press Enter... ")
            obs, info = self.env.reset()
            self.egg_initial, _ = self.compute_reward(obs)
        obs['state'] = np.concatenate((obs['state'], np.array([self.egg_initial])[None]), axis=-1)
        info['succeed'] = 0
        return obs, info

    def compute_reward(self, obs):
        logits = self.reward_classifier_func(obs)
        class_probs = jax.nn.softmax(logits)
        egg_class = np.argmax(class_probs).item()
        confidence = np.max(class_probs).item()
        return egg_class, confidence

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        egg_class, confidence = self.compute_reward(obs)
        info["succeed"] = 0
        if confidence >= self.confidence_threshold:
            if (egg_class == 1 and self.egg_initial == 0) or (egg_class == 0 and self.egg_initial == 1):
                print(f"Egg class: {egg_class}, Confidence: {confidence}")
                rew = 1
                done = True
                info["succeed"] = 1
        obs['state'] = np.concatenate((obs['state'], np.array([egg_class])[None]), axis=-1)
        return obs, rew, done, truncated, info
    
class EggFlipActionWrapper(gym.ActionWrapper):
    """Only control translation x, z, and rotation y."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,))

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[0] = action[0]
        new_action[2] = action[1]
        new_action[4] = action[2]
        return new_action

class EggFlipSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        expert_a = np.array([expert_a[0], expert_a[2], expert_a[4]])
        
        self.left, self.right = tuple(buttons)

        if np.linalg.norm(expert_a) > 0.001:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
    