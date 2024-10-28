import math
from collections import defaultdict
from typing import Dict

import gymnasium as gym
import jax
import numpy as np


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_info(info):
    filter_keys = [
        "object_names",
        "target_object",
        "initial_positions",
        "target_position",
        "goal",
    ]
    for k in filter_keys:
        if k in info:
            del info[k]
    return info


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


def bootstrap_std(arr, f=np.mean, n=30):
    arr = np.array(arr)
    return np.std([f(arr[np.random.choice(len(arr), len(arr))]) for _ in range(n)])


def parallel_evaluate(policy_fn, eval_envs, num_eval, verbose=True):
    n_envs = len(eval_envs.reset())
    eval_episode_rewards = []
    eval_episode_time_rewards = []
    counter = np.zeros(n_envs)

    obs = eval_envs.reset()
    if verbose:
        print("Evaluating Envs")
    n_per = int(math.ceil(num_eval / n_envs))
    n_to_eval = n_per * n_envs
    while len(eval_episode_rewards) < n_to_eval:
        action = policy_fn(obs)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        for n, info in enumerate(infos):
            if "episode" in info.keys() and counter[n] < n_per:
                eval_episode_rewards.append(info["episode"]["r"])
                eval_episode_time_rewards.append(info["episode"]["time_r"])
                counter[n] += 1
    if verbose:
        print(
            f"Evaluation using {len(eval_episode_rewards)} episodes: mean reward {np.mean(eval_episode_rewards):.5f} +- {bootstrap_std(eval_episode_rewards):.5f} \n"
        )
    return eval_episode_rewards, eval_episode_time_rewards
