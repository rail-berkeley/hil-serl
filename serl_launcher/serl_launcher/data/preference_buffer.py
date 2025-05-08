import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import jax
import numpy as np
from serl_launcher.data.dataset import Dataset, DatasetDict
from serl_launcher.data.replay_buffer import _init_replay_dict, _insert_recursively


class PreferenceBuffer(Dataset):
    def __init__(self, pre_obs_space: gym.Space, post_obs_space: gym.Space, a_pi_space: gym.Space, a_exp_space: gym.Space, capacity: int):
        pre_obs_data = _init_replay_dict(pre_obs_space, capacity)
        post_obs_data = _init_replay_dict(post_obs_space, capacity)

        dataset_dict = dict(
            pre_obs=pre_obs_data,
            post_obs=post_obs_data,
            a_pi=np.empty((capacity, *a_pi_space.shape), dtype=a_pi_space.dtype),
            a_exp=np.empty((capacity, *a_exp_space.shape), dtype=a_exp_space.dtype),
            t=np.empty((capacity,), dtype=np.float32)
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch