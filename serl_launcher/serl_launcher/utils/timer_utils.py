"""Timer utility."""

import time
from collections import defaultdict


class _TimerContextManager:
    def __init__(self, timer: "Timer", key: str):
        self.timer = timer
        self.key = key

    def __enter__(self):
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.tock(self.key)


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def context(self, key):
        """
        Use this like:

        with timer.context("key"):
            # do stuff

        Then timer.tock("key") will be called automatically.
        """
        return _TimerContextManager(self, key)

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret
