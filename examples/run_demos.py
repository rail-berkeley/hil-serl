from absl import app, flags
import time
import numpy as np
import pickle

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("demo_path", None, "Path to the demos.")

def pull_demos(demo_path: str):
    with open(demo_path, "rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, list)
    demos = []
    for transition in obj:
        assert isinstance(transition, dict)
        assert set(["observations", "actions", "next_observations", "dones"]).issubset(set(transition.keys())), f"{transition.keys()}"
        demos.append(transition)
    return demos

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    assert FLAGS.demo_path is not None
    demos = pull_demos(FLAGS.demo_path)

    obs, info = env.reset()
    print("reset start")
    time.sleep(5.0)
    print("reset end")

    done = False
    truncated = False
    i = 0
    while not (done or truncated):
        actions = demos[i]['actions']
        t1 = time.time()
        next_obs, rew, done, truncated, info = env.step(actions)
        t2 = time.time()
        print(f"Robot operating at {1 / (t2 - t1)} Hz.")
        obs = next_obs
        if demos[i]['dones']:
            break
        i += 1


if __name__ == "__main__":
    app.run(main)
