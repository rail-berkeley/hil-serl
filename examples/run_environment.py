from absl import app, flags
import time
import numpy as np

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    while True:
        obs, info = env.reset()
        print("reset start")
        time.sleep(5.0)
        print("reset end")

        done = False
        truncated = False
        while not (done or truncated):
            actions = np.zeros(env.action_space.sample().shape)
            t1 = time.time()
            next_obs, rew, done, truncated, info = env.step(actions)
            t2 = time.time()
            print(f"Robot operating at {1 / (t2 - t1)} Hz.")
            obs = next_obs


if __name__ == "__main__":
    app.run(main)
