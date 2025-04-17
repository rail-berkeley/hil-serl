# How to use:
# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../record_success_fail.py --exp_name=cube_reach2 --successes_needed=200

import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


success_key = False
failure_key = False
def on_press(key):
    global success_key
    global failure_key
    try:
        if str(key) == 'Key.space':
            success_key = True
        elif str(key) == "f":
            failure_key = True
    except AttributeError:
        pass

def main(_):
    global success_key
    global failure_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    failure_file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"

    def dump_data(successes, failures):
        print("dumping...")
        with open(success_file_name, "wb") as f:
            pkl.dump(successes, f)
            print(f"saved {len(successes)} successful transitions to {success_file_name}")

        with open(failure_file_name, "wb") as f:
            pkl.dump(failures, f)
            print(f"saved {len(failures)} failure transitions to {failure_file_name}")
        print("dumped")

    in_success = False
    while True:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key or in_success:
            in_success = True
            successes.append(transition)
            pbar.update(1)
            success_key = False
        # elif failure_key:
        #     failures.append(transition)
        #     failure_key = False
        #     print("failure recorded")
        else:
            failures.append(transition)

        if done or truncated:
            in_success = False
            obs, _ = env.reset()
            print("reset start")
            dump_data(successes, failures)
            time.sleep(7.0)
            print("reset end")
            print("==")

    # except Exception as e:
    #     if not os.path.exists("./classifier_data"):
    #         os.makedirs("./classifier_data")
    #     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    #     with open(file_name, "wb") as f:
    #         pkl.dump(successes, f)
    #         print(f"saved {success_needed} successful transitions to {file_name}")

    #     file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    #     with open(file_name, "wb") as f:
    #         pkl.dump(failures, f)
    #         print(f"saved {len(failures)} failure transitions to {file_name}")
    
    return
        
if __name__ == "__main__":
    app.run(main)
