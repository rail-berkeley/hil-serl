import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
import sys
from absl import app, flags
from pynput import keyboard
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer(
    "successes_needed", 200, "Number of successful transistions to collect."
)


success_key = False
failed_key = False
quit=False

def on_press(key):
    global success_key
    global failed_key
    global quit
    try:
        if key.char == 's':
            success_key = True
            failed_key = False
        if key.char == 'f':
            success_key = False
            failed_key = True
        if key.char == 'q':
            quit=True
    except AttributeError:
        pass

def on_release(key):
    global failed_key
    try:
        if key.char == 'f':
            failed_key = False
    except AttributeError:
        pass


def main(_):
    global success_key
    global failed_key
    global quit
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = (
        env.reset()
    ) 
    print("finished reset reset") # obs: currpos, currvel, curr_gripper_pos, currforce, currtorque
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
        #print(f"action: {actions}")
        print(f"press success key {success_key}")
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
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        elif failed_key:
            print("insert failed")
            failures.append(transition)

        # if done or truncated:
        #     obs, _ = env.reset()
        if quit:
            break
    
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = (
        f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    )
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")


if __name__ == "__main__":
    app.run(main)
