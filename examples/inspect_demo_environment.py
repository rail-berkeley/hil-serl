# sudo /home/qirico/miniconda3/envs/hilserl3/bin/python examples/inspect_demo_environment.py 

from absl import app, flags
import time
import numpy as np
import os
import pickle
import imageio
import cv2
import queue
from pynput import keyboard
import threading

from experiments.ram_insertion.config import EnvConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import EnvConfig as USBPickupInsertionTrainConfig
from experiments.cube_reach.config import EnvConfig as CubeReachTrainConfig
from experiments.cube_reach2.config import EnvConfig as CubeReach2TrainConfig
from experiments.cube_reach3.config import EnvConfig as CubeReach3TrainConfig

ENV_CONFIG_MAPPING = {
    "ram_insertion": RAMInsertionTrainConfig,
    "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    "cube_reach": CubeReachTrainConfig,
    "cube_reach2": CubeReach2TrainConfig,
    "cube_reach3": CubeReach3TrainConfig,
}

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.zed_capture import ZedCapture

from run_demos import pull_demos

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name: str, keys: list[str], imgsize: int):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name
        self.keys = keys
        self.imgsize = imgsize

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            xn = len(self.keys)
            yn = len(self.keys[0])
            sqsize = self.imgsize
            margin = 5
            cellxl = sqsize + margin * 2
            cellyl = sqsize + margin * 2

            frame = np.zeros((cellxl * xn, cellyl * yn, 3), dtype=np.uint8)
            for x in range(xn):
                for y in range(yn):
                    k = self.keys[x][y]
                    cur_frame = np.full((cellxl, cellyl, 3), 255, dtype=np.uint8)
                    cur_frame[margin : margin + sqsize, margin : margin + sqsize, :] = cv2.resize(img_array[k], (sqsize, sqsize))
                    frame[x * cellxl : (x + 1) * cellxl, y * cellyl : (y + 1) * cellyl, :] = cur_frame

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

FLAGS = flags.FLAGS

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_cyan(x):
    return print("\033[96m {}\033[00m".format(x))

quit_key = False
def on_press(key):
    global quit_key
    try:
        if str(key) == "'q'":
            print_yellow("quit")
            quit_key = True
    except AttributeError:
        print("error")
        pass

def _get_exp_name(config_mapping: dict, default_exp_name: str = "cube_reach3"):
    print()
    print("Please provide a exp_name from the following:")
    print(f"{list(config_mapping.keys())}")
    exp_name = input(f"['{default_exp_name}'] ")
    if exp_name == "":
        exp_name = default_exp_name
    print(exp_name)
    return exp_name

def _get_demo_file_name(default_demo_file_name: str = "/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/cube_reach3/side_only/demo_data/cube_reach3_20_demos_2025-04-25_09-41-14.pkl"):    
    print()
    print("Please provide a demo_file_name that ends with .pkl:")
    demo_file_name = input(f"['{default_demo_file_name}'] ")
    if demo_file_name == "":
        demo_file_name = default_demo_file_name
    assert demo_file_name.endswith(".pkl")
    print(demo_file_name)

    return demo_file_name

def main(_):
    global quit_key

    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    exp_name = _get_exp_name(ENV_CONFIG_MAPPING)
    config = ENV_CONFIG_MAPPING[exp_name]()

    demo_file_name = _get_demo_file_name()
    demos = pull_demos(demo_file_name)
    assert len(demos) > 0

    demo_images = demos[0]["observations"]
    del demo_images['state']
    assert set(demo_images.keys()) == set(config.REALSENSE_CAMERAS.keys())
    for k in demo_images.keys():
        if demo_images[k].ndim == 4 and demo_images[k].shape[0] == 1:
            demo_images[k] = demo_images[k][0]

    keys = [
        ['side_1_cur', 'side_1_orig', 'side_1_avg', 'side_1_diff']
    ]
    displaysize = 400

    img_queue = queue.Queue()
    displayer = ImageDisplayer(img_queue, config.SERVER_URL, keys=keys, imgsize=displaysize)
    displayer.start()

    caps = {}
    for cam_name, kwargs in config.REALSENSE_CAMERAS.items():
        assert "camera_type" in kwargs
        camera_type = kwargs['camera_type']
        del kwargs['camera_type']
        if camera_type == "rs":
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
        elif camera_type == "zed":
            cap = VideoCapture(
                ZedCapture(name=cam_name, **kwargs)
            )
        else:
            raise NotImplementedError
        caps[cam_name] = cap

    while True:
        if quit_key:
            break
        images = {}
        for key in caps.keys():
            cur_image = cap.read()
            cur_image = config.IMAGE_CROP[key](cur_image) if key in config.IMAGE_CROP else cur_image
            cur_image = cv2.resize(cur_image, demo_images[key].shape[:2])
            demo_image = demo_images[key][:,:,::-1]
            images[key + "_cur"] = cur_image
            images[key + "_orig"] = demo_image
            images[key + "_avg"] = np.uint8((np.float32(cur_image) + np.float32(demo_image)) / 2)
            images[key + "_diff"] = np.uint8(np.abs(np.float32(cur_image) - np.float32(demo_image)) / 2)
        img_queue.put(images)
        time.sleep(0.1)

    img_queue.put(None)
    cv2.destroyAllWindows()
    displayer.join()

if __name__ == "__main__":
    app.run(main)
