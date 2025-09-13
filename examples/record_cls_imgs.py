import copy
import os
from tqdm import tqdm
import numpy as np
import datetime
import cv2
from absl import app, flags
from pynput import keyboard
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer(
    "successes_needed", 400, "Number of successful transistions to collect."
)

success_key = False
failed_key = False
quit = False

# 定义保存路径
SUCCESS_DIR = "/home/amigo/jiaqi/yolov/dataset2/success"
FAIL_DIR = "/home/amigo/jiaqi/yolov/dataset2/fail"

def create_directories():
    """创建保存图像的目录"""
    os.makedirs(SUCCESS_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)

def save_image_from_transition(obs, save_dir, image_type, index):
    """从transition中提取并保存图像"""
    image_array = obs["side"][0]      
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
            
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{image_type}_{index:06d}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
            
    result = cv2.imwrite(filepath, image_array)
    return True
    
    

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
            quit = True
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
    
    create_directories()
    
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, sample=True)
    obs, _ = env.reset()
    print("finished reset reset")
    
    success_needed = FLAGS.successes_needed
    success_count = 0
    failure_count = 0
    
    pbar = tqdm(total=success_needed)
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        
        if "intervene_action" in info:
            actions = info["intervene_action"]
        
        obs = next_obs
        
        if success_key:
            save_image_from_transition(obs, SUCCESS_DIR, "success", success_count)
            success_count += 1
            pbar.update(1)
            success_key = False
            
        elif failed_key:
            save_image_from_transition(obs, FAIL_DIR, "fail", failure_count)
            failure_count += 1
            print("add negative")
        if quit:
            break
    
    print(f"\nImage saving summary:")
    print(f"  Success images saved: {success_count} -> {SUCCESS_DIR}")
    print(f"  Failure images saved: {failure_count} -> {FAIL_DIR}")

if __name__ == "__main__":
    app.run(main)