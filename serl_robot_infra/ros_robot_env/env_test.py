from robot_env import RobotRosEnv
from robot_env import DefaultEnvConfig
import numpy as np

import time

from tf.transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换
from tf.transformations import euler_from_quaternion  # 用于欧拉角到四元数的转换
from geometry_msgs.msg import Pose

class LeftEnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    ARM = "left"
    CAMERAS = {
        "wrist_1":"image_l/camer_color/raw",
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }

env = RobotRosEnv(
    config=LeftEnvConfig(), fake_env=False, save_video=False
)
count = 0
while True:
    if count == 0:
        position = [-0.2, 0.0, 0.307, 191.0, 0.0, 0.0, 0]
        euler = [3.140, 0.0, 0.0] #[position[a] * 3.1415926 / 180 for a in range(3, 6)]
        env._send_gripper_command(20, "f")
    elif count == 400:
        position = [-0.307, -0.07, 0.305, 0.0, 0.0, 0, 0]
        euler = [3.14, 0.0, 0.0] #[position[a] * 3.1415926 / 180 for a in range(3, 6)]
        env._send_gripper_command(200, "f")
    elif count == 800:
        count = -1
    else:
        time.sleep(0.01)
        count += 1
        continue

    pose = np.concatenate([position[:3], quaternion_from_euler(euler[0], euler[1], euler[2])])
    print(f"euler: {euler}")
    quat = quaternion_from_euler(euler[0], euler[1], euler[2])
    print(f"quat: {quat}")
    new_euler = euler_from_quaternion(quat)
    print(f"new euler: {new_euler}")
    #env._send_pos_command(pose)
    time.sleep(0.01)
    count += 1
