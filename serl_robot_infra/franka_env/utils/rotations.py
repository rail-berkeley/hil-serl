from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion


def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements
