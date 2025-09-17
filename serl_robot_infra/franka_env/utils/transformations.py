from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    # adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

def construct_adjoint_matrix_from_euler(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_euler("xyz", tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix_from_euler(tcp_pose, order="xyz"):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_euler(order, tcp_pose[3:6]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T

def calculate_tool_tip_with_matrix(current_pose, tool_length=0.152, x_offset=0.01):
    """
    使用齐次变换矩阵计算工具末端位置，并额外加上工具的X正方向偏移
    
    参数:
        current_pose: [x, y, z, qx, qy, qz, qw] - TCP位姿
        tool_length: 工具长度（沿TCP的Z轴方向）
        x_offset: 工具X正方向的额外偏移量
    
    返回:
        tool_tip_pose: [x, y, z, qx, qy, qz, qw] - 工具末端位姿
    """
    # 提取位置和四元数
    position = np.array(current_pose[:3])
    quaternion = np.array(current_pose[3:])
    
    # 创建旋转矩阵
    rotation = R.from_quat(quaternion)
    Rm = rotation.as_matrix()
    
    # 构建齐次变换矩阵 T_tcp_base (TCP相对于基座标系)
    T_tcp_base = np.eye(4)
    T_tcp_base[:3, :3] = Rm
    T_tcp_base[:3, 3] = position
    
    # 构建工具末端的变换矩阵 T_tool_tcp (工具末端相对于TCP)
    # 先沿TCP的Z轴平移tool_length，再沿TCP的X轴平移x_offset
    T_tool_tcp = np.eye(4)
    T_tool_tcp[:3, 3] = [x_offset, 0, tool_length]  # [x, y, z] 偏移
    
    # 计算工具末端相对于基座标系的变换矩阵
    T_tool_base = T_tcp_base @ T_tool_tcp
    
    # 提取工具末端的位置和姿态
    tool_tip_position = T_tool_base[:3, 3]
    tool_tip_rotation = R.from_matrix(T_tool_base[:3, :3])
    tool_tip_quat = tool_tip_rotation.as_quat()  # [x, y, z, w] 格式
    
    return np.concatenate([tool_tip_position, tool_tip_quat])