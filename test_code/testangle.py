import numpy as np
from scipy.spatial.transform import Rotation

# 测试角度向量
omega = np.array([0.1, 0.05, 0.02])  # 弧度
print(f"输入角度向量 ω = {omega} (弧度)")
print(f"对应度数: {omega * 180 / np.pi}")

print("\n" + "="*50)

# 方法1: 轴角方式 (Twist - 旋转向量指数映射)
print("方法1: 轴角方式 (旋转向量指数映射)")
R_axis_angle = Rotation.from_rotvec(omega).as_matrix()
print("R_axis_angle =")
print(R_axis_angle)

print("\n" + "="*50)

# 方法2: 欧拉角方式 (XYZ顺序)
print("方法2: 欧拉角方式 (XYZ顺序)")
R_euler = Rotation.from_euler('XYZ', omega).as_matrix()
print("R_euler =")
print(R_euler)

print("\n" + "="*50)

# 计算差异
print("差异矩阵 (R_euler - R_axis_angle):")
diff = R_euler - R_axis_angle
print(diff)

print(f"\n最大元素差异: {np.max(np.abs(diff)):.2e}")
print(f"Frobenius范数: {np.linalg.norm(diff):.2e}")
print(f"相对误差: {np.linalg.norm(diff) / np.linalg.norm(R_axis_angle) * 100:.4f}%")

print("\n" + "="*50)

# 手动实现对比 (验证scipy结果)
print("手动实现验证:")

# 轴角方式 - 罗德里格斯公式
def axis_angle_to_matrix(omega):
    """轴角到旋转矩阵 - 罗德里格斯公式"""
    theta = np.linalg.norm(omega)
    if theta < 1e-6:
        return np.eye(3)
    
    k = omega / theta  # 单位轴
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])  # 反对称矩阵
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R

# 欧拉角方式 - 手动计算
def euler_xyz_to_matrix(angles):
    """欧拉角XYZ到旋转矩阵"""
    x, y, z = angles
    
    # 单轴旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    
    # R = Rz * Ry * Rx (外旋XYZ = 内旋ZYX)
    return Rz @ Ry @ Rx

R_axis_manual = axis_angle_to_matrix(omega)
R_euler_manual = euler_xyz_to_matrix(omega)

print("手动实现 - 轴角:")
print(R_axis_manual)
print("\n手动实现 - 欧拉角:")
print(R_euler_manual)

print(f"\n验证scipy一致性:")
print(f"轴角差异: {np.linalg.norm(R_axis_angle - R_axis_manual):.2e}")
print(f"欧拉角差异: {np.linalg.norm(R_euler - R_euler_manual):.2e}")

print("\n" + "="*50)

# 不同角度大小的误差分析
print("不同角度大小的误差分析:")
test_scales = [0.01, 0.05, 0.1, 0.2, 0.5]
base_omega = np.array([0.1, 0.05, 0.02])

print("角度倍数 | 最大角度(°) | Frob误差 | 相对误差(%)")
print("-" * 50)

for scale in test_scales:
    test_omega = base_omega * scale
    R_axis = Rotation.from_rotvec(test_omega).as_matrix()
    R_euler = Rotation.from_euler('XYZ', test_omega).as_matrix()
    
    diff = R_euler - R_axis
    frob_error = np.linalg.norm(diff)
    rel_error = frob_error / np.linalg.norm(R_axis) * 100
    max_angle_deg = np.max(test_omega) * 180 / np.pi
    
    print(f"{scale:8.2f} | {max_angle_deg:10.1f} | {frob_error:.2e} | {rel_error:.4f}")

print("\n结论:")
print("1. 两种方法产生不同的旋转矩阵")
print("2. 角度越小，差异越小，但永远不相等")
print("3. 原代码中的近似在小角度时是合理的工程选择")