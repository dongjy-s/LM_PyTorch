"""机器人位姿误差计算函数

该模块提供了计算机器人正向运动学位姿与激光跟踪仪测量位姿之间误差的功能。

Returns:
    error_norm: 误差的2-范数
"""

#! 标准库导入
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.transform import Rotation
from fwd_kinematic import RokaeRobot

#! 常量定义
JOINT_ANGLE_FILE = 'data/joint_angle.csv'
LASER_POS_FILE = 'data/laser_pos.csv'
ERROR_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

#! DH参数求解工具变换矩阵
def get_fk_tool_matrix():
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    robot = RokaeRobot()
    fwd_tool_matrix = []
    
    # 计算工具正运动学
    for i, joint_angle in enumerate(joint_angles):
        result = robot.calculate_tool_matrix(joint_angle)
        if result['valid']:
            fwd_tool_matrix.append(result['transform_matrix'])
        else:
            raise ValueError(f"关节角度索引 {i} 无效: {result['error_msg']}")
    
    # 整体转换为 ndarray，输出形状 (样本数,4,4)
    fwd_tool_matrix = np.array(fwd_tool_matrix)
    return fwd_tool_matrix

#! 激光跟踪仪工具位姿变换矩阵
def get_laser_tool_matrix():
    laser_data = pd.read_csv(LASER_POS_FILE, delimiter=',', skiprows=1, header=None).values
    
    num_samples = laser_data.shape[0]
    laser_tool_matrix = np.zeros((num_samples, 4, 4))
    
    for i, data in enumerate(laser_data):
        x, y, z, rx, ry, rz = data
        
        # 计算旋转矩阵
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        
        # 创建变换矩阵
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        
        laser_tool_matrix[i] = T
    
    return laser_tool_matrix

#! 计算DH计算姿态与激光测量姿态之间的相对旋转
def rotation_matrix_to_axis_angle(R):
    # 创建Rotation对象
    try:
        rot = Rotation.from_matrix(R)
        # 获取轴角表示
        return rot.as_rotvec()
    except ValueError:
        # 处理可能出现的无效旋转矩阵
        # 确保R是有效的旋转矩阵
        U, _, Vt = linalg.svd(R)
        R_valid = U @ Vt
        rot = Rotation.from_matrix(R_valid)
        return rot.as_rotvec()

#! 计算误差函数
def compute_loss(fwd_kinematic, laser_matrix, weights=None):
    # 如果未提供权重，则使用默认权重
    if weights is None:
        weights = ERROR_WEIGHTS
    
    # 计算位置误差
    position1 = fwd_kinematic[0:3, 3]
    position2 = laser_matrix[0:3, 3]
    position_error = (position2 - position1) * weights[0:3]

    # 计算旋转误差
    rotation1 = fwd_kinematic[0:3, 0:3]
    rotation2 = laser_matrix[0:3, 0:3]
    
    # 计算相对旋转
    rotation_diff = rotation2 @ rotation1.T
    # 转换为轴角表示
    rotvec = rotation_matrix_to_axis_angle(rotation_diff)
    
    # 将弧度转换为角度，然后应用权重
    rotation_error_deg = rotvec * (180.0 / np.pi) * weights[3:6]

    # 组合位置和旋转误差
    error_vector = np.concatenate((position_error, rotation_error_deg))

    # 计算误差的2-范数
    error_norm = linalg.norm(error_vector)
    return error_norm, position_error, rotation_error_deg, error_vector

if __name__ == "__main__":
    fwd_tool_matrix = get_fk_tool_matrix()
    laser_tool_matrix = get_laser_tool_matrix()
    print(fwd_tool_matrix)
    print("-"*100)  
    print(laser_tool_matrix)

