import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import ast

#! 数据文件路径配置
JOINT_ANGLE_FILE = 'data/extracted_joint_angles.csv'
LASER_POS_FILE = 'data/extracted_laser_positions.csv'
CALIBRATION_RESULTS_FILE = 'results/calibration_results.csv'

#! 机器人配置常量
# 误差权重
ERROR_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

# 待优化的DH参数: alpha, a, d, theta_offset 单位:mm,度
INIT_DH_PARAMS = [
    0, 0, 285.5, 0,
    -90, 0, 0, -90,
    180, 760, 0, -90,
    -90, 0, 540, 0,
    90, 0, 150, 0,
    -90, 0, 127, 0
]

# 关节限位(度)
JOINT_LIMITS = np.array([
    [-175, 175],
    [-110, 110],
    [-135, 135],
    [-175, 175],
    [-115, 115],
    [-175, 175]
])

def get_laser_tool_matrix():
    """
    把激光跟踪仪测量的位姿转换为 4 * 4 变换矩阵
    
    返回:
        numpy.ndarray: 形状为 (num_samples, 4, 4) 的变换矩阵数组
    """
    laser_data = pd.read_csv(LASER_POS_FILE, delimiter=',', skiprows=1, header=None).values
    num_samples = laser_data.shape[0]
    laser_tool_matrix = np.zeros((num_samples, 4, 4))
    
    for i, data in enumerate(laser_data):
        x, y, z, rx, ry, rz = data
        
        #* 计算旋转矩阵（xyz内旋）
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        
        #* 创建变换矩阵
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        
        laser_tool_matrix[i] = T
    return laser_tool_matrix

def load_joint_angles():
    """
    加载关节角度数据
    
    返回:
        numpy.ndarray: 形状为 (num_samples, 6) 的关节角度数组
    """
    return np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)

def load_calibration_data():
    """
    加载所有校准需要的数据
    
    返回:
        tuple: (joint_angles, laser_matrices)
            - joint_angles: 关节角度数据 (num_samples, 6)
            - laser_matrices: 激光跟踪仪变换矩阵 (num_samples, 4, 4)
    """
    joint_angles = load_joint_angles()
    laser_matrices = get_laser_tool_matrix()
    
    print(f"已加载 {joint_angles.shape[0]} 组关节角度数据")
    print(f"已加载 {laser_matrices.shape[0]} 组激光跟踪仪位姿数据")
    
    # 检查数据数量是否一致
    if joint_angles.shape[0] != laser_matrices.shape[0]:
        print(f"警告: 关节角度数据组数 ({joint_angles.shape[0]}) 与激光位姿数据组数 ({laser_matrices.shape[0]}) 不一致")
    
    return joint_angles, laser_matrices

def check_data_files():
    """
    检查数据文件是否存在
    
    返回:
        bool: 所有数据文件都存在返回 True，否则返回 False
    """
    files_to_check = [JOINT_ANGLE_FILE, LASER_POS_FILE, CALIBRATION_RESULTS_FILE]
    all_exist = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"错误: 数据文件不存在: {file_path}")
            all_exist = False
        else:
            print(f"数据文件检查通过: {file_path}")
    
    return all_exist

def load_calibration_params():
    """
    从校准结果文件中读取基座和工具偏移参数
    返回: (tool_offset_params, laser_base_params)
    """
    if not os.path.exists(CALIBRATION_RESULTS_FILE):
        print(f"警告: 校准结果文件 {CALIBRATION_RESULTS_FILE} 不存在，使用默认值")
        # 返回默认值
        tool_default = np.array([2.601768, -0.516418, 96.299777, 0.707103, -0.000219, -0.002932, 0.707105])
        base_default = np.array([2480.125231, 2904.172735, 34.503993, 0.001107, 0.000795, -0.591930, 0.805988])
        return tool_default, base_default
    
    try:
        with open(CALIBRATION_RESULTS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tool_offset_params = None
        laser_base_params = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('base:'):
                # 提取base参数
                param_str = line.split(':', 1)[1].strip()
                laser_base_params = np.array(ast.literal_eval(param_str))
            elif line.startswith('tool:'):
                # 提取tool参数  
                param_str = line.split(':', 1)[1].strip()
                tool_offset_params = np.array(ast.literal_eval(param_str))
        
        if tool_offset_params is None or laser_base_params is None:
            raise ValueError("未能从文件中读取完整的校准参数")
            
        print(f"成功从 {CALIBRATION_RESULTS_FILE} 读取校准参数")
        return tool_offset_params, laser_base_params
        
    except Exception as e:
        print(f"读取校准结果文件时出错: {e}")
        print("使用默认值")
        # 返回默认值
        tool_default = np.array([2.601768, -0.516418, 96.299777, 0.707103, -0.000219, -0.002932, 0.707105])
        base_default = np.array([2480.125231, 2904.172735, 34.503993, 0.001107, 0.000795, -0.591930, 0.805988])
        return tool_default, base_default

def load_all_calibration_data():
    """
    加载所有校准相关的数据和参数
    
    返回:
        tuple: (joint_angles, laser_matrices, tool_offset_params, laser_base_params)
            - joint_angles: 关节角度数据 (num_samples, 6)
            - laser_matrices: 激光跟踪仪变换矩阵 (num_samples, 4, 4)
            - tool_offset_params: 工具偏移参数 (7,) [x,y,z,qx,qy,qz,qw]
            - laser_base_params: 激光基座参数 (7,) [x,y,z,qx,qy,qz,qw]
    """
    # 加载数据
    joint_angles, laser_matrices = load_calibration_data()
    
    # 加载校准参数
    tool_offset_params, laser_base_params = load_calibration_params()
    
    print("所有校准数据和参数加载完成")
    
    return joint_angles, laser_matrices, tool_offset_params, laser_base_params

def get_robot_config():
    """
    获取机器人配置参数
    
    返回:
        dict: 包含所有机器人配置的字典
    """
    return {
        'error_weights': ERROR_WEIGHTS,
        'init_dh_params': INIT_DH_PARAMS,
        'joint_limits': JOINT_LIMITS
    }

def get_initial_params():
    """
    获取优化算法的初始参数
    
    返回:
        numpy.ndarray: 完整的初始参数数组 [DH参数(24) + TCP参数(7) + 基座参数(7)]
    """
    tool_offset_params, laser_base_params = load_calibration_params()
    initial_params = np.concatenate((
        INIT_DH_PARAMS,
        tool_offset_params,
        laser_base_params
    ))
    return initial_params

#! 优化配置常量
# 固定的参数索引(为空则是全优化)
ALL_FIXED_INDICES = []

def get_optimization_config():
    """
    获取优化配置参数
    
    返回:
        dict: 包含优化配置的字典
    """
    return {
        'all_fixed_indices': ALL_FIXED_INDICES,
        'total_params': 38,  # DH(24) + TCP(7) + 基座(7)
    }

def get_parameter_groups(exclude_fixed=True):
    """
    获取参数分组索引
    
    参数:
        exclude_fixed (bool): 是否排除固定参数
    
    返回:
        tuple: (group1_indices, group2_indices)
            - group1_indices: 第一组参数索引 (DH + TCP + 激光XYZ)
            - group2_indices: 第二组参数索引 (激光四元数)
    """
    # 第一组：DH参数 + 工具TCP + 激光跟踪仪XYZ
    all_indices_group1 = list(range(0, 34))
    # 第二组：激光跟踪仪四元数
    all_indices_group2 = list(range(34, 38))
    
    if exclude_fixed:
        # 排除固定参数
        group1_indices = [idx for idx in all_indices_group1 if idx not in ALL_FIXED_INDICES]
        group2_indices = [idx for idx in all_indices_group2 if idx not in ALL_FIXED_INDICES]
        return group1_indices, group2_indices
    else:
        return all_indices_group1, all_indices_group2

def get_optimizable_indices():
    """
    获取所有可优化的参数索引
    
    返回:
        list: 可优化参数的索引列表
    """
    return [i for i in range(38) if i not in ALL_FIXED_INDICES]
