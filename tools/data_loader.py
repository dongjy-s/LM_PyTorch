import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import ast

#! 数据文件路径配置
JOINT_ANGLE_FILE = 'data/extracted_joint_angles.csv'
LASER_POS_FILE = 'data/extracted_laser_positions.csv'
CALIBRATION_RESULTS_FILE = 'results/calibration_results.csv'

# 固定的参数索引(为空则是全优化)
ALL_FIXED_INDICES = []
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

#! 把原始激光数据转换为4*4的变换矩阵
def get_laser_tool_matrix():
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

#! 加载关节角度数据
def load_joint_angles():
    return np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)

#! 加载初始基座和工具偏移参数
def load_calibration_params():
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
       

#! 获取初始参数(38个参数)
def get_initial_params():
    tool_offset_params, laser_base_params = load_calibration_params()
    initial_params = np.concatenate((
        INIT_DH_PARAMS,
        tool_offset_params,
        laser_base_params
    ))
    return initial_params


def get_parameter_groups(exclude_fixed=True):
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
    return [i for i in range(38) if i not in ALL_FIXED_INDICES]
