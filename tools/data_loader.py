import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import ast

#! 数据文件路径配置
JOINT_ANGLE_FILE = 'data/extracted_joint_angles.csv'
LASER_POS_FILE = 'data/extracted_laser_positions.csv'
CALIBRATION_RESULTS_FILE = 'results/calibration_results.csv'
DH_PARAMS_FILE = 'data/dh.csv'  # DH参数文件路径
JOINT_LIMITS_FILE = 'data/joint_limits.csv'  # 关节限位文件路径

# 固定的参数索引(为空则是全优化)
ALL_FIXED_INDICES = []
#! 机器人配置常量
# 误差权重
ERROR_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01])



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

#! 加载DH参数从CSV文件
def load_dh_params():
    dh_data = pd.read_csv(DH_PARAMS_FILE, delimiter=',', skiprows=1, header=None).values
    
    if dh_data.shape[0] != 6 or dh_data.shape[1] != 4:
        raise ValueError(f"DH参数文件格式错误：期望6行4列，实际{dh_data.shape[0]}行{dh_data.shape[1]}列")
    
    # 将6x4的矩阵展平为24个参数的一维数组
    dh_params_flat = dh_data.flatten()
    
    print(f"成功从 {DH_PARAMS_FILE} 加载DH参数")
    print(f"加载的DH参数: {dh_params_flat.tolist()}")
    return dh_params_flat
        
#! 加载关节限位从CSV文件
def load_joint_limits():

    limits_data = pd.read_csv(JOINT_LIMITS_FILE, delimiter=',', skiprows=1, header=None).values
    
    if limits_data.shape[0] != 6 or limits_data.shape[1] != 2:
        raise ValueError(f"关节限位文件格式错误：期望6行2列，实际{limits_data.shape[0]}行{limits_data.shape[1]}列")
    
    # CSV格式是 [UpperLimit, LowerLimit]，需要转换为 [LowerLimit, UpperLimit]
    joint_limits = np.zeros((6, 2))
    joint_limits[:, 0] = limits_data[:, 1]  # LowerLimit (最小值)
    joint_limits[:, 1] = limits_data[:, 0]  # UpperLimit (最大值)
    
    print(f"成功从 {JOINT_LIMITS_FILE} 加载关节限位")
    print(f"加载的关节限位:")
    for i, (min_val, max_val) in enumerate(joint_limits):
        print(f"  关节{i+1}: [{min_val:.1f}°, {max_val:.1f}°]")
    
    return joint_limits
        

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
    dh_params = load_dh_params()  
    initial_params = np.concatenate((
        dh_params,  
        tool_offset_params,
        laser_base_params
    ))
    return initial_params


def get_parameter_groups(exclude_fixed=True):
    all_indices_group1 = list(range(0, 34))
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

