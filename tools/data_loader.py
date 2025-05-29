import os
import re
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
import ast

# 默认配置文件路径
DEFAULT_CONFIG_FILE = 'config/config.yaml'

# 全局配置变量
_config = None

def load_config(config_file=None):
    """加载配置文件"""
    global _config
    
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(config_file):
        print(f"警告: 配置文件 {config_file} 不存在，使用默认配置")
        _config = get_default_config()
        return _config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_file}")
        return _config
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        print("使用默认配置")
        _config = get_default_config()
        return _config

def get_default_config():
    """获取默认配置"""
    return {
        'data_files': {
            'joint_angle_raw': 'data/joint_angle.csv',
            'laser_pos_raw': 'data/laser_pos.csv', 
            'dh_params': 'data/dh.csv',
            'joint_limits': 'data/joint_limits.csv',
            'calibration_results': 'data/hand_eye_calibration.csv'
        },
        'robot_config': {
            'error_weights': [1.0, 1.0, 1.0, 0.01, 0.01, 0.01],
            'fixed_indices': []
        },
        'optimization': {
            'max_iterations': 1000,
            'convergence_threshold': 1e-8,
            'verbose': True
        }
    }

def get_config():
    """获取当前配置"""
    global _config
    if _config is None:
        load_config()
    return _config

def get_file_path(key):
    """获取文件路径"""
    config = get_config()
    return config['data_files'].get(key, '')

def get_error_weights():
    """获取误差权重"""
    config = get_config()
    return np.array(config['robot_config']['error_weights'])

def get_fixed_indices():
    """获取固定参数索引"""
    config = get_config()
    return config['robot_config']['fixed_indices']

def extract_joint_angles_from_raw(file_path=None):
    """直接从原始关节角度文件提取数据"""
    config = get_config()
    
    if file_path is None:
        file_path = config['data_files']['joint_angle_raw']
    
    # 使用固定的提取模式
    pattern = re.compile(r'j:\s*{\s*([^}]*?)\s*}')
    
    joint_angles = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    # 获取大括号内的内容
                    content_in_braces = match.group(1)
                    
                    # 按逗号分割，并去除空白
                    angles_str = [angle.strip() for angle in content_in_braces.split(',')]
                    
                    # 确保有足够的角度值
                    if len(angles_str) >= 6:
                        # 提取前6个角度值并转换为float
                        try:
                            extracted_angles = [float(angle) for angle in angles_str[:6] if angle]
                            if len(extracted_angles) == 6:
                                joint_angles.append(extracted_angles)
                            else:
                                print(f"警告: 第{line_num}行未能提取6个角度值")
                        except ValueError as e:
                            print(f"警告: 第{line_num}行数值转换错误: {e}")
                    else:
                        print(f"警告: 第{line_num}行数据不足6个角度值")
        
        if not joint_angles:
            raise ValueError("未能从文件中提取到任何关节角度数据")
            
        result = np.array(joint_angles)
        print(f"成功从 {file_path} 提取 {result.shape[0]} 组关节角度数据")
        return result
        
    except FileNotFoundError:
        print(f"错误: 关节角度文件未找到: {file_path}")
        raise
    except Exception as e:
        print(f"提取关节角度时发生错误: {e}")
        raise

def extract_laser_positions_from_raw(file_path=None):
    """直接从原始激光位置文件提取数据"""
    config = get_config()
    
    if file_path is None:
        file_path = config['data_files']['laser_pos_raw']
    
    # 使用固定的提取配置
    measurement_columns = [7, 8, 9, 10, 11, 12]  # 测量X,Y,Z,Rx,Ry,Rz列
    
    laser_positions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            next(f)
            
            for line_num, line in enumerate(f, start=2):
                stripped_line = line.strip()
                if not stripped_line:  # 跳过空行
                    continue
                
                # 分割行数据（使用空格分隔）
                parts = stripped_line.split()
                parts = [part.strip() for part in parts if part.strip()]  # 去除空白部分
                
                # 检查是否有足够的列
                if len(parts) > max(measurement_columns):
                    try:
                        # 提取测量数据列
                        measured_data = [float(parts[i]) for i in measurement_columns]
                        laser_positions.append(measured_data)
                    except (ValueError, IndexError) as e:
                        print(f"警告: 第{line_num}行数据解析错误: {e}")
                else:
                    print(f"警告: 第{line_num}行数据列数不足，无法提取")
        
        if not laser_positions:
            raise ValueError("未能从文件中提取到任何激光位置数据")
            
        result = np.array(laser_positions)
        print(f"成功从 {file_path} 提取 {result.shape[0]} 组激光位置数据")
        return result
        
    except FileNotFoundError:
        print(f"错误: 激光位置文件未找到: {file_path}")
        raise
    except Exception as e:
        print(f"提取激光位置时发生错误: {e}")
        raise

def get_laser_tool_matrix():
    """把原始激光数据转换为4*4的变换矩阵"""
    laser_data = extract_laser_positions_from_raw()
    num_samples = laser_data.shape[0]
    laser_tool_matrix = np.zeros((num_samples, 4, 4))
    
    for i, data in enumerate(laser_data):
        x, y, z, rx, ry, rz = data
        
        # 计算旋转矩阵（xyz内旋）
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        
        # 创建变换矩阵
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        
        laser_tool_matrix[i] = T
    return laser_tool_matrix

def load_dh_params():
    """加载DH参数从CSV文件"""
    config = get_config()
    file_path = config['data_files']['dh_params']
    
    dh_data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None).values
    
    if dh_data.shape[0] != 6 or dh_data.shape[1] != 4:
        raise ValueError(f"DH参数文件格式错误：期望6行4列，实际{dh_data.shape[0]}行{dh_data.shape[1]}列")
    
    # 将6x4的矩阵展平为24个参数的一维数组
    dh_params_flat = dh_data.flatten()
    
    print(f"成功从 {file_path} 加载DH参数")
    print(f"加载的DH参数: {dh_params_flat.tolist()}")
    return dh_params_flat
        
def load_joint_limits():
    """加载关节限位从CSV文件"""
    config = get_config()
    file_path = config['data_files']['joint_limits']
    
    limits_data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None).values
    
    if limits_data.shape[0] != 6 or limits_data.shape[1] != 2:
        raise ValueError(f"关节限位文件格式错误：期望6行2列，实际{limits_data.shape[0]}行{limits_data.shape[1]}列")
    
    # CSV格式是 [UpperLimit, LowerLimit]，需要转换为 [LowerLimit, UpperLimit]
    joint_limits = np.zeros((6, 2))
    joint_limits[:, 0] = limits_data[:, 1]  # LowerLimit (最小值)
    joint_limits[:, 1] = limits_data[:, 0]  # UpperLimit (最大值)
    
    print(f"成功从 {file_path} 加载关节限位")
    print(f"加载的关节限位:")
    for i, (min_val, max_val) in enumerate(joint_limits):
        print(f"  关节{i+1}: [{min_val:.1f}°, {max_val:.1f}°]")
    
    return joint_limits

def load_joint_angles():
    """加载关节角度数据 - 直接从原始文件提取"""
    return extract_joint_angles_from_raw()

def load_calibration_params():
    """加载初始基座和工具偏移参数"""
    config = get_config()
    file_path = config['data_files']['calibration_results']
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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
            
        print(f"成功从 {file_path} 读取校准参数")
        return tool_offset_params, laser_base_params
        
    except Exception as e:
        print(f"读取校准结果文件时出错: {e}")
        raise

def get_initial_params():
    """获取初始参数(38个参数)"""
    tool_offset_params, laser_base_params = load_calibration_params()
    dh_params = load_dh_params()  
    initial_params = np.concatenate((
        dh_params,  
        tool_offset_params,
        laser_base_params
    ))
    return initial_params

def get_parameter_groups(exclude_fixed=True):
    """获取参数组"""
    all_indices_group1 = list(range(0, 34))
    all_indices_group2 = list(range(34, 38))
    
    fixed_indices = get_fixed_indices()
    
    if exclude_fixed:
        # 排除固定参数
        group1_indices = [idx for idx in all_indices_group1 if idx not in fixed_indices]
        group2_indices = [idx for idx in all_indices_group2 if idx not in fixed_indices]
        return group1_indices, group2_indices
    else:
        return all_indices_group1, all_indices_group2

def get_optimizable_indices():
    """获取可优化参数索引"""
    fixed_indices = get_fixed_indices()
    return [i for i in range(38) if i not in fixed_indices]

# 初始化配置
load_config()

