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

# 全局缓存变量
_joint_limits_cache = None

def load_config(config_file=None):
    """加载配置文件"""
    global _config
    
    # 如果已经加载过配置，直接返回
    if _config is not None:
        return _config
        
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
        print(f"✅ 成功加载配置文件: {config_file}")
        return _config
    except Exception as e:
        print(f"❌ 加载配置文件时出错: {e}")
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
            'lm_optimization': {
                'alternate_optimization': {
                    'max_alt_iterations': 4,
                    'convergence_tol': 1e-4,
                    'max_sub_iterations_group1': 10,
                    'max_sub_iterations_group2': 10
                },
                'damping': {
                    'lambda_init_group1': 2.0,
                    'lambda_init_group2': 0.001,
                    'lambda_init_default': 0.01,
                    'lambda_max': 1e8,
                    'lambda_min': 1e-7,
                    'damping_type': 'marquardt'
                },
                'convergence': {
                    'parameter_tol': 1e-10,
                    'error_tol': 1e-12,
                    'max_inner_iterations': 10,
                    'rho_threshold': 0.0
                },
                'constraints': {
                    'max_theta_change_degrees': 1.0,
                    'enable_quaternion_normalization': True
                },
                'output': {
                    'save_delta_values': True,
                    'delta_csv_file': 'results/delta_values.csv',
                    'save_optimization_results': True,
                    'results_prefix': 'results/optimized'
                }
            }
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
    global ERROR_WEIGHTS
    if ERROR_WEIGHTS is None:
        _ensure_config_loaded()
    return ERROR_WEIGHTS

def get_fixed_indices():
    """获取固定参数索引"""
    global ALL_FIXED_INDICES
    if ALL_FIXED_INDICES is None:
        _ensure_config_loaded()
    return ALL_FIXED_INDICES

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
        print(f"✅ 加载关节角度: {result.shape[0]}组数据")
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
        print(f"✅ 加载激光数据: {result.shape[0]}组数据")
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
    
    print(f"✅ 加载DH参数: {dh_params_flat.shape[0]}个参数")
    return dh_params_flat
        
def load_joint_limits():
    """加载关节限位从CSV文件（带缓存）"""
    global _joint_limits_cache
    
    # 如果已经缓存，直接返回
    if _joint_limits_cache is not None:
        return _joint_limits_cache
        
    config = get_config()
    file_path = config['data_files']['joint_limits']
    
    limits_data = pd.read_csv(file_path, delimiter=',', skiprows=1, header=None).values
    
    if limits_data.shape[0] != 6 or limits_data.shape[1] != 2:
        raise ValueError(f"关节限位文件格式错误：期望6行2列，实际{limits_data.shape[0]}行{limits_data.shape[1]}列")
    
    # CSV格式是 [UpperLimit, LowerLimit]，需要转换为 [LowerLimit, UpperLimit]
    joint_limits = np.zeros((6, 2))
    joint_limits[:, 0] = limits_data[:, 1]  # LowerLimit (最小值)
    joint_limits[:, 1] = limits_data[:, 0]  # UpperLimit (最大值)
    
    # 缓存结果
    _joint_limits_cache = joint_limits
    print(f"✅ 加载关节限位: {joint_limits.shape[0]}个关节")
    
    return joint_limits

def load_joint_angles():
    """加载关节角度数据 - 直接从原始文件提取"""
    return extract_joint_angles_from_raw()

def load_calibration_params():
    """加载初始基座和工具偏移参数
    
    返回:
        tool_offset_params: TCP偏移参数 (7个值：x,y,z,qx,qy,qz,qw) - 数值较小
        laser_base_params: 激光基座参数 (7个值：x,y,z,qx,qy,qz,qw) - 数值较大
    """
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
                # ⚠️ base参数 = 激光基座变换 (Laser -> Base) - 数值较大，通常数百到数千
                param_str = line.split(':', 1)[1].strip()
                laser_base_params = np.array(ast.literal_eval(param_str))
            elif line.startswith('tool:'):
                # ⚠️ tool参数 = TCP偏移 (Flange -> Tool) - 数值较小，通常几毫米到几十毫米  
                param_str = line.split(':', 1)[1].strip()
                tool_offset_params = np.array(ast.literal_eval(param_str))
        
        if tool_offset_params is None or laser_base_params is None:
            raise ValueError("未能从文件中读取完整的校准参数")
        
        # 数值范围检查，帮助发现参数标签错误
        tool_pos_magnitude = np.linalg.norm(tool_offset_params[:3])
        base_pos_magnitude = np.linalg.norm(laser_base_params[:3]) 
        
        if tool_pos_magnitude > 1000:  # TCP偏移超过1米，可能有问题
            print(f"⚠️ 警告: TCP偏移参数数值异常大 ({tool_pos_magnitude:.1f}), 请检查是否与base参数标签搞反")
        if base_pos_magnitude < 100:   # 激光基座距离小于10cm，可能有问题  
            print(f"⚠️ 警告: 激光基座参数数值异常小 ({base_pos_magnitude:.1f}), 请检查是否与tool参数标签搞反")
            
        print(f"✅ 加载校准参数: TCP + 激光基座变换")
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

#! LM优化配置读取函数

def get_lm_config():
    """获取LM优化完整配置"""
    config = get_config()
    return config.get('optimization', {}).get('lm_optimization', {})

def get_alternate_optimization_config():
    """获取交替优化配置"""
    lm_config = get_lm_config()
    config = lm_config.get('alternate_optimization', {
        'max_alt_iterations': 4,
        'convergence_tol': 1e-4,
        'max_sub_iterations_group1': 10,
        'max_sub_iterations_group2': 10
    })
    
    # 确保数值参数是正确的类型
    if 'convergence_tol' in config:
        config['convergence_tol'] = float(config['convergence_tol'])
    if 'max_alt_iterations' in config:
        config['max_alt_iterations'] = int(config['max_alt_iterations'])
    if 'max_sub_iterations_group1' in config:
        config['max_sub_iterations_group1'] = int(config['max_sub_iterations_group1'])
    if 'max_sub_iterations_group2' in config:
        config['max_sub_iterations_group2'] = int(config['max_sub_iterations_group2'])
    
    return config

def get_damping_config():
    """获取阻尼参数配置"""
    lm_config = get_lm_config()
    config = lm_config.get('damping', {
        'lambda_init_group1': 2.0,
        'lambda_init_group2': 0.001,
        'lambda_init_default': 0.01,
        'lambda_max': 1e8,
        'lambda_min': 1e-7,
        'damping_type': 'marquardt'
    })
    
    # 确保数值参数是浮点数类型
    for key in ['lambda_init_group1', 'lambda_init_group2', 'lambda_init_default', 
                'lambda_max', 'lambda_min']:
        if key in config:
            config[key] = float(config[key])
    
    return config

def get_convergence_config():
    """获取收敛控制配置"""
    lm_config = get_lm_config()
    config = lm_config.get('convergence', {
        'parameter_tol': 1e-10,
        'error_tol': 1e-12,
        'max_inner_iterations': 10,
        'rho_threshold': 0.0
    })
    
    # 确保数值参数是正确类型
    for key in ['parameter_tol', 'error_tol', 'rho_threshold']:
        if key in config:
            config[key] = float(config[key])
    if 'max_inner_iterations' in config:
        config['max_inner_iterations'] = int(config['max_inner_iterations'])
    
    return config

def get_constraints_config():
    """获取参数约束配置"""
    lm_config = get_lm_config()
    config = lm_config.get('constraints', {
        'max_theta_change_degrees': 1.0,
        'enable_quaternion_normalization': True
    })
    
    # 确保数值参数是浮点数类型
    if 'max_theta_change_degrees' in config:
        config['max_theta_change_degrees'] = float(config['max_theta_change_degrees'])
    if 'enable_quaternion_normalization' in config:
        config['enable_quaternion_normalization'] = bool(config['enable_quaternion_normalization'])
    
    return config

def get_output_config():
    """获取输出设置配置"""
    lm_config = get_lm_config()
    return lm_config.get('output', {
        'save_delta_values': True,
        'delta_csv_file': 'results/delta_values.csv',
        'save_optimization_results': True,
        'results_prefix': 'results/optimized'
    })

def get_max_theta_change_radians():
    """获取theta参数最大变化量（弧度）"""
    constraints_config = get_constraints_config()
    max_theta_degrees = constraints_config.get('max_theta_change_degrees', 1.0)
    return np.deg2rad(max_theta_degrees)

# 延迟加载的全局变量，避免导入时重复输出
def _get_error_weights():
    """内部函数：获取误差权重"""
    config = get_config()
    return np.array(config['robot_config']['error_weights'])

def _get_fixed_indices():
    """内部函数：获取固定参数索引"""
    config = get_config()
    return config['robot_config']['fixed_indices']

# 导出变量（向后兼容），在首次访问时才加载
ERROR_WEIGHTS = None
ALL_FIXED_INDICES = None

def _ensure_config_loaded():
    """确保配置已加载"""
    global ERROR_WEIGHTS, ALL_FIXED_INDICES
    if ERROR_WEIGHTS is None:
        ERROR_WEIGHTS = _get_error_weights()
        ALL_FIXED_INDICES = _get_fixed_indices()

