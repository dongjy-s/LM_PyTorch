import os
import re
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
import ast

# 添加Excel支持和编码检测
try:
    import openpyxl  # Excel支持
except ImportError:
    openpyxl = None
    print("警告: 未安装openpyxl，Excel格式支持不可用")

try:
    import chardet  # 编码检测
except ImportError:
    chardet = None
    print("警告: 未安装chardet，编码检测不可用")

# 默认配置文件路径
DEFAULT_CONFIG_FILE = 'config/config.yaml'

# 全局配置变量
_config = None

# 全局缓存变量
_joint_limits_cache = None

class SmartFormatDetector:
    """智能格式检测器"""
    
    def __init__(self):
        self.rokae_pattern = re.compile(r'LOCAL VAR jointtarget.*?j:\s*{\s*([^}]*?)\s*}')
        
    def detect_format(self, file_path):
        """检测文件格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return 'txt'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext == '.csv':
            if self._is_rokae_format(file_path):
                return 'rokae_csv'
            else:
                return 'csv'
        else:
            return 'csv'  # 默认尝试CSV格式
    
    def _is_rokae_format(self, file_path):
        """检测关节角度是否为rokae机器人格式"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                first_lines = f.read(500)
                return 'LOCAL VAR jointtarget' in first_lines
        except:
            return False
    
    def detect_encoding(self, file_path):
        """检测文件编码"""
        if chardet is None:
            return 'utf-8'
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except:
            return 'utf-8'
    
    def detect_delimiter(self, file_path):
        """检测分隔符"""
        encoding = self.detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
            
            delimiters = [',', '\t', ' ', ';', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count
            
            if not delimiter_counts:
                return ','
            
            return max(delimiter_counts, key=delimiter_counts.get)
        except:
            return ','

class SmartDataParser:
    """智能数据解析器"""
    
    def __init__(self):
        self.detector = SmartFormatDetector()
        self.config = get_config()
    
    def parse_joint_angles(self, file_path):
        """解析关节角度数据"""
        file_format = self.detector.detect_format(file_path)
        
        try:
            if file_format == 'rokae_csv':
                return self._parse_rokae_joint_angles(file_path)
            elif file_format == 'csv':
                return self._parse_csv_joint_angles(file_path)
            elif file_format == 'txt':
                return self._parse_txt_joint_angles(file_path)  
            elif file_format == 'excel':
                return self._parse_excel_joint_angles(file_path)
            else:
                # 回退到原始格式解析
                return self._parse_rokae_joint_angles(file_path)
        except Exception as e:
            print(f"智能解析失败，尝试原始格式: {e}")
            return self._parse_rokae_joint_angles(file_path)
    
    def parse_positions(self, file_path):
        """解析位置数据"""
        file_format = self.detector.detect_format(file_path)
        
        try:
            if file_format == 'csv':
                return self._parse_csv_positions(file_path)
            elif file_format == 'txt':
                return self._parse_txt_positions(file_path)
            elif file_format == 'excel':
                return self._parse_excel_positions(file_path)
            else:
                # 回退到原始激光跟踪仪格式
                return self._parse_laser_tracker_positions(file_path)
        except Exception as e:
            print(f"智能解析失败，尝试原始激光跟踪仪格式: {e}")
            return self._parse_laser_tracker_positions(file_path)
    
    def _parse_rokae_joint_angles(self, file_path):
        """解析rokae格式关节角度（保持原有逻辑）"""
        config = self.config.get('data_input', {}).get('joint_angles', {}).get('rokae_format', {})
        pattern_str = config.get('pattern', r'j:\s*{\s*([^}]*?)\s*}')
        pattern = re.compile(pattern_str)
        
        joint_angles = []
        encoding = self.detector.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                match = pattern.search(line)
                if match:
                    content_in_braces = match.group(1)
                    angles_str = [angle.strip() for angle in content_in_braces.split(',')]
                    
                    if len(angles_str) >= 6:
                        try:
                            extracted_angles = [float(angle) for angle in angles_str[:6] if angle]
                            if len(extracted_angles) == 6:
                                joint_angles.append(extracted_angles)
                        except ValueError as e:
                            print(f"警告: 第{line_num}行数值转换错误: {e}")
        
        if not joint_angles:
            raise ValueError("未能提取关节角度数据")
        
        result = np.array(joint_angles)
        print(f"✅ rokae格式加载关节角度: {result.shape[0]}组数据")
        return result
    
    def _parse_csv_joint_angles(self, file_path):
        """解析CSV格式关节角度"""
        encoding = self.detector.detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        
        # 智能字段识别
        joint_data = self._extract_joint_angles_from_df(df)
        print(f"✅ CSV格式加载关节角度: {joint_data.shape[0]}组数据")
        return joint_data
    
    def _parse_txt_joint_angles(self, file_path):
        """解析TXT格式关节角度"""
        encoding = self.detector.detect_encoding(file_path)
        delimiter = self.detector.detect_delimiter(file_path)
        
        # 先检查第一行是否包含列名（如J1,J2等）
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline().strip()
        
        # 如果第一行包含字母（可能是列名），使用header=0，否则使用header=None
        if any(c.isalpha() for c in first_line):
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=0)
        else:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, 
                            header=None, comment='#')
        
        joint_data = self._extract_joint_angles_from_df(df)
        print(f"✅ TXT格式加载关节角度: {joint_data.shape[0]}组数据")
        return joint_data
    
    def _parse_excel_joint_angles(self, file_path):
        """解析Excel格式关节角度"""
        if openpyxl is None:
            raise ImportError("需要安装openpyxl来支持Excel格式")
        
        df = pd.read_excel(file_path, sheet_name=0)
        joint_data = self._extract_joint_angles_from_df(df)
        print(f"✅ Excel格式加载关节角度: {joint_data.shape[0]}组数据")
        return joint_data
    
    def _extract_joint_angles_from_df(self, df):
        """从DataFrame智能提取关节角度数据"""
        config = self.config.get('data_input', {}).get('joint_angles', {})
        field_patterns = config.get('field_patterns', [
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            ["J1", "J2", "J3", "J4", "J5", "J6"],
        ])
        
        # 尝试通过列名匹配
        joint_columns = []
        for pattern in field_patterns:
            matching_cols = []
            for col_name in pattern:
                for df_col in df.columns:
                    if col_name.lower() in str(df_col).lower():
                        matching_cols.append(df_col)
                        break
            if len(matching_cols) == 6:
                joint_columns = matching_cols
                break
        
        # 如果找不到匹配的列名，使用前6列
        if not joint_columns:
            if df.shape[1] >= 6:
                joint_columns = df.columns[:6].tolist()
            else:
                raise ValueError(f"数据列数不足，需要至少6列，实际{df.shape[1]}列")
        
        # 提取数据并转换为numpy数组
        joint_data = df[joint_columns].values.astype(float)
        
        if joint_data.shape[1] != 6:
            raise ValueError(f"关节角度数据列数错误，期望6列，实际{joint_data.shape[1]}列")
        
        return joint_data
    
    def _parse_laser_tracker_positions(self, file_path):
        """解析激光跟踪仪格式位置数据（保持原有逻辑）"""
        config = self.config.get('data_input', {}).get('positions', {}).get('laser_tracker', {})
        measurement_columns = config.get('measurement_columns', [7, 8, 9, 10, 11, 12])
        
        laser_positions = []
        encoding = self.detector.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, start=2):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                parts = stripped_line.split()
                parts = [part.strip() for part in parts if part.strip()]
                
                if len(parts) > max(measurement_columns):
                    try:
                        measured_data = [float(parts[i]) for i in measurement_columns]
                        laser_positions.append(measured_data)
                    except (ValueError, IndexError) as e:
                        print(f"警告: 第{line_num}行数据解析错误: {e}")
        
        if not laser_positions:
            raise ValueError("未能提取激光位置数据")
        
        result = np.array(laser_positions)
        print(f"✅ 激光跟踪仪格式加载位置数据: {result.shape[0]}组数据")
        return result
    
    def _parse_csv_positions(self, file_path):
        """解析CSV格式位置数据"""
        encoding = self.detector.detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        
        position_data = self._extract_positions_from_df(df)
        print(f"✅ CSV格式加载位置数据: {position_data.shape[0]}组数据")
        return position_data
    
    def _parse_txt_positions(self, file_path):
        """解析TXT格式位置数据"""
        encoding = self.detector.detect_encoding(file_path)
        delimiter = self.detector.detect_delimiter(file_path)
        
        # 先检查第一行是否包含列名（如X,Y,Z等）
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline().strip()
        
        # 如果第一行包含字母（可能是列名），使用header=0，否则使用header=None
        if any(c.isalpha() for c in first_line):
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=0)
        else:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, 
                            header=None, comment='#')
        
        position_data = self._extract_positions_from_df(df)
        print(f"✅ TXT格式加载位置数据: {position_data.shape[0]}组数据")
        return position_data
    
    def _parse_excel_positions(self, file_path):
        """解析Excel格式位置数据"""
        if openpyxl is None:
            raise ImportError("需要安装openpyxl来支持Excel格式")
        
        df = pd.read_excel(file_path, sheet_name=0)
        position_data = self._extract_positions_from_df(df)
        print(f"✅ Excel格式加载位置数据: {position_data.shape[0]}组数据")
        return position_data
    
    def _extract_positions_from_df(self, df):
        """从DataFrame智能提取位置数据"""
        config = self.config.get('data_input', {}).get('positions', {})
        pos_patterns = config.get('position_patterns', ["X", "Y", "Z"])
        ori_patterns = config.get('orientation_patterns', ["Rx", "Ry", "Rz"])
        
        # 查找位置列
        pos_columns = []
        for pattern in pos_patterns:
            for df_col in df.columns:
                if pattern.lower() in str(df_col).lower():
                    pos_columns.append(df_col)
                    break
            if len(pos_columns) == 3:
                break
        
        # 查找姿态列
        ori_columns = []
        for pattern in ori_patterns:
            for df_col in df.columns:
                if pattern.lower() in str(df_col).lower():
                    ori_columns.append(df_col)
                    break
            if len(ori_columns) == 3:
                break
        
        # 如果找不到匹配列，使用默认列
        if len(pos_columns) != 3 or len(ori_columns) != 3:
            if df.shape[1] >= 6:
                all_columns = df.columns[:6].tolist()
                pos_columns = all_columns[:3]
                ori_columns = all_columns[3:6]
            else:
                raise ValueError(f"位置数据列数不足，需要至少6列，实际{df.shape[1]}列")
        
        # 提取数据
        position_data = df[pos_columns + ori_columns].values.astype(float)
        
        if position_data.shape[1] != 6:
            raise ValueError(f"位置数据列数错误，期望6列，实际{position_data.shape[1]}列")
        
        return position_data

# 全局解析器实例
_smart_parser = None

def get_smart_parser():
    """获取智能解析器实例"""
    global _smart_parser
    if _smart_parser is None:
        _smart_parser = SmartDataParser()
    return _smart_parser

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
    config = get_config()
    return np.array(config['robot_config']['error_weights'])

def get_fixed_indices():
    """获取固定参数索引"""
    config = get_config()
    return config['robot_config']['fixed_indices']

def extract_joint_angles_from_raw(file_path=None):
    """智能提取关节角度数据 - 支持多种格式"""
    config = get_config()
    
    if file_path is None:
        file_path = config['data_files']['joint_angle_raw']
    
    try:
        # 使用智能解析器
        parser = get_smart_parser()
        result = parser.parse_joint_angles(file_path)
        return result
        
    except FileNotFoundError:
        print(f"错误: 关节角度文件未找到: {file_path}")
        raise
    except Exception as e:
        print(f"提取关节角度时发生错误: {e}")
        raise

def extract_laser_positions_from_raw(file_path=None):
    """智能提取激光位置数据 - 支持多种格式"""
    config = get_config()
    
    if file_path is None:
        file_path = config['data_files']['laser_pos_raw']
    
    try:
        # 使用智能解析器
        parser = get_smart_parser()
        result = parser.parse_positions(file_path)
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

def clear_cache():
    """清理所有缓存，确保重新加载"""
    global _config, _joint_limits_cache
    _config = None
    _joint_limits_cache = None
    print("🧹 已清理所有缓存")

