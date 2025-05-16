import sys
import os
import numpy as np

import pandas as pd 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from jacobian_torch import get_laser_tool_matrix
from lm_optimize_pytorch import compute_total_error_avg, ERROR_WEIGHTS

JOINT_ANGLE_FILE_PATH = os.path.join(PROJECT_ROOT, 'data/extracted_joint_angles.csv')

# CSV文件路径定义
OPTIMIZED_DH_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'results/optimized_dh_parameters.csv')
OPTIMIZED_TCP_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'results/optimized_tcp_parameters.csv')
OPTIMIZED_T_LASER_BASE_FILE = os.path.join(PROJECT_ROOT, 'results/optimized_t_laser_base_parameters.csv')

def get_specified_parameters():
    """定义并组合用户在脚本中直接指定的DH、工具和基座参数。"""
    dh_params_matrix = np.array([
        [0.0487, -1.9024, 285.3056, 0.0005],
        [-89.9989, -0.3083, -0.0239, -89.9977],
        [179.9991, 760.8683, 0.0239, -89.998],
        [-90.0082, 0.0045, 539.5412, 0.0025],
        [89.5959, 0.0348, 147.7834, -0.0008],
        [-90.0114, 0.0808, 128.2501, 0.001]
    ], dtype=np.float64)
    dh = dh_params_matrix.flatten()
    tcp = np.array([1.7652, -0.7405, 95.9384, 0.7072, -0.001, -0.0022, 0.707], dtype=np.float64)
    base = np.array([2482.8681, 2904.818, 36.0253, 0.0019, 0.0009, -0.592, 0.806], dtype=np.float64)

    if len(dh) != 24: raise ValueError(f"指定的DH参数数量应为24, 实际为 {len(dh)}。")
    if len(tcp) != 7: raise ValueError(f"指定的TCP参数数量应为7, 实际为 {len(tcp)}。")
    if len(base) != 7: raise ValueError(f"指定的基座变换参数数量应为7, 实际为 {len(base)}。")
    return np.concatenate((dh, tcp, base))

def load_calibrated_params_from_csv():
    """从CSV文件加载标定后的机器人参数。"""
    try:
        df_dh = pd.read_csv(OPTIMIZED_DH_PARAMS_FILE)
        dh_params = df_dh[['alpha', 'a', 'd', 'theta_offset']].values.flatten()
        
        df_tcp = pd.read_csv(OPTIMIZED_TCP_PARAMS_FILE)
        tcp_params = df_tcp['value'].values
        
        df_t_laser_base = pd.read_csv(OPTIMIZED_T_LASER_BASE_FILE)
        t_laser_base_params = df_t_laser_base['value'].values

        if len(dh_params) != 24: raise ValueError(f"从CSV加载的DH参数数量不正确 ({len(dh_params)}/24)。")
        if len(tcp_params) != 7: raise ValueError(f"从CSV加载的TCP参数数量不正确 ({len(tcp_params)}/7)。")
        if len(t_laser_base_params) != 7: raise ValueError(f"从CSV加载的基座参数数量不正确 ({len(t_laser_base_params)}/7)。")
        
        print("成功从CSV文件加载标定后的参数。")
        return np.concatenate((dh_params, tcp_params, t_laser_base_params))
    except FileNotFoundError as e:
        print(f"错误: CSV参数文件未找到: {e.filename}")
        return None
    except Exception as e:
        print(f"加载CSV标定参数时发生错误: {e}")
        return None

def main():
    # 1. 加载测量数据
    try:
        all_joint_angles_np = np.loadtxt(JOINT_ANGLE_FILE_PATH, delimiter=',', skiprows=1)
        all_T_laser_tool_measured_np = get_laser_tool_matrix()
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到: {e.filename} 或其依赖的 'data/extracted_laser_positions.csv'")
        sys.exit(1)
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        sys.exit(1)
    
    print(f"成功加载 {all_joint_angles_np.shape[0]} 组关节角度和 {all_T_laser_tool_measured_np.shape[0]} 组激光测量数据。")
    num_effective_samples = min(all_joint_angles_np.shape[0], all_T_laser_tool_measured_np.shape[0])
    if num_effective_samples == 0:
        print("错误：没有可供测试的数据样本。")
        sys.exit(1)

    weights_np = ERROR_WEIGHTS 
    print(f"\n使用的误差权重: 位置={weights_np[:3].tolist()}, 姿态={weights_np[3:].tolist()}")

    rmse_results = {}
  
    try:
        specified_params = get_specified_parameters()
        print(f"\n使用【激光跟踪仪LM标定后参数】进行RMSE计算 (共 {len(specified_params)} 个参数)。")
        rmse_specified = compute_total_error_avg(
            specified_params,
            all_joint_angles_np[:num_effective_samples],
            all_T_laser_tool_measured_np[:num_effective_samples],
            weights_np
        ).item()
        rmse_results["激光跟踪仪标定参数"] = rmse_specified
    except ValueError as e:
        print(f"获取激光跟踪仪标定参数时出错: {e}")

    # 3. 计算"LM标定后参数（来自CSV）"的RMSE
    calibrated_params_csv = load_calibrated_params_from_csv()
    if calibrated_params_csv is not None:
        print(f"\n使用【LM标定后参数 (来自CSV)】进行RMSE计算 (共 {len(calibrated_params_csv)} 个参数)。")
        rmse_calibrated_csv = compute_total_error_avg(
            calibrated_params_csv,
            all_joint_angles_np[:num_effective_samples],
            all_T_laser_tool_measured_np[:num_effective_samples],
            weights_np
        ).item()
        rmse_results["LM标定后参数 (CSV)"] = rmse_calibrated_csv
    else:
        print("未能加载标定后的参数，跳过其RMSE计算。")

    # 4. 打印RMSE对比结果
    print(f"\n{'='*50}")
    print(f" RMSE 计算结果对比")
    print(f"{'-'*50}")
    if rmse_results:
        for name, rmse_value in rmse_results.items():
            print(f"  {name:<25}: {rmse_value:.8f}")
    else:
        print("  未能计算任何RMSE值。")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
