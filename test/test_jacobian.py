""" 
    测试 jacobian_torch.py 中的函数 
    1. get_laser_tool_matrix  得到激光跟踪仪工具位姿矩阵
    2. forward_kinematics_T  得到正向运动学结果
    3. extract_pose_from_T    从变换矩阵中提取位姿
    4. compute_error_vector_jacobian 计算误差向量对组合参数的雅可比矩阵
"""
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从 jacobian_torch 导入函数和初始参数
from jacobian_torch import (
    get_laser_tool_matrix,
    forward_kinematics_T,
    extract_pose_from_T,
    compute_error_vector_jacobian,
    ERROR_WEIGHTS,
)

# 使用脚本内定义的初始参数和样本角度
INIT_TOOL_OFFSET_POSITION = np.array([1.081, 1.1316, 97.2485])
INIT_TOOL_OFFSET_QUATERNION = np.array([0.5003, 0.5012, 0.5002, 0.4983])
INIT_DH_PARAMS = np.array([0, 0, 380, 0,
                    -90, -90, 0, 30,
                    0, 0, 0, 440,
                    0, -90, 435, 35,
                    0, 90, 0, 0,       # Corrected 5th joint DH based on previous context
                    180, -90, 83, 0])  # Corrected 6th joint DH

SAMPLE_Q_DEG = np.array([42.91441824,-0.414388123,49.04196013,-119.3252973,78.65535552,-5.225972875])

print("--- 测试 jacobian_torch.py 中的函数 ---")

#! --- 1. 测试 get_laser_tool_matrix ---
print("\n1. 测试 get_laser_tool_matrix():")
try:
    laser_matrices = get_laser_tool_matrix()
    print(f"  - 成功加载激光跟踪仪矩阵。形状: {laser_matrices.shape}")
    sample_laser_matrix_np = laser_matrices[0] # 使用 NumPy 数组
    print(f"  - 第一个激光矩阵 (NumPy):\n{sample_laser_matrix_np}")

    # 从第一个激光矩阵提取位姿
    try:
        sample_laser_matrix_torch = torch.tensor(sample_laser_matrix_np, dtype=torch.float64)
        pose_from_laser = extract_pose_from_T(sample_laser_matrix_torch)
        print(f"  - 从第一个激光矩阵提取的位姿 (x,y,z,rx,ry,rz): {pose_from_laser.numpy()}")
    except Exception as e_extract:
        print(f"  - 从激光矩阵提取位姿时出错: {e_extract}")

except Exception as e:
    print(f"  - 调用 get_laser_tool_matrix 时出错: {e}")
    sample_laser_matrix_np = None # 确保变量不存在以跳过后续测试

#! --- 2. 测试 forward_kinematics_T ---
print("\n2. 测试 forward_kinematics_T():")
T_fk = None # 初始化 T_fk 为 None
try:
    # 准备输入: 使用全局定义的样本角度和参数
    sample_q_deg_torch = torch.tensor(SAMPLE_Q_DEG, dtype=torch.float64)
    # 确保初始四元数已归一化
    norm_q = np.linalg.norm(INIT_TOOL_OFFSET_QUATERNION)
    tool_quat_norm = INIT_TOOL_OFFSET_QUATERNION / norm_q if norm_q > 1e-9 else np.array([0,0,0,1])
    initial_params = np.concatenate((INIT_DH_PARAMS, INIT_TOOL_OFFSET_POSITION, tool_quat_norm))
    params_torch = torch.tensor(initial_params, dtype=torch.float64)

    print(f"  - 使用关节角度测试: {SAMPLE_Q_DEG}")
    T_fk = forward_kinematics_T(sample_q_deg_torch, params_torch) # 计算结果赋给 T_fk
    print(f"  - 得到的 T_base_tool:\n{T_fk.detach().numpy()}")

    # 测试关节限位违规 (保持这个测试)
    print("  - 测试关节限位违规 (关节 1 > 170):")
    q_invalid = torch.tensor([180.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    # 需要从 jacobian_torch 导入 JOINT_LIMITS 才能运行这个测试
    try:
        # 需要临时从 jacobian_torch 导入 JOINT_LIMITS 或在测试脚本中定义它
        from jacobian_torch import JOINT_LIMITS
        forward_kinematics_T(q_invalid, params_torch)
    except ValueError as ve:
        print(f"  - 成功捕获预期错误: {ve}")
    except NameError:
         print("  - 跳过关节限位测试，因为 JOINT_LIMITS 未在测试脚本中定义或导入。")

except Exception as e:
    print(f"  - 调用 forward_kinematics_T 时出错: {e}")

#! --- 3. 测试 extract_pose_from_T ---
print("\n3. 测试 extract_pose_from_T():")
try:
    # 检查 T_fk 是否在测试 2 中成功计算
    if T_fk is not None:
        pose_extracted = extract_pose_from_T(T_fk)
        print(f"  - 从正向运动学结果 (T_fk) 提取的位姿: {pose_extracted.detach().numpy()}")
    else:
        print("  - 因前面的正向运动学 (测试 2) 未成功计算 T_fk，跳过提取测试。")

except Exception as e:
    print(f"  - 调用 extract_pose_from_T 时出错: {e}")

#! --- 4. 测试 compute_error_vector_jacobian ---
print("\n4. 测试 compute_error_vector_jacobian():")
try:
    # 检查前面步骤的必要输入是否存在
    if sample_laser_matrix_np is not None and 'initial_params' in locals():
        q_np = SAMPLE_Q_DEG
        print(f"  - 计算样本0的雅可比矩阵，关节角度: {q_np}")
        # 使用 NumPy 数组作为 laser_matrix 输入
        J = compute_error_vector_jacobian(initial_params, q_np, sample_laser_matrix_np, ERROR_WEIGHTS)
        print(f"  - 雅可比矩阵已计算。形状: {J.shape}")
    else:
        print("  - 因缺少前面步骤的输入，跳过雅可比矩阵测试。")

except Exception as e:
    print(f"  - 调用 compute_error_vector_jacobian 时出错: {e}")

print("\n--- 测试完成 ---")
