import numpy as np
import cv2
import os
import sys
from scipy.spatial.transform import Rotation as R
import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from jacobian_torch import (
    forward_kinematics_T,
    INIT_DH_PARAMS,
    INIT_TOOL_OFFSET_PARAMS,
    JOINT_ANGLE_FILE,
    LASER_POS_FILE
)

# 禁用科学计数法，使输出更易读
np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

def calculate_T_flange(joint_angles_file=None):
    """
    使用 jacobian_torch 中的正运动学计算法兰变换矩阵
    
    参数:
        joint_angles_file: 关节角度文件路径，如果为None则使用默认文件
    
    返回:
        T_flange_list: 包含所有 T_flange (基座到法兰) 的列表
    """
    if joint_angles_file is None:
        joint_angles_file = JOINT_ANGLE_FILE
    
    # 读取关节角度数据
    joint_angles_data = pd.read_csv(joint_angles_file, header=None, skiprows=1).values
    
    # 构建初始参数（DH参数 + TCP参数）
    initial_params = np.concatenate([INIT_DH_PARAMS, INIT_TOOL_OFFSET_PARAMS])
    params_torch = torch.tensor(initial_params, dtype=torch.float64)
    
    T_flange_list = []
    
    # 遍历每组关节角度
    for i, joint_angles in enumerate(joint_angles_data):
        joint_angles_torch = torch.tensor(joint_angles, dtype=torch.float64)
        
        # 使用正运动学计算，返回 (T_base_tool, T_base_flange)
        _, T_base_flange = forward_kinematics_T(joint_angles_torch, params_torch)
        
        # 转换为numpy数组并添加到列表
        T_flange_numpy = T_base_flange.detach().numpy()
        T_flange_list.append(T_flange_numpy)
        
        print(f"第 {i+1} 组关节角度的法兰变换矩阵计算完成")
    
    print(f"共计算了 {len(T_flange_list)} 个法兰变换矩阵")
    return T_flange_list

def tool_pos_to_transform_matrix(tool_pos_list):
    """
    将 tool_pos_list 中的位姿数据转换为变换矩阵(xyz 内旋顺序)
    
    参数:
        tool_pos_list: 包含多组位姿的二维列表，每组为 [x, y, z, rx, ry, rz]
    
    返回:
        Tool_transform_matrix_list: 包含所有变换矩阵的列表
    """
    Tool_transform_matrix_list = []
    
    for pos in tool_pos_list:
        x, y, z, rx, ry, rz = pos
        
        # 1. 处理旋转部分（xyz 内旋）
        rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # 2. 处理平移部分
        translation = np.array([x, y, z])
        
        # 3. 组合为 4x4 变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        
        Tool_transform_matrix_list.append(T)

    
    return Tool_transform_matrix_list

def calibrate_AX_equals_YB(A_list, B_list):
    """
    使用 OpenCV 的 calibrateRobotWorldHandEye 函数求解 AX = YB 标定问题.
    使用 PARK 方法。

    该函数用于确定两个未知变换：
    1. 工具坐标系相对于机器人法兰坐标系的变换 (X: Flange -> Tool) 
    2. 激光跟踪仪坐标系相对于机器人基座坐标系的变换 (Y: Laser -> Base)
    3. 机器人基座坐标系相对于激光跟踪仪坐标系的变换 (Y_inv: Base -> Laser)

    方程为: A * X = Y * B

    其中:
        A: Base -> Flange (机器人法兰相对于基座的位姿，来自机器人正解)
        B: Laser -> Tool (工具末端在激光跟踪仪坐标系下的位姿，来自外部测量)
        X: Flange -> Tool (待求解的法兰到工具的变换)
        Y: Laser -> Base (待求解的激光跟踪仪到基座的变换)
        Y_inv: Base -> Laser (Y的逆变换，基座到激光跟踪仪的变换)

    参数:
        A_list: 包含多个 A 变换矩阵 (4x4 numpy array) 的列表 (Base -> Flange)
        B_list: 包含多个 B 变换矩阵 (4x4 numpy array) 的列表 (Laser -> Tool)

    返回:
        X: 求解得到的 Flange -> Tool 变换矩阵 (4x4 numpy array)
        Y: 求解得到的 Laser -> Base 变换矩阵 (4x4 numpy array)
        Y_inv: 求解得到的 Base -> Laser 变换矩阵，即Y的逆矩阵 (4x4 numpy array)
    """
    if len(A_list) != len(B_list) or len(A_list) < 3:
        raise ValueError("输入列表长度必须相同且至少为 3 (建议更多组非共面/共线的运动)")

    # 分解 A (Base -> Flange)
    R_base2gripper = [A[:3, :3] for A in A_list]
    t_base2gripper = [A[:3, 3].reshape(3, 1) for A in A_list]

    # 分解 B (Laser -> Tool)
    R_world2tool = [B[:3, :3] for B in B_list]
    t_world2tool = [B[:3, 3].reshape(3, 1) for B in B_list]

    # 使用 PARK 方法
    R_base2world, t_base2world, R_gripper2tool, t_gripper2tool = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2tool,   
        t_world2cam=t_world2tool,   
        R_base2gripper=R_base2gripper, 
        t_base2gripper=t_base2gripper,  
        method=cv2.CALIB_HAND_EYE_PARK 
    )

    # 组合 X (Flange -> Tool)
    X = np.eye(4)
    X[:3, :3] = R_gripper2tool
    X[:3, 3] = t_gripper2tool.flatten()

    # R_base2world, t_base2world 定义了从机器人基座到世界坐标系(激光跟踪仪)的变换
    # 因此，这个变换是 Base -> Laser，根据约定，它应该是 Y_inv
    # 组合 Y_inv (Base -> Laser/World)
    Y_inv = np.eye(4)
    Y_inv[:3, :3] = R_base2world
    Y_inv[:3, 3] = t_base2world.flatten()
    
    # 计算 Y (Laser -> Base)，即 Y_inv 的逆矩阵
    Y = np.linalg.inv(Y_inv)

    # 创建 results 目录（如果不存在）
    os.makedirs('results', exist_ok=True)

    # 在控制台输出结果
    print(f"\n--- AX=YB 标定结果 (PARK 方法) ---")
    X_pos = X[:3, 3]
    X_rot = R.from_matrix(X[:3, :3])
    X_quat = X_rot.as_quat()
    print("X (Laser -> Base):") # X 是激光跟踪仪到基座的变换
    print("  矩阵:")
    print(X)
    print(f"  平移 (x, y, z): {X_pos[0]:.6f}, {X_pos[1]:.6f}, {X_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {X_quat[0]:.6f}, {X_quat[1]:.6f}, {X_quat[2]:.6f}, {X_quat[3]:.6f}")
    
    Y_pos = Y[:3, 3]
    Y_rot = R.from_matrix(Y[:3, :3])
    Y_quat = Y_rot.as_quat()
    print("\nY (Flange -> Tool):") # Y 是法兰到工具的变换
    print("  矩阵:")
    print(Y)
    print(f"  平移 (x, y, z): {Y_pos[0]:.6f}, {Y_pos[1]:.6f}, {Y_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {Y_quat[0]:.6f}, {Y_quat[1]:.6f}, {Y_quat[2]:.6f}, {Y_quat[3]:.6f}")

    # 保存标定结果到 CSV 文件
    with open('results/calibration_results.csv', mode='w', encoding='utf-8') as file:
        base_data = X_pos.tolist() + X_quat.tolist()
        tool_data = Y_pos.tolist() + Y_quat.tolist()
        file.write(f"base: {base_data}\n")
        file.write(f"tool: {tool_data}\n")

    return X, Y, Y_inv



# 测试
if __name__ == "__main__":
    # 读取关节角度数据
    df_joint_angles = pd.read_csv(JOINT_ANGLE_FILE, header=None, skiprows=1)
    joint_angles_list = df_joint_angles.iloc[:, :6].values.tolist()

    # 读取工具位姿数据
    df_tool_pos = pd.read_csv(LASER_POS_FILE, header=None, skiprows=1)
    tool_pos_list = df_tool_pos.iloc[:, :6].values.tolist()

    T_flange_list = calculate_T_flange()
    Tool_transform_matrix_list = tool_pos_to_transform_matrix(tool_pos_list)


    #! 调用 AX=YB 标定函数 (只使用 PARK 方法)
    X_flange2tool, Y_laser2base, Y_inv_base2laser = calibrate_AX_equals_YB(
        T_flange_list, 
        Tool_transform_matrix_list
    )

