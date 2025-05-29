import numpy as np
import cv2
import os
import sys
from scipy.spatial.transform import Rotation as R
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from jacobian_torch import  modified_dh_matrix
from tools.data_loader import (
    load_joint_angles,
    extract_laser_positions_from_raw,
    get_file_path,
    load_dh_params
)


np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

def calculate_T_flange(joint_angles_data=None):
    """计算基座到法兰的变换矩阵（只使用DH参数）"""
    if joint_angles_data is None:
        joint_angles_data = load_joint_angles()
    
    # 加载DH参数
    dh_params = load_dh_params()
    
    T_flange_list = []
    
    # 遍历每组关节角度
    for i, joint_angles in enumerate(joint_angles_data):
        joint_angles_torch = torch.tensor(joint_angles, dtype=torch.float64)
        dh_params_torch = torch.tensor(dh_params, dtype=torch.float64)
        
        # 初始化累积变换矩阵
        T_total = torch.eye(4, dtype=torch.float64)
        
        # 计算6个关节的DH变换
        for j in range(6):
            base_idx = j * 4
            alpha_deg = dh_params_torch[base_idx]
            a = dh_params_torch[base_idx + 1]
            d = dh_params_torch[base_idx + 2]
            theta_offset_deg = dh_params_torch[base_idx + 3]
            
            # 计算实际关节角度
            actual_theta_deg = joint_angles_torch[j] + theta_offset_deg
            actual_theta_rad = torch.deg2rad(actual_theta_deg)
            alpha_rad = torch.deg2rad(alpha_deg)
            
            # 计算当前关节的DH变换矩阵
            A_j = modified_dh_matrix(actual_theta_rad, alpha_rad, d, a)
            
            # 累积变换
            T_total = T_total @ A_j
        
        # 转换为numpy并添加到列表
        T_flange_numpy = T_total.detach().numpy()
        T_flange_list.append(T_flange_numpy)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"已计算 {i+1}/{len(joint_angles_data)} 个法兰变换矩阵")
    
    print(f"共计算了 {len(T_flange_list)} 个法兰变换矩阵")
    return T_flange_list

def tool_pos_to_transform_matrix(tool_pos_data):

    Tool_transform_matrix_list = []
    
    for pos in tool_pos_data:
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

   
    X = np.eye(4)
    X[:3, :3] = R_gripper2tool
    X[:3, 3] = t_gripper2tool.flatten()
  
    Y_inv = np.eye(4)
    Y_inv[:3, :3] = R_base2world
    Y_inv[:3, 3] = t_base2world.flatten()
    
    # 计算 Y (Laser -> Base)，即 Y_inv 的逆矩阵
    Y = np.linalg.inv(Y_inv)


    # 在控制台输出结果
    print(f"\n--- AX=YB 标定结果 (PARK 方法) ---")
    X_pos = X[:3, 3]
    X_rot = R.from_matrix(X[:3, :3])
    X_quat = X_rot.as_quat()
    print("X (实际为激光基座参数):") # 实际上X是激光基座的大数值
    print("  矩阵:")
    print(X)
    print(f"  平移 (x, y, z): {X_pos[0]:.6f}, {X_pos[1]:.6f}, {X_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {X_quat[0]:.6f}, {X_quat[1]:.6f}, {X_quat[2]:.6f}, {X_quat[3]:.6f}")
    
    Y_pos = Y[:3, 3]
    Y_rot = R.from_matrix(Y[:3, :3])
    Y_quat = Y_rot.as_quat()
    print("\nY (实际为TCP偏移参数):") # 实际上Y是TCP偏移的小数值
    print("  矩阵:")
    print(Y)
    print(f"  平移 (x, y, z): {Y_pos[0]:.6f}, {Y_pos[1]:.6f}, {Y_pos[2]:.6f}")
    print(f"  旋转 (四元数 x, y, z, w): {Y_quat[0]:.6f}, {Y_quat[1]:.6f}, {Y_quat[2]:.6f}, {Y_quat[3]:.6f}")

    # 保存标定结果到配置的结果文件
    results_file = get_file_path('calibration_results')
    # 确保目录存在
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, mode='w', encoding='utf-8') as file:
        # 直接互换X和Y的含义：
        # X实际是激光基座参数（大数值） -> 但我们要把它作为base保存
        # Y实际是TCP偏移参数（小数值） -> 但我们要把它作为tool保存
        
        # 直接互换：Y作为TCP偏移，X作为激光基座
        tool_data = Y_pos.tolist() + Y_quat.tolist()  # Y是TCP偏移 (小数值)
        base_data = X_pos.tolist() + X_quat.tolist()  # X是激光基座 (大数值)
        
        print(f"修正后：Y作为TCP偏移参数，X作为激光基座参数")
        
        # 写入文件：tool行是TCP偏移，base行是激光基座变换
        file.write(f"tool: {tool_data}\n")  # TCP偏移 (Y的小数值)
        file.write(f"base: {base_data}\n")  # 激光基座 (X的大数值)
    
    print(f"\n标定结果已保存到: {results_file}")

    return X, Y, Y_inv



# 测试
if __name__ == "__main__":
    print("开始AX=YB标定...")
    
    # 直接从配置的原始数据文件中读取数据
    joint_angles_data = load_joint_angles()
    tool_pos_data = extract_laser_positions_from_raw()
    
    print(f"加载了 {joint_angles_data.shape[0]} 组关节角度数据")
    print(f"加载了 {tool_pos_data.shape[0]} 组激光位置数据")

    # 计算法兰变换矩阵
    T_flange_list = calculate_T_flange(joint_angles_data)
    
    # 将工具位姿转换为变换矩阵
    Tool_transform_matrix_list = tool_pos_to_transform_matrix(tool_pos_data)

    # 调用 AX=YB 标定函数 (只使用 PARK 方法)
    X_flange2tool, Y_laser2base, Y_inv_base2laser = calibrate_AX_equals_YB(
        T_flange_list, 
        Tool_transform_matrix_list
    )
    
    print("AX=YB标定完成！")

