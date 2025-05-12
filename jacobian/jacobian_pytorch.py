"""使用PyTorch的自动微分计算雅可比矩阵"""

import torch
import numpy as np
import os
import torch.autograd.functional as F

#! MDH变换矩阵公式,角度单位为弧度，长度单位为毫米
def modified_dh_matrix(theta_val_rad, alpha_val_rad, d_val, a_val):
    cos_theta = torch.cos(theta_val_rad)
    sin_theta = torch.sin(theta_val_rad)
    cos_alpha = torch.cos(alpha_val_rad)
    sin_alpha = torch.sin(alpha_val_rad)
    
    # 使用PyTorch创建矩阵
    MDH_matrix = torch.zeros(4, 4, dtype=torch.float64)
    MDH_matrix[0, 0] = cos_theta
    MDH_matrix[0, 1] = -sin_theta
    MDH_matrix[0, 2] = 0
    MDH_matrix[0, 3] = a_val
    
    MDH_matrix[1, 0] = sin_theta * cos_alpha
    MDH_matrix[1, 1] = cos_theta * cos_alpha
    MDH_matrix[1, 2] = -sin_alpha
    MDH_matrix[1, 3] = -sin_alpha * d_val
    
    MDH_matrix[2, 0] = sin_theta * sin_alpha
    MDH_matrix[2, 1] = cos_theta * sin_alpha
    MDH_matrix[2, 2] = cos_alpha
    MDH_matrix[2, 3] = cos_alpha * d_val
    
    MDH_matrix[3, 0] = 0
    MDH_matrix[3, 1] = 0
    MDH_matrix[3, 2] = 0
    MDH_matrix[3, 3] = 1
    return MDH_matrix   

#! 正向运动学，得到0-6的变换矩阵
def forward_kinematics_T(dh_params, q_deg_array):
    T_totle = torch.eye(4, dtype=torch.float64)
    num_joints = 6
    
    # 将关节角度转换为tensor
    if not torch.is_tensor(q_deg_array):
        q_deg_array = torch.tensor(q_deg_array, dtype=torch.float64)
    
    for i in range(num_joints):
        base_idx = i * 4
        theta_offset_i = dh_params[base_idx + 0]
        alpha_i_deg    = dh_params[base_idx + 1]
        d_i            = dh_params[base_idx + 2]
        a_i            = dh_params[base_idx + 3]
        
        q_i_deg = q_deg_array[i]
        actual_theta_i_deg = q_i_deg + theta_offset_i
        
        # 转换为弧度
        actual_theta_i_rad = torch.deg2rad(actual_theta_i_deg)
        alpha_i_rad        = torch.deg2rad(alpha_i_deg)
        
        # 计算变换矩阵
        A_i_MDH = modified_dh_matrix(actual_theta_i_rad, alpha_i_rad, d_i, a_i)
        T_totle = T_totle @ A_i_MDH
    
    return T_totle

#! 从变换矩阵提取6维位姿向量
def extract_pose_from_T(T, euler_convention='zyx'):
    position = T[0:3, 3]
    R = T[0:3, 0:3]
    
    if euler_convention == 'zyx':
        r11, r12, r13 = R[0,0], R[0,1], R[0,2]
        r21, r22, r23 = R[1,0], R[1,1], R[1,2]
        r31, r32, r33 = R[2,0], R[2,1], R[2,2]
        
        # 计算pitch角，这是ZYX欧拉角的核心，可能导致万向锁问题
        ry = torch.asin(-r31)
        cos_ry = torch.cos(ry)
        
        # 使用条件计算避免除零问题，处理奇异点
        # 当cos_ry接近零时，我们处于万向锁奇异点，rx和rz不能独立确定
        epsilon = 1e-6
        mask = torch.abs(cos_ry) > epsilon
        
        # 计算rz (yaw)，处理奇异情况
        rz_normal = torch.atan2(r21, r11)
        rz = mask * rz_normal  # 如果cos_ry接近0，则rz为0
        
        # 计算rx (roll)，处理奇异情况
        rx_normal = torch.atan2(r32, r33)
        rx = mask * rx_normal  # 如果cos_ry接近0，则rx为0
        
        euler_angles_torch = torch.stack([rx, ry, rz])
    else:
        raise ValueError("Unsupported Euler angle convention")
        
    # 连接位置和姿态
    return torch.cat([position, euler_angles_torch])

#! 包装前向运动学函数
def fk_for_dh_sensitivity(dh_params, joint_angles):
    T_totle = forward_kinematics_T(dh_params, joint_angles)
    pose_vectoR = extract_pose_from_T(T_totle)
    return pose_vectoR

#! 前向运动学敏感度函数，接受 DH 参数和关节角度
def pose_for_jacobian(dh_params_torch, q_deg_array_torch):
    return fk_for_dh_sensitivity(dh_params_torch, q_deg_array_torch)

#! 计算雅可比矩阵，只对 DH 参数求导
def compute_jacobian_matrix(dh_params_torch, q_deg_array_torch):
    J_dh, _ = F.jacobian(pose_for_jacobian, (dh_params_torch, q_deg_array_torch))
    return J_dh

#! 保存张量雅可比到 CSV
def save_jacobian_to_csv(jacobian_tensor, filepath="results/jacobian_pytorch_result.csv"):
    # 转为 NumPy 并保存
    jacobian_np = jacobian_tensor.detach().numpy()
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(filepath, jacobian_np, delimiter=',', fmt='%.12f')
    print(f"DH 参数雅可比矩阵已保存到文件: {filepath}")

# 主函数
if __name__ == "__main__":
    #* 初始化DH参数和关节角度（theta_offset, alpha, d, a）
    q_deg_array = np.array([42.91441824,-0.414388123,49.04196013,-119.3252973,78.65535552,-5.225972875])
    dh_params = [0, 0, 380, 0, -90, -90, 0, 30, 0, 0, 0, 440, 0, -90, 435, 35, 0, 90, 0, 0, 180, -90, 83, 0]
    
    #* 把关节角度和DH参数转换为torch
    dh_params_torch = torch.tensor(dh_params, dtype=torch.float64, requires_grad=True)
    q_deg_array_torch = torch.tensor(q_deg_array, dtype=torch.float64)
    
    #* 计算 DH 参数雅可比矩阵
    print("正在计算PyTorch的DH参数雅可比矩阵...")
    J_dh_torch = compute_jacobian_matrix(dh_params_torch, q_deg_array_torch).squeeze()
    
    # 保存 DH 参数雅可比矩阵
    save_jacobian_to_csv(J_dh_torch)
    
    # 计算变换矩阵
    T_total = forward_kinematics_T(dh_params_torch, q_deg_array_torch)
    print("\n最终变换矩阵:")
    print(T_total.detach().numpy())
    

    

