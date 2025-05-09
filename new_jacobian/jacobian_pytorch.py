import torch
import numpy as np
import os
import torch.autograd.functional as F

# PyTorch版DH变换矩阵，输入均为弧度/浮点
def torch_modified_dh_matrix(theta_val_rad, alpha_val_rad, d_val, a_val):
    cos_theta = torch.cos(theta_val_rad)
    sin_theta = torch.sin(theta_val_rad)
    cos_alpha = torch.cos(alpha_val_rad)
    sin_alpha = torch.sin(alpha_val_rad)
    
    # 使用PyTorch创建矩阵
    A = torch.zeros(4, 4, dtype=torch.float64)
    A[0, 0] = cos_theta
    A[0, 1] = -sin_theta
    A[0, 2] = 0
    A[0, 3] = a_val
    
    A[1, 0] = sin_theta * cos_alpha
    A[1, 1] = cos_theta * cos_alpha
    A[1, 2] = -sin_alpha
    A[1, 3] = -sin_alpha * d_val
    
    A[2, 0] = sin_theta * sin_alpha
    A[2, 1] = cos_theta * sin_alpha
    A[2, 2] = cos_alpha
    A[2, 3] = cos_alpha * d_val
    
    A[3, 0] = 0
    A[3, 1] = 0
    A[3, 2] = 0
    A[3, 3] = 1
    
    return A

# PyTorch版正运动学，输入：24维DH参数（度/长度），6维关节角度（度）
def torch_forward_kinematics_T_for_dh_sensitivity(all_dh_params_flat_torch, q_deg_array_fixed):
    # 初始化单位矩阵
    T_total_torch = torch.eye(4, dtype=torch.float64)
    num_joints = 6
    
    # 将NumPy数组转换为PyTorch张量（如果输入不是张量）
    if not torch.is_tensor(q_deg_array_fixed):
        q_deg_array_fixed = torch.tensor(q_deg_array_fixed, dtype=torch.float64)
    
    for i in range(num_joints):
        base_idx = i * 4
        theta_offset_i = all_dh_params_flat_torch[base_idx + 0]
        alpha_i_deg    = all_dh_params_flat_torch[base_idx + 1]
        d_i            = all_dh_params_flat_torch[base_idx + 2]
        a_i            = all_dh_params_flat_torch[base_idx + 3]
        
        q_i_deg = q_deg_array_fixed[i]
        actual_theta_i_deg = q_i_deg + theta_offset_i
        
        # 转换为弧度
        actual_theta_i_rad = torch.deg2rad(actual_theta_i_deg)
        alpha_i_rad        = torch.deg2rad(alpha_i_deg)
        
        # 计算变换矩阵
        A_i_torch = torch_modified_dh_matrix(actual_theta_i_rad, alpha_i_rad, d_i, a_i)
        T_total_torch = T_total_torch @ A_i_torch
    
    return T_total_torch

# 从4x4变换矩阵提取6维位姿向量
def torch_extract_pose_from_T(T_torch, euler_convention='zyx'):
    position_torch = T_torch[0:3, 3]
    R_torch = T_torch[0:3, 0:3]
    
    if euler_convention == 'zyx':
        r11, r12, r13 = R_torch[0,0], R_torch[0,1], R_torch[0,2]
        r21, r22, r23 = R_torch[1,0], R_torch[1,1], R_torch[1,2]
        r31, r32, r33 = R_torch[2,0], R_torch[2,1], R_torch[2,2]
        
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
    return torch.cat([position_torch, euler_angles_torch])


def torch_final_fk_for_dh_sensitivity(all_dh_params_flat_torch, fixed_joint_angles):
    T_total_torch = torch_forward_kinematics_T_for_dh_sensitivity(all_dh_params_flat_torch, fixed_joint_angles)
    pose_vector_torch = torch_extract_pose_from_T(T_total_torch)
    return pose_vector_torch

# 主函数
if __name__ == "__main__":

    q_deg_array_current_fixed = np.array([42.91441824,-0.414388123,49.04196013,-119.3252973,78.65535552,-5.225972875])
    initial_dh_params_flat_list = [0, 0, 380, 0, -90, -90, 0, 30, 0, 0, 0, 440, 0, -90, 435, 35, 0, 90, 0, 0, 180, -90, 83, 0]
    
    # 转换为torch张量并要求梯度计算
    current_dh_params_torch = torch.tensor(initial_dh_params_flat_list, dtype=torch.float64, requires_grad=True)
    q_deg_array_torch = torch.tensor(q_deg_array_current_fixed, dtype=torch.float64)
    
    # 包装函数
    def compute_pose_from_dh_torch(dh_params_torch):
        return torch_final_fk_for_dh_sensitivity(dh_params_torch, q_deg_array_torch)
    
    # 使用torch.autograd.functional.jacobian计算雅可比矩阵
    print("正在计算PyTorch的雅可比矩阵...")
    jacobian_matrix_6x24_torch = F.jacobian(compute_pose_from_dh_torch, current_dh_params_torch)
    
    # 雅可比矩阵形状调整
    jacobian_matrix_6x24_torch = jacobian_matrix_6x24_torch.squeeze()

    # 将结果转为numpy保存为CSV
    if not os.path.exists("data"):
        os.makedirs("data")
    torch_np_jacobian = jacobian_matrix_6x24_torch.detach().numpy()
    np.savetxt("data/jacobian_pytorch_result.csv", torch_np_jacobian, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到文件: data/jacobian_pytorch_result.csv")
    
    # 计算变换矩阵
    T_total = torch_forward_kinematics_T_for_dh_sensitivity(current_dh_params_torch, q_deg_array_torch)
    print("\n最终变换矩阵:")
    print(T_total.detach().numpy())
    

    

