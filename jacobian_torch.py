import os
import numpy as np
import torch
import torch.autograd.functional as F
from error_function import get_laser_tool_matrix

#* 常量定义
JOINT_ANGLE_FILE = 'data/joint_angle.csv'
LASER_POS_FILE = 'data/laser_pos.csv'
ERROR_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

#* 全局DH参数: theta_offset, alpha, d, a
GLOBAL_DH_PARAMS = [0, 0, 380, 0,
                    -90, -90, 0, 30,
                    0, 0, 0, 440,
                    0, -90, 435, 35,
                    0, 90, 0, 0,
                    180, -90, 83, 0]

#* 添加关节限位(度)
JOINT_LIMITS = np.array([
    [-170, 170],
    [-96, 130],
    [-195, 65],
    [-179, 170],
    [-95, 95],
    [-180, 180]
])


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

#! 四元数转旋转矩阵
def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = torch.zeros(3, 3, dtype=torch.float64)
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*w*z
    R[0, 2] = 2*x*z + 2*w*y
    R[1, 0] = 2*x*y + 2*w*z
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*w*x
    R[2, 0] = 2*x*z - 2*w*y
    R[2, 1] = 2*y*z + 2*w*x
    R[2, 2] = 1 - 2*x*x - 2*y*y
    return R

#! 正向运动学，计算基座到工具的变换矩阵，并检查关节限位
def forward_kinematics_T(q_deg_array, params_torch):
    """
    计算基座到工具的变换矩阵。

    参数:
    q_deg_array (torch.Tensor): 6个关节的角度 (度).
    params_torch (torch.Tensor): 包含31个参数的张量:
                                 - 前24个是DH参数 (theta_offset, alpha, d, a for 6 joints)
                                 - 后7个是TCP参数 (3位置 + 4四元数)
    返回:
    torch.Tensor: 基座到工具的4x4变换矩阵.
    """
    q_deg_array = torch.as_tensor(q_deg_array, dtype=torch.float64)

    # 解析参数
    dh_params = params_torch[0:24]
    tool_offset_position = params_torch[24:27]
    tool_offset_quaternion = params_torch[27:31]

    #* 检查关节限位
    q_list = q_deg_array.detach().cpu().numpy().tolist()
    for idx, q in enumerate(q_list):
        min_lim, max_lim = JOINT_LIMITS[idx]
        if q < min_lim or q > max_lim:
            raise ValueError(f"关节{idx+1}角度 {q}° 超出限位范围 [{min_lim}°, {max_lim}°]")

    #* 计算基座到末端法兰的变换
    T_totle = torch.eye(4, dtype=torch.float64)
    num_joints = 6
    for i in range(num_joints):
        base_idx = i * 4
        theta_offset_i = dh_params[base_idx]    
        alpha_i_deg    = dh_params[base_idx + 1] 
        d_i            = dh_params[base_idx + 2] 
        a_i            = dh_params[base_idx + 3] 
        q_i_deg = q_deg_array[i]
        actual_theta_deg = q_i_deg + theta_offset_i
        actual_theta_rad = torch.deg2rad(actual_theta_deg)
        alpha_rad = torch.deg2rad(alpha_i_deg)
        A_i = modified_dh_matrix(actual_theta_rad, alpha_rad, d_i, a_i)
        T_totle = T_totle @ A_i

    #* 添加工具偏移，计算基座到工具的变换
    T_flange_tool = torch.eye(4, dtype=torch.float64)
    T_flange_tool[0:3, 3] = tool_offset_position
    R_tool = quaternion_to_rotation_matrix(tool_offset_quaternion)
    T_flange_tool[0:3, 0:3] = R_tool
    T_base_tool = T_totle @ T_flange_tool
    return T_base_tool

#! 从变换矩阵提取6维位姿向量
def extract_pose_from_T(T):
    position = T[0:3, 3]
    R = T[0:3, 0:3]
   
    # 计算sy和奇异性
    sy = torch.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-8

    # 非奇异情况下计算
    x1 = torch.atan2(R[2,1], R[2,2])
    y = torch.atan2(-R[2,0], sy)
    z1 = torch.atan2(R[1,0], R[0,0])

    # 奇异情况备用公式
    x2 = torch.atan2(-R[1,2], R[1,1])
    z2 = torch.zeros_like(x1)

    # 选择结果并转度
    x = torch.where(~singular, x1, x2)
    z = torch.where(~singular, z1, z2)
    rx = torch.rad2deg(x)
    ry = torch.rad2deg(y)
    rz = torch.rad2deg(z)
    euler_angles_torch = torch.stack([rx, ry, rz])
        
    #* 连接位置和姿态
    return torch.cat([position, euler_angles_torch])

# 保存雅可比矩阵到 CSV
def save_jacobian_to_csv(jacobian_tensor, filepath='results/jacobian_error_jacobian.csv'):
    jacobian_np = jacobian_tensor.detach().numpy()
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(filepath, jacobian_np, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到: {filepath}")

# #! 计算误差范数对 DH 参数的雅可比
# def compute_error_jacobian(dh_params=GLOBAL_DH_PARAMS, joint_angle_file=JOINT_ANGLE_FILE, weights=ERROR_WEIGHTS, index=0):
#     joint_angles = np.loadtxt(joint_angle_file, delimiter=',', skiprows=1)[index]
#     T_laser_np = get_laser_tool_matrix()[index]
#     #* 转为 torch 张量
#     # dh_torch = torch.tensor(dh_params, dtype=torch.float64, requires_grad=True) # 旧的，仅DH
#     #* 应该使用组合参数，但此函数现在有点冗余，主要关注 _vector_jacobian
#     #* 这里暂时跳过修改，因为核心在 compute_error_vector_jacobian
#     q_torch = torch.tensor(joint_angles, dtype=torch.float64)
#     T_laser_torch = torch.tensor(T_laser_np, dtype=torch.float64)
#     weights_torch = torch.tensor(weights, dtype=torch.float64)

#     #* 定义误差范数函数
#     def err_norm_fn(params_tensor): # 需要接收组合参数
#         T_pred = forward_kinematics_T(q_torch, params_tensor)
#         pose_pred = extract_pose_from_T(T_pred)
#         pose_laser = extract_pose_from_T(T_laser_torch)
#         err_vec = (pose_pred - pose_laser) * weights_torch
#         return torch.linalg.norm(err_vec)

#     # 计算雅可比
#     # J_err = F.jacobian(err_norm_fn, dh_torch) # 旧的
#     #* 需要用组合参数张量来计算，暂时注释掉此函数的实现
#     print("compute_error_jacobian function needs update for combined parameters")
#     J_err = torch.zeros(24) # Placeholder
#     print(J_err)
#     return J_err

#! 计算误差向量对 组合参数 的雅可比
def compute_error_vector_jacobian(params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    """计算单帧数据的误差向量对 组合参数 (DH+TCP) 的雅可比矩阵"""
    # 转为 torch 张量
    params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True) # 组合参数
    q_torch = torch.as_tensor(joint_angles, dtype=torch.float64)
    T_laser_torch = torch.as_tensor(laser_matrix, dtype=torch.float64)
    weights_torch = torch.as_tensor(weights, dtype=torch.float64)

    # 定义误差向量函数
    def err_vec_fn(params_tensor): # 接收组合参数张量
        T_pred = forward_kinematics_T(q_torch, params_tensor) # 使用更新后的正解函数
        pose_pred = extract_pose_from_T(T_pred)
        pose_laser = extract_pose_from_T(T_laser_torch)
        return (pose_pred - pose_laser) * weights_torch

    # 计算并返回雅可比矩阵 (维度应为 6x31)
    J = F.jacobian(err_vec_fn, params_torch) # 针对组合参数计算
    return J

if __name__ == '__main__':
    # 定义初始 TCP 参数 (之前是全局常量)
    initial_tool_offset_position = np.array([1,1,97])
    initial_tool_offset_quaternion = np.array([0.5003, 0.5012, 0.5002, 0.4983])

    # 合并初始 DH 参数和初始 TCP 参数
    initial_params = np.concatenate((GLOBAL_DH_PARAMS, initial_tool_offset_position, initial_tool_offset_quaternion))

    # 加载数据
    joint_angles_data = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices_data = get_laser_tool_matrix()
    frame_index = 0 # 选择要计算的帧

    # 计算并保存雅可比矩阵
    jacobian_result = compute_error_vector_jacobian(
        params=initial_params, 
        joint_angles=joint_angles_data[frame_index],
        laser_matrix=laser_matrices_data[frame_index],
        weights=ERROR_WEIGHTS
    )
    save_jacobian_to_csv(jacobian_result)
