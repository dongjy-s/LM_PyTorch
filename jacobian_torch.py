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

#* 工具偏移(位置mm + 四元数[x,y,z,w])
TOOL_OFFSET_POSITION = np.array([1.081, 1.1316, 97.2485])
TOOL_OFFSET_QUATERNION = np.array([0.5003, 0.5012, 0.5002, 0.4983])

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
def forward_kinematics_T(q_deg_array, dh_params=None):
    # 默认使用全局DH参数
    if dh_params is None:
        dh_params = GLOBAL_DH_PARAMS
    # 将关节角度转换为tensor并准备检查限位
    if not torch.is_tensor(q_deg_array):
        q_deg_array = torch.tensor(q_deg_array, dtype=torch.float64)
    # 检查关节限位
    q_list = q_deg_array.detach().cpu().numpy().tolist()
    for idx, q in enumerate(q_list):
        min_lim, max_lim = JOINT_LIMITS[idx]
        if q < min_lim or q > max_lim:
            raise ValueError(f"关节{idx+1}角度 {q}° 超出限位范围 [{min_lim}°, {max_lim}°]")
    # 计算基座到末端法兰的变换
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
        alpha_rad = torch.deg2rad(torch.tensor(alpha_i_deg, dtype=torch.float64))
        A_i = modified_dh_matrix(actual_theta_rad, alpha_rad, d_i, a_i)
        T_totle = T_totle @ A_i
    # 添加工具偏移，计算基座到工具的变换
    T_flange_tool = torch.eye(4, dtype=torch.float64)
    T_flange_tool[0:3, 3] = torch.tensor(TOOL_OFFSET_POSITION, dtype=torch.float64)
    R_tool = quaternion_to_rotation_matrix(torch.tensor(TOOL_OFFSET_QUATERNION, dtype=torch.float64))
    T_flange_tool[0:3, 0:3] = R_tool
    T_base_tool = T_totle @ T_flange_tool
    return T_base_tool

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

# 保存雅可比矩阵到 CSV
def save_jacobian_to_csv(jacobian_tensor, filepath='results/jacobian_error_jacobian.csv'):
    jacobian_np = jacobian_tensor.detach().numpy()
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(filepath, jacobian_np, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到: {filepath}")

#! 计算误差范数对 DH 参数的雅可比
def compute_error_jacobian(dh_params=None, joint_angle_file=JOINT_ANGLE_FILE, weights=ERROR_WEIGHTS, index=0):
    if dh_params is None:
        dh_params = GLOBAL_DH_PARAMS
    # 读取关节角度
    joint_angles = np.loadtxt(joint_angle_file, delimiter=',', skiprows=1)[index]
    # 读取激光测量矩阵
    T_laser_np = get_laser_tool_matrix()[index]
    # 转为 torch 张量
    dh_torch = torch.tensor(dh_params, dtype=torch.float64, requires_grad=True)
    q_torch = torch.tensor(joint_angles, dtype=torch.float64)
    T_laser_torch = torch.tensor(T_laser_np, dtype=torch.float64)
    weights_torch = torch.tensor(weights, dtype=torch.float64)

    # 定义误差范数函数
    def err_norm_fn(dh_tensor):
        T_pred = forward_kinematics_T(q_torch, dh_tensor)
        pose_pred = extract_pose_from_T(T_pred)
        pose_laser = extract_pose_from_T(T_laser_torch)
        err_vec = (pose_pred - pose_laser) * weights_torch
        return torch.linalg.norm(err_vec)

    # 计算雅可比
    J_err = F.jacobian(err_norm_fn, dh_torch)
    return J_err

#! 计算误差向量对 DH 参数的雅可比
def compute_error_vector_jacobian(joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    """
    对单帧数据，计算误差向量 (6,) 对 DH 参数 (24,) 的雅可比矩阵，返回形状 (6,24) 的 torch.Tensor
    """
    # 构建 torch 张量
    dh_torch = torch.tensor(GLOBAL_DH_PARAMS, dtype=torch.float64, requires_grad=True)
    q_torch = torch.tensor(joint_angles, dtype=torch.float64)
    T_laser_torch = torch.tensor(laser_matrix, dtype=torch.float64)
    weights_torch = torch.tensor(weights, dtype=torch.float64)
    # 定义输出误差向量函数
    def err_vec_fn(dh):
        T_pred = forward_kinematics_T(q_torch, dh)
        pose_pred = extract_pose_from_T(T_pred)
        pose_laser = extract_pose_from_T(T_laser_torch)
        return (pose_pred - pose_laser) * weights_torch
    # 计算雅可比
    J = F.jacobian(err_vec_fn, dh_torch)
    return J

if __name__ == '__main__':
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)[0]
    T_base_tool = forward_kinematics_T(joint_angles)
    print(T_base_tool)


