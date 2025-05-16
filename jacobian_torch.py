import os
import numpy as np
import torch
import torch.autograd.functional as F
from scipy.spatial.transform import Rotation
import pandas as pd

#! 常量定义
JOINT_ANGLE_FILE = 'data/joint_angle.csv'
LASER_POS_FILE = 'data/laser_pos.csv'
ERROR_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

#! 待优化的DH参数: theta_offset, alpha, d, a 单位:mm,度
<<<<<<< HEAD
GLOBAL_DH_PARAMS = [0, 0, 380, 0,
                    -90, -90, 0, 30,
                    0, 0, 0, 440,
                    0, -90, 435, 35,
                    0, 90, 0, 0,
                    180, -90, 83, 0]
=======
#* 出厂DH参数
INIT_DH_PARAMS = [
    0, 0, 487, 0,
    -90, -90, 0, 85,
    0, 0, 0, 640,
    0, -90, 720, 205,
    0, 90, 0, 0,
    180, -90, 75, 0
]

#* 激光优化的DH参数
# INIT_DH_PARAMS = [
#     -0.0063, 0, 487.4009, 0,
#     -90.2267, -90, 0, 85.5599 ,
#     -0.4689, 0, 0, 639.8143,
#     0.5631, -90, 720.3035, 205.288 ,
#     0.0723, 90, 0, 0,
#     179.6983, -90, 75.7785, 0
# ]
>>>>>>> d65923ab4aa3302280a2aa55d9ac91c940d386cb

#! 关节限位(度)
JOINT_LIMITS = np.array([
    [-100, 100],
    [-90, 100],
    [-100, 600],
    [-100, 100],
    [-90, 90],
    [-120, 120]
])

#! 初始TCP参数
<<<<<<< HEAD
INITIAL_TCP_POSITION = np.array([0.15,1.18,200])
INITIAL_TCP_QUATERNION = np.array([0.50, 0.50, 0.50, 0.50])
=======
# INIT_TOOL_OFFSET_POSITION = np.array([0.15, 1.2, 240])
# INIT_TOOL_OFFSET_QUATERNION = np.array([0.5, 0.5, 0.5, 0.5])
>>>>>>> d65923ab4aa3302280a2aa55d9ac91c940d386cb

#* 激光拟合的TCP参数
INIT_TOOL_OFFSET_POSITION = np.array([0.1731, 1.1801, 238.3535])
INIT_TOOL_OFFSET_QUATERNION = np.array([0.4961, 0.5031, 0.505, 0.4957])
#! 初始基座在激光跟踪仪坐标系下的位姿参数 [x, y, z, qx, qy, qz, qw]

# INIT_T_LASER_BASE_PARAMS = np.array([3611, 3301, 14, 0.005078, -0.005171, 0.786759, -0.617218])
#*激光拟合的基座参数
INIT_T_LASER_BASE_PARAMS = np.array([3610.831933, 3300.7233, 13.6472, 0.0014, -0.0055, 0.7873, -0.6166])

#! 激光跟踪仪测量的位姿转换为变换矩阵
def get_laser_tool_matrix():
    laser_data = pd.read_csv(LASER_POS_FILE, delimiter=',', skiprows=1, header=None).values
    num_samples = laser_data.shape[0]
    laser_tool_matrix = np.zeros((num_samples, 4, 4))
    
    for i, data in enumerate(laser_data):
        x, y, z, rx, ry, rz = data
        
        #* 计算旋转矩阵（xyz内旋）
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        
        #* 创建变换矩阵
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        
        laser_tool_matrix[i] = T
    return laser_tool_matrix

#! 构建MDH变换矩阵
def modified_dh_matrix(theta_val_rad, alpha_val_rad, d_val, a_val):
    """
        MDH变换矩阵公式:
            [cos_theta, -sin_theta, 0, a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
            [0, 0, 0, 1]
    """
    cos_theta = torch.cos(theta_val_rad)
    sin_theta = torch.sin(theta_val_rad)
    cos_alpha = torch.cos(alpha_val_rad)
    sin_alpha = torch.sin(alpha_val_rad)
    
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
    """
        四元数和旋转矩阵变换公式：
            q = [x, y, z, w]
            R = [
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
            ]
    """
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

#! 正向运动学
def forward_kinematics_T(q_deg_array, params_torch):
    """
        正向运动学公式：
            T_base_tool = T_totle @ A_i
    """
    #* 从总参数中提取DH参数和TCP参数
    q_deg_array = torch.as_tensor(q_deg_array, dtype=torch.float64)
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
def save_jacobian_to_csv(jacobian_tensor, filepath='results/PyTorch_jacobian.csv'):
    jacobian_np = jacobian_tensor.detach().numpy()
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(filepath, jacobian_np, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到: {filepath}")

<<<<<<< HEAD
 #! 计算误差范数对 DH 参数的雅可比(冗余)
#// def compute_error_jacobian(dh_params=GLOBAL_DH_PARAMS, joint_angle_file=JOINT_ANGLE_FILE, weights=ERROR_WEIGHTS, index=0):
#//     joint_angles = np.loadtxt(joint_angle_file, delimiter=',', skiprows=1)[index]
#//   T_laser_np = get_laser_tool_matrix()[index]
#//     # 转为 torch 张量
#//     # dh_torch = torch.tensor(dh_params, dtype=torch.float64, requires_grad=True) # 旧的，仅DH
#//     q_torch = torch.tensor(joint_angles, dtype=torch.float64)
#//     T_laser_torch = torch.tensor(T_laser_np, dtype=torch.float64)
#//     weights_torch = torch.tensor(weights, dtype=torch.float64)

#//     #定义误差范数函数
#//     def err_norm_fn(params_tensor): # 需要接收组合参数
#//         T_pred = forward_kinematics_T(q_torch, params_tensor)
#//         pose_pred = extract_pose_from_T(T_pred)
#//         pose_laser = extract_pose_from_T(T_laser_torch)
#//         err_vec = (pose_pred - pose_laser) * weights_torch
#//         return torch.linalg.norm(err_vec)
#//     # 计算雅可比
#//     # J_err = F.jacobian(err_norm_fn, dh_torch) # 旧的
#//     print("compute_error_jacobian function needs update for combined parameters")
#//     J_err = torch.zeros(24) # Placeholder
#//     print(J_err)
#//     return J_err

=======
>>>>>>> d65923ab4aa3302280a2aa55d9ac91c940d386cb
#! 计算误差向量对组合参数的雅可比
def compute_error_vector_jacobian(params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    """计算单组数据的误差向量对 组合参数 (DH+TCP+T_laser_base) 的雅可比矩阵"""
    #* 转为 torch 张量
    params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    q_torch = torch.as_tensor(joint_angles, dtype=torch.float64)
    T_laser_tool_measured_torch = torch.as_tensor(laser_matrix, dtype=torch.float64)
    weights_torch = torch.as_tensor(weights, dtype=torch.float64)

    #* 定义包装函数
    def err_vec_fn(params_tensor):
        #* 提取参数
        params_for_fk = params_tensor[0:31] # DH (24) + TCP (3+4)
        t_laser_base_pos = params_tensor[31:34]
        t_laser_base_quat = params_tensor[34:38]

        #* 1. 计算机器人模型预测的工具在基座坐标系下的位姿
        T_pred_robot_base = forward_kinematics_T(q_torch, params_for_fk)
        
        #* 2. 构建 T_laser_base 变换矩阵 (基座在激光坐标系下的位姿)
        R_laser_base = quaternion_to_rotation_matrix(t_laser_base_quat)
        T_laser_base_matrix = torch.eye(4, dtype=torch.float64)
        T_laser_base_matrix[0:3, 0:3] = R_laser_base
        T_laser_base_matrix[0:3, 3] = t_laser_base_pos
        
        #* 3. 将机器人预测位姿转换到激光跟踪仪坐标系 
        T_pred_in_laser_frame = torch.matmul(T_laser_base_matrix, T_pred_robot_base)

        #* 4. 从变换矩阵提取位姿向量
        pose_pred_in_laser = extract_pose_from_T(T_pred_in_laser_frame)
        pose_measured_in_laser = extract_pose_from_T(T_laser_tool_measured_torch) # Measured data is already in laser frame
        #! 六维误差向量
        return (pose_pred_in_laser - pose_measured_in_laser) * weights_torch

    #* 计算并返回雅可比矩阵 (维度应为 6x38)
    J = F.jacobian(err_vec_fn, params_torch) # 针对组合参数计算
    return J

if __name__ == '__main__':
    
<<<<<<< HEAD
    initial_params = np.concatenate((GLOBAL_DH_PARAMS, INITIAL_TCP_POSITION, INITIAL_TCP_QUATERNION))
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)[0]
    T_laser_np = get_laser_tool_matrix()[0]
    jacobian = compute_error_vector_jacobian(initial_params, joint_angles, T_laser_np, ERROR_WEIGHTS)
=======
    initial_params_np = np.concatenate((
        INIT_DH_PARAMS, 
        INIT_TOOL_OFFSET_POSITION, 
        INIT_TOOL_OFFSET_QUATERNION,
        INIT_T_LASER_BASE_PARAMS 
    ))
    # 将初始参数转换为PyTorch张量以用于后续计算
    initial_params_torch = torch.tensor(initial_params_np, dtype=torch.float64)

    # 加载所有关节角度
    all_joint_angles_np = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    # 获取激光跟踪仪测量的所有工具位姿矩阵 (虽然在这个测试中我们主要用预测的)
    all_T_laser_tool_measured_np = get_laser_tool_matrix()

    print("--- 雅可比矩阵计算（第一帧） ---")
    jacobian = compute_error_vector_jacobian(initial_params_np, all_joint_angles_np[0], all_T_laser_tool_measured_np[0], ERROR_WEIGHTS)
>>>>>>> d65923ab4aa3302280a2aa55d9ac91c940d386cb
    save_jacobian_to_csv(jacobian)
    print("雅可比矩阵已保存（第一帧）。")
    print("\n--- 预测工具位姿在激光坐标系下（前5组关节角） ---")

    # 提取用于正向运动学的参数 (DH + TCP)
    params_for_fk_torch = initial_params_torch[0:31]
    # 提取用于构建T_laser_base的参数
    t_laser_base_pos_torch = initial_params_torch[31:34]
    t_laser_base_quat_torch = initial_params_torch[34:38]

    # 构建 T_laser_base 变换矩阵 (基座在激光坐标系下的位姿)
    R_laser_base_torch = quaternion_to_rotation_matrix(t_laser_base_quat_torch)
    T_laser_base_matrix_torch = torch.eye(4, dtype=torch.float64)
    T_laser_base_matrix_torch[0:3, 0:3] = R_laser_base_torch
    T_laser_base_matrix_torch[0:3, 3] = t_laser_base_pos_torch

    # num_frames_to_test = min(5, all_joint_angles_np.shape[0]) # 测试前5帧或所有帧（如果少于5帧）
    # 指定要测试的帧的索引 (0-based)
    frames_to_test_indices = [1, 8, 18, 26] # 对应用户请求的第 2, 9, 19, 27 组

    # for i in range(num_frames_to_test):
    for i in frames_to_test_indices:
        # 检查索引是否越界
        if i >= all_joint_angles_np.shape[0]:
            print(f"\n警告: 索引 {i+1} 超出关节角度数据范围 (共 {all_joint_angles_np.shape[0]} 组). 跳过此索引.")
            continue

        q_deg_array_np = all_joint_angles_np[i]
        q_torch = torch.as_tensor(q_deg_array_np, dtype=torch.float64)

        print(f"\n关节角度（度） {i+1}: {q_deg_array_np.tolist()}")

        # 1. 计算机器人模型预测的工具在基座坐标系下的位姿
        T_pred_robot_base_torch = forward_kinematics_T(q_torch, params_for_fk_torch)
        
        # 2. 将机器人预测位姿转换到激光跟踪仪坐标系
        T_pred_in_laser_frame_torch = torch.matmul(T_laser_base_matrix_torch, T_pred_robot_base_torch)

        # 3. 从变换矩阵提取位姿向量
        pose_pred_in_laser_torch = extract_pose_from_T(T_pred_in_laser_frame_torch)
        
        pose_pred_in_laser_np = pose_pred_in_laser_torch.detach().cpu().numpy()
        print(f"预测位姿在激光坐标系下 (x,y,z,rx,ry,rz): {pose_pred_in_laser_np.tolist()}")

   
    