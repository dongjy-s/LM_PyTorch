import os
import numpy as np
import torch
import torch.autograd.functional as F
from tools.data_loader import (
    get_laser_tool_matrix, load_joint_angles,  
    ERROR_WEIGHTS, JOINT_LIMITS,
    get_initial_params
)

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

#! 把四元数转换为旋转矩阵
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
    #* 提取关节DH参数
    for i in range(num_joints):
        base_idx = i * 4
        alpha_i_deg    = dh_params[base_idx]
        a_i            = dh_params[base_idx + 1]
        d_i            = dh_params[base_idx + 2]
        theta_offset_i = dh_params[base_idx + 3]

        #* 计算该关节角度下变换矩阵
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
    return T_base_tool, T_totle

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
    return torch.cat([position, euler_angles_torch])

#! 保存雅可比矩阵
def save_jacobian_to_csv(jacobian_tensor, filepath='results/PyTorch_jacobian.csv'):
    jacobian_np = jacobian_tensor.detach().numpy()
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(filepath, jacobian_np, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到: {filepath}")

#! 将PyTorch旋转矩阵转换为四元数 [qx, qy, qz, qw]
def _rotation_matrix_to_quaternion_torch(R_matrix):
    """ 
    旋转矩阵转换为四元数公式：
        q = [qx, qy, qz, qw]
        q = [
            (R_matrix[2,1] - R_matrix[1,2]) / S,
            (R_matrix[0,2] - R_matrix[2,0]) / S,
            (R_matrix[1,0] - R_matrix[0,1]) / S,
            0.25 * S
        ]
    """
    if not torch.is_tensor(R_matrix):
        R_matrix = torch.as_tensor(R_matrix, dtype=torch.float64)
    elif R_matrix.dtype != torch.float64: # 确保数据类型为 float64
        R_matrix = R_matrix.to(dtype=torch.float64)

    q = torch.zeros(4, dtype=R_matrix.dtype, device=R_matrix.device)
    trace = R_matrix[0,0] + R_matrix[1,1] + R_matrix[2,2]

    if trace > 1e-8:
        S = torch.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * S 
        q[0] = (R_matrix[2,1] - R_matrix[1,2]) / S 
        q[1] = (R_matrix[0,2] - R_matrix[2,0]) / S 
        q[2] = (R_matrix[1,0] - R_matrix[0,1]) / S 
    elif (R_matrix[0,0] > R_matrix[1,1]) and (R_matrix[0,0] > R_matrix[2,2]):
        S = torch.sqrt(1.0 + R_matrix[0,0] - R_matrix[1,1] - R_matrix[2,2] + 1e-12) * 2.0 # 添加eps防止开方负数
        q[0] = 0.25 * S 
        q[1] = (R_matrix[0,1] + R_matrix[1,0]) / S 
        q[2] = (R_matrix[0,2] + R_matrix[2,0]) / S 
        q[3] = (R_matrix[2,1] - R_matrix[1,2]) / S 
    elif R_matrix[1,1] > R_matrix[2,2]:
        S = torch.sqrt(1.0 + R_matrix[1,1] - R_matrix[0,0] - R_matrix[2,2] + 1e-12) * 2.0 # 添加eps防止开方负数
        q[1] = 0.25 * S 
        q[0] = (R_matrix[0,1] + R_matrix[1,0]) / S 
        q[2] = (R_matrix[1,2] + R_matrix[2,1]) / S 
        q[3] = (R_matrix[0,2] - R_matrix[2,0]) / S
    else:
        S = torch.sqrt(1.0 + R_matrix[2,2] - R_matrix[0,0] - R_matrix[1,1] + 1e-12) * 2.0 # 添加eps防止开方负数
        q[2] = 0.25 * S 
        q[0] = (R_matrix[0,2] + R_matrix[2,0]) / S
        q[1] = (R_matrix[1,2] + R_matrix[2,1]) / S
        q[3] = (R_matrix[1,0] - R_matrix[0,1]) / S 

    #* 归一化四元数
    norm_q = torch.linalg.norm(q)
    if norm_q > 1e-9:
        q = q / norm_q
    else:
        # 如果模长非常小，则返回一个标准的单位四元数 (通常表示无旋转)
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=R_matrix.dtype, device=R_matrix.device)
    return q # 返回四元数 [qx, qy, qz, qw]

#! 将PyTorch四元数转换为欧拉角 (ZYX顺序: yaw, pitch, roll)
def _quaternion_to_euler_angles_torch(q):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
 
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
   
    sinp = 2 * (qw * qy - qz * qx)
    if torch.abs(sinp) >= 1:
        pitch = torch.copysign(torch.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = torch.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([yaw, pitch, roll])

#! 计算误差向量和优化参数之间的雅可比矩阵
def compute_error_vector_jacobian(params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):

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
        T_pred_robot_base, _ = forward_kinematics_T(q_torch, params_for_fk)
        
        #* 2. 构建 T_laser_base 变换矩阵 (基座在激光坐标系下的位姿)
        R_laser_base = quaternion_to_rotation_matrix(t_laser_base_quat)
        T_laser_base_matrix = torch.eye(4, dtype=torch.float64)
        T_laser_base_matrix[0:3, 0:3] = R_laser_base
        T_laser_base_matrix[0:3, 3] = t_laser_base_pos
        
        #* 3. 将DH参数计算的位姿转换到激光跟踪仪坐标系 
        T_pred_in_laser_frame = torch.matmul(T_laser_base_matrix, T_pred_robot_base)

        
        #* 4.1 计算位置误差
        pos_pred_in_laser = T_pred_in_laser_frame[0:3, 3]
        pos_measured_in_laser = T_laser_tool_measured_torch[0:3, 3]
        pos_error = pos_pred_in_laser - pos_measured_in_laser

        #* 4.2 计算姿态误差 (使用四元数)
        R_pred = T_pred_in_laser_frame[0:3, 0:3]  # 预测的旋转矩阵
        R_meas = T_laser_tool_measured_torch[0:3, 0:3] # 测量的旋转矩阵

        q_pred = _rotation_matrix_to_quaternion_torch(R_pred) # 预测的四元数 [qx, qy, qz, qw]
        q_meas = _rotation_matrix_to_quaternion_torch(R_meas) # 测量的四元数 [qx, qy, qz, qw]

        # 计算测量四元数的共轭: q_meas_conj = [-x, -y, -z, w]
        q_meas_conj_x = -q_meas[0]
        q_meas_conj_y = -q_meas[1]
        q_meas_conj_z = -q_meas[2]
        q_meas_conj_w =  q_meas[3]

        q_err_w = q_pred[3] * q_meas_conj_w - q_pred[0] * q_meas_conj_x - q_pred[1] * q_meas_conj_y - q_pred[2] * q_meas_conj_z
        q_err_x = q_pred[3] * q_meas_conj_x + q_pred[0] * q_meas_conj_w + q_pred[1] * q_meas_conj_z - q_pred[2] * q_meas_conj_y
        q_err_y = q_pred[3] * q_meas_conj_y - q_pred[0] * q_meas_conj_z + q_pred[1] * q_meas_conj_w + q_pred[2] * q_meas_conj_x
        q_err_z = q_pred[3] * q_meas_conj_z + q_pred[0] * q_meas_conj_y - q_pred[1] * q_meas_conj_x + q_pred[2] * q_meas_conj_w
        
        # 将误差四元数转换为欧拉角误差 [yaw, pitch, roll]
        error_quaternion = torch.stack([q_err_x, q_err_y, q_err_z, q_err_w])
        # 归一化误差四元数
        norm_error_q = torch.linalg.norm(error_quaternion)
        if norm_error_q > 1e-9:
            error_quaternion_normalized = error_quaternion / norm_error_q
        else:
            error_quaternion_normalized = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=params_tensor.dtype, device=params_tensor.device)

        orient_error = _quaternion_to_euler_angles_torch(error_quaternion_normalized) 
        
        #* 5. 组合误差向量
        error_before_weight = torch.cat([pos_error, orient_error])
        
        #! 六维误差向量
        return error_before_weight * weights_torch

    #* 计算并返回雅可比矩阵 (维度应为 6x38)
    J = F.jacobian(err_vec_fn, params_torch) 
    return J

#! 测试
if __name__ == '__main__':
    
    initial_params_np = get_initial_params()
    initial_params_torch = torch.tensor(initial_params_np, dtype=torch.float64)
 
    all_joint_angles_np = load_joint_angles()
    all_T_laser_tool_measured_np = get_laser_tool_matrix()

    #* 计算雅可比矩阵
    print("--- 雅可比矩阵计算（第一组） ---")
    jacobian = compute_error_vector_jacobian(initial_params_np, all_joint_angles_np[0], all_T_laser_tool_measured_np[0], ERROR_WEIGHTS)
    save_jacobian_to_csv(jacobian)
    print("雅可比矩阵已保存（第一组）。")

    #* 预测工具位姿在激光坐标系下 和 计算并打印姿态误差
    print("\n--- 预测工具位姿在激光坐标系下并计算姿态误差（前5组关节角） ---")
    params_for_fk_torch = initial_params_torch[0:31]
    t_laser_base_pos_torch = initial_params_torch[31:34]
    t_laser_base_quat_torch = initial_params_torch[34:38]

    #* 构建 T_laser_base 变换矩阵 (基座在激光坐标系下的位姿)
    R_laser_base_torch = quaternion_to_rotation_matrix(t_laser_base_quat_torch)
    T_laser_base_matrix_torch = torch.eye(4, dtype=torch.float64)
    T_laser_base_matrix_torch[0:3, 0:3] = R_laser_base_torch
    T_laser_base_matrix_torch[0:3, 3] = t_laser_base_pos_torch

    #* 指定要测试的组的索引 (0-based)
    frames_to_test_indices = [0,1,2,3,4] 
    for i in frames_to_test_indices:
        #* 检查索引是否越界
        if i >= all_joint_angles_np.shape[0]:
            print(f"\n警告: 索引 {i+1} 超出关节角度数据范围 (共 {all_joint_angles_np.shape[0]} 组). 跳过此索引.")
            continue

        q_deg_array_np = all_joint_angles_np[i]
        q_torch = torch.as_tensor(q_deg_array_np, dtype=torch.float64)

        print(f"\n关节角度（度） {i+1}: {q_deg_array_np.tolist()}")
        T_pred_robot_base_torch, _ = forward_kinematics_T(q_torch, params_for_fk_torch)
        T_pred_in_laser_frame_torch = torch.matmul(T_laser_base_matrix_torch, T_pred_robot_base_torch)
        pose_pred_in_laser_torch = extract_pose_from_T(T_pred_in_laser_frame_torch)
        pose_pred_in_laser_np = pose_pred_in_laser_torch.detach().cpu().numpy()
        print(f"预测位姿在激光坐标系下 (x,y,z,rx,ry,rz): {pose_pred_in_laser_np.tolist()}")

        # 计算姿态误差 (将 T_meas_torch 的定义提前，确保位置和姿态误差都使用它)
        T_meas_torch = torch.as_tensor(all_T_laser_tool_measured_np[i], dtype=torch.float64)

        # 计算位置误差
        pos_pred_in_laser = T_pred_in_laser_frame_torch[0:3, 3]
        pos_measured_in_laser = T_meas_torch[0:3, 3] 
        pos_error = pos_pred_in_laser - pos_measured_in_laser
        print(f"  样本 {i+1} - 位置误差 (dx, dy, dz): [{pos_error[0]:.4f}, {pos_error[1]:.4f}, {pos_error[2]:.4f}]")
        
        R_pred = T_pred_in_laser_frame_torch[0:3, 0:3]
        R_meas = T_meas_torch[0:3, 0:3]

        q_pred = _rotation_matrix_to_quaternion_torch(R_pred)
        q_meas = _rotation_matrix_to_quaternion_torch(R_meas)

        q_meas_conj_x = -q_meas[0]
        q_meas_conj_y = -q_meas[1]
        q_meas_conj_z = -q_meas[2]
        q_meas_conj_w =  q_meas[3]

        q_err_w = q_pred[3] * q_meas_conj_w - q_pred[0] * q_meas_conj_x - q_pred[1] * q_meas_conj_y - q_pred[2] * q_meas_conj_z
        q_err_x = q_pred[3] * q_meas_conj_x + q_pred[0] * q_meas_conj_w + q_pred[1] * q_meas_conj_z - q_pred[2] * q_meas_conj_y
        q_err_y = q_pred[3] * q_meas_conj_y - q_pred[0] * q_meas_conj_z + q_pred[1] * q_meas_conj_w + q_pred[2] * q_meas_conj_x
        q_err_z = q_pred[3] * q_meas_conj_z + q_pred[0] * q_meas_conj_y - q_pred[1] * q_meas_conj_x + q_pred[2] * q_meas_conj_w
        
        error_quaternion = torch.stack([q_err_x, q_err_y, q_err_z, q_err_w])
        norm_error_q = torch.linalg.norm(error_quaternion)
        if norm_error_q > 1e-9:
            error_quaternion_normalized = error_quaternion / norm_error_q
        else:
            error_quaternion_normalized = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=initial_params_torch.dtype, device=initial_params_torch.device)

        orient_error_euler = _quaternion_to_euler_angles_torch(error_quaternion_normalized) 
        print(f"  样本 {i+1} - 姿态误差 (yaw, pitch, roll): [{orient_error_euler[0]:.4f}, {orient_error_euler[1]:.4f}, {orient_error_euler[2]:.4f}]")

   
    