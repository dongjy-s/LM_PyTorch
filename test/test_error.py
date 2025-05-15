import os
import numpy as np
import torch
import torch.autograd.functional as F
from scipy.spatial.transform import Rotation
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
np.set_printoptions(precision=8)
#! 常量定义
JOINT_ANGLE_FILE = os.path.join(PROJECT_ROOT, 'data', 'joint_angle.csv')
LASER_POS_FILE = os.path.join(PROJECT_ROOT, 'data', 'laser_pos.csv') 


#! 关节限位(度)
JOINT_LIMITS = np.array([
    [-100, 100],
    [-90, 100],
    [-100, 600],
    [-100, 100],
    [-90, 90],
    [-120, 120]
])

# 构建到优化参数结果文件的绝对路径
OPTIMIZED_DH_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'results', 'optimized_dh_parameters.csv')
OPTIMIZED_TCP_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'results', 'optimized_tcp_parameters.csv')
OPTIMIZED_T_LASER_BASE_PARAMS_FILE = os.path.join(PROJECT_ROOT, 'results', 'optimized_t_laser_base_parameters.csv')

#! 雷达数据 (顺序: theta_offset, alpha, d, a) 
LASER_DH_PARAMS = [
    -0.0063, 0, 487.4009, 0,                
    -90.2267, -90, 0, 85.5599,            
    -0.4689, 0, 0, 639.8143,              
    0.5631, -90, 720.3035, 205.288,         
    0.0723, 90, 0, 0,                     
    179.6983, -90, 75.7785, 0             
]
LASER_TCP_OFFSET_POSITION = [0.1731, 1.1801, 238.3535]
LASER_TCP_OFFSET_QUATERNION = [0.4961, 0.5031, 0.505, 0.4957]
LASER_BASE_POSITION = [3610.8319, 3300.7233, 13.6472]
LASER_BASE_QUATERNION = [0.0014, -0.0055, 0.7873, -0.6166]


#! 从CSV加载优化后的DH参数 (CSV列顺序: theta_offset,alpha,d,a)
def load_optimized_dh_params(filepath):
    df = pd.read_csv(filepath)
    dh_params = df[['theta_offset', 'alpha', 'd', 'a']].values.flatten()
    return dh_params

#! 从CSV加载优化后的TCP参数
def load_optimized_tcp_params(filepath):
    df = pd.read_csv(filepath, index_col=0)
    position = df.loc[['tx', 'ty', 'tz'], 'value'].values
    quaternion = df.loc[['qx', 'qy', 'qz', 'qw'], 'value'].values
    return position, quaternion

#! 从CSV加载优化后的T_laser_base参数
def load_optimized_t_laser_base_params(filepath):
    df = pd.read_csv(filepath, index_col=0)
    params = df['value'].values
    return params

#! 激光跟踪仪工具位姿变换矩阵 
def get_laser_tool_matrix():
    laser_data = pd.read_csv(LASER_POS_FILE, delimiter=',', skiprows=1, header=None).values
    num_samples = laser_data.shape[0]
    laser_tool_matrix = np.zeros((num_samples, 4, 4))
    for i, data in enumerate(laser_data):
        x, y, z, rx, ry, rz = data
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]
        laser_tool_matrix[i] = T
    return laser_tool_matrix

#! 构建MDH变换矩阵
def modified_dh_matrix(theta_val_rad, alpha_val_rad, d_val, a_val):
    cos_theta = torch.cos(theta_val_rad)
    sin_theta = torch.sin(theta_val_rad)
    cos_alpha = torch.cos(alpha_val_rad)
    sin_alpha = torch.sin(alpha_val_rad)
    
    MDH_matrix = torch.zeros(4, 4, dtype=torch.float64)
    MDH_matrix[0, 0] = cos_theta
    MDH_matrix[0, 1] = -sin_theta
    MDH_matrix[0, 3] = a_val
    
    MDH_matrix[1, 0] = sin_theta * cos_alpha
    MDH_matrix[1, 1] = cos_theta * cos_alpha
    MDH_matrix[1, 2] = -sin_alpha
    MDH_matrix[1, 3] = -sin_alpha * d_val
    
    MDH_matrix[2, 0] = sin_theta * sin_alpha
    MDH_matrix[2, 1] = cos_theta * sin_alpha
    MDH_matrix[2, 2] = cos_alpha
    MDH_matrix[2, 3] = cos_alpha * d_val
    
    MDH_matrix[3, 3] = 1.0 # 确保为浮点数，尽管dtype已指定
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

#! 正向运动学
def forward_kinematics_T(q_deg_array, params_torch):
    q_deg_array = torch.as_tensor(q_deg_array, dtype=torch.float64)
    dh_params = params_torch[0:24]
    tool_offset_position = params_torch[24:27]
    tool_offset_quaternion = params_torch[27:31]

    q_list = q_deg_array.detach().cpu().numpy().tolist()
    for idx, q_val in enumerate(q_list):
        min_lim, max_lim = JOINT_LIMITS[idx]
        if not (min_lim <= q_val <= max_lim):
            raise ValueError(f"关节{idx+1}角度 {q_val}° 超出限位范围 [{min_lim}°, {max_lim}°]")

    T_total = torch.eye(4, dtype=torch.float64)
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
        T_total = T_total @ A_i

    T_flange_tool = torch.eye(4, dtype=torch.float64)
    T_flange_tool[0:3, 3] = tool_offset_position
    R_tool = quaternion_to_rotation_matrix(tool_offset_quaternion)
    T_flange_tool[0:3, 0:3] = R_tool
    
    T_base_tool = T_total @ T_flange_tool
    return T_base_tool

#! 从变换矩阵提取6维位姿向量
def extract_pose_from_T(T):
    position = T[0:3, 3]
    R = T[0:3, 0:3]
   
    sy = torch.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-8 # 使用epsilon比较

    if not singular:
        x = torch.atan2(R[2,1], R[2,2])
        y = torch.atan2(-R[2,0], sy)
        z = torch.atan2(R[1,0], R[0,0])
    else: # 奇异情况
        x = torch.atan2(-R[1,2], R[1,1])
        y = torch.atan2(-R[2,0], sy)
        z = torch.tensor(0.0, dtype=T.dtype) # 将z设为0，避免万向节死锁问题中的不确定性
        
    rx = torch.rad2deg(x)
    ry = torch.rad2deg(y)
    rz = torch.rad2deg(z)
    euler_angles_torch = torch.stack([rx, ry, rz])
        
    return torch.cat([position, euler_angles_torch])

#! 计算预测位姿与测量位姿之间的平均误差
def compute_average_pose_errors(joint_angles_all_frames_np, 
                                fk_params_torch, 
                                T_laser_base_matrix_torch, 
                                measured_T_matrices_all_frames_np, 
                                frames_to_process_indices):
    position_error_magnitudes = []
    orientation_error_magnitudes = []

    if not frames_to_process_indices:
        return 0.0, 0.0

    for frame_idx in frames_to_process_indices:
        if frame_idx >= joint_angles_all_frames_np.shape[0] or frame_idx >= measured_T_matrices_all_frames_np.shape[0]:
            print(f"警告: 帧索引 {frame_idx} 超出数据范围，跳过误差计算。")
            continue
        current_joint_angles_torch = torch.as_tensor(joint_angles_all_frames_np[frame_idx], dtype=torch.float64)
        T_pred_robot_base_torch = forward_kinematics_T(current_joint_angles_torch, fk_params_torch)
        T_pred_in_laser_torch = torch.matmul(T_laser_base_matrix_torch, T_pred_robot_base_torch)
        pose_pred_np = extract_pose_from_T(T_pred_in_laser_torch).detach().cpu().numpy()
        T_measured_this_frame_torch = torch.as_tensor(measured_T_matrices_all_frames_np[frame_idx], dtype=torch.float64)
        pose_measured_np = extract_pose_from_T(T_measured_this_frame_torch).detach().cpu().numpy()
        pos_error_vector = pose_pred_np[:3] - pose_measured_np[:3]
        ori_errors_normalized = []
        for i in range(3):
            diff = pose_pred_np[3+i] - pose_measured_np[3+i]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            ori_errors_normalized.append(diff)
        ori_error_vector_normalized = np.array(ori_errors_normalized)
        position_error_magnitudes.append(np.linalg.norm(pos_error_vector))
        orientation_error_magnitudes.append(np.linalg.norm(ori_error_vector_normalized)) # 使用归一化后的误差向量
    avg_position_error = np.mean(position_error_magnitudes) if position_error_magnitudes else 0.0
    avg_orientation_error = np.mean(orientation_error_magnitudes) if orientation_error_magnitudes else 0.0
    return avg_position_error, avg_orientation_error

def perform_kinematics_analysis_and_print_results(use_optimized_csv_data: bool):
    if use_optimized_csv_data:
        print("--- 尝试从CSV文件加载优化参数进行计算 ---")
        try:
            current_dh_params = load_optimized_dh_params(OPTIMIZED_DH_PARAMS_FILE)
            current_tool_offset_position, current_tool_offset_quaternion = load_optimized_tcp_params(OPTIMIZED_TCP_PARAMS_FILE)
            temp_optimized_t_laser_base_params = load_optimized_t_laser_base_params(OPTIMIZED_T_LASER_BASE_PARAMS_FILE)
            # current_t_laser_base_params 需要是7个元素的数组 [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
            # load_optimized_t_laser_base_params 返回的就是这个格式
            current_t_laser_base_params = temp_optimized_t_laser_base_params
            print("成功从CSV文件加载优化后的参数。")
            param_source_name = "优化后的参数 (来自CSV)"
        except FileNotFoundError as e:
            print(f"错误: 找不到优化参数文件: {e.filename}")
            print("请确保以下文件存在于项目根目录下的 'results' 文件夹中:")
            print(f"- {os.path.basename(OPTIMIZED_DH_PARAMS_FILE)}")
            print(f"- {os.path.basename(OPTIMIZED_TCP_PARAMS_FILE)}")
            print(f"- {os.path.basename(OPTIMIZED_T_LASER_BASE_PARAMS_FILE)}")
            print("程序将退出。")
            exit(1)
        except Exception as e:
            print(f"从CSV加载优化参数时发生错误: {e}")
            print("程序将退出。")
            exit(1)
    else:
        print("--- 使用预定义的参考/雷达参数进行计算 ---")
        current_dh_params = np.array(LASER_DH_PARAMS)
        current_tool_offset_position = np.array(LASER_TCP_OFFSET_POSITION)
        current_tool_offset_quaternion = np.array(LASER_TCP_OFFSET_QUATERNION)
        current_t_laser_base_params = np.concatenate([
            np.array(LASER_BASE_POSITION),
            np.array(LASER_BASE_QUATERNION)
        ])
        param_source_name = "参考/雷达参数 (预定义)"

    # 打印正在使用的参数
    print(f"\n--- 当前使用的 {param_source_name} ---")
    print(f"DH 参数 (theta_offset, alpha, d, a) x {current_dh_params.shape[0]//4}:")
    for i in range(0, len(current_dh_params), 4):
        print(f"  关节 {i//4 + 1}: {current_dh_params[i:i+4].tolist()}")
    
    print(f"\n工具TCP偏移 - 位置 (tx, ty, tz): {current_tool_offset_position.tolist()}")
    print(f"工具TCP偏移 - 四元数 (qx, qy, qz, qw): {current_tool_offset_quaternion.tolist()}")
    
    base_pos = current_t_laser_base_params[0:3]
    base_quat = current_t_laser_base_params[3:7]
    print(f"\n基座在激光跟踪仪坐标系下的 - 位置 (x, y, z): {base_pos.tolist()}")
    print(f"基座在激光跟踪仪坐标系下的 - 四元数 (qx, qy, qz, qw): {base_quat.tolist()}")
    print("------------------------\n")

    combined_params_np = np.concatenate((
        current_dh_params, 
        current_tool_offset_position, 
        current_tool_offset_quaternion,
        current_t_laser_base_params
    ))
    combined_params_torch = torch.tensor(combined_params_np, dtype=torch.float64)

    all_joint_angles_np = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    all_T_laser_tool_measured_np = get_laser_tool_matrix()

    print(f"\n--- 使用 {param_source_name} 预测工具位姿在激光坐标系下 ---")

    params_for_fk_torch = combined_params_torch[0:24 + 3 + 4]
    t_laser_base_pos_torch = combined_params_torch[24+3+4 : 24+3+4+3]
    t_laser_base_quat_torch = combined_params_torch[24+3+4+3 : 24+3+4+3+4]

    R_laser_base_torch = quaternion_to_rotation_matrix(t_laser_base_quat_torch)
    T_laser_base_matrix_torch = torch.eye(4, dtype=torch.float64)
    T_laser_base_matrix_torch[0:3, 0:3] = R_laser_base_torch
    T_laser_base_matrix_torch[0:3, 3] = t_laser_base_pos_torch

    num_total_frames = all_joint_angles_np.shape[0]
    if all_T_laser_tool_measured_np.shape[0] < num_total_frames:
        print(f"警告: 激光测量数据帧数 ({all_T_laser_tool_measured_np.shape[0]}) 少于关节角度帧数 ({num_total_frames}).")
        print(f"将仅处理 {all_T_laser_tool_measured_np.shape[0]} 帧数据.")
        num_total_frames = all_T_laser_tool_measured_np.shape[0]
        
    frames_to_test_indices = list(range(num_total_frames))
    
    for frame_idx in frames_to_test_indices:
        if frame_idx >= all_joint_angles_np.shape[0]:
            print(f"\n警告: 索引 {frame_idx+1} 超出关节角度数据范围 (共 {all_joint_angles_np.shape[0]} 组). 跳过此索引.")
            continue

        current_joint_angles_np = all_joint_angles_np[frame_idx]
        current_joint_angles_torch = torch.as_tensor(current_joint_angles_np, dtype=torch.float64)

        print(f"\n--- 第 {frame_idx+1} 帧 --- ")
        print(f"关节角度 (度): {current_joint_angles_np.tolist()}")

        T_pred_robot_base_torch = forward_kinematics_T(current_joint_angles_torch, params_for_fk_torch)
        T_pred_in_laser_torch = torch.matmul(T_laser_base_matrix_torch, T_pred_robot_base_torch)
        pose_pred_in_laser_torch = extract_pose_from_T(T_pred_in_laser_torch)
        pose_pred_in_laser_np = pose_pred_in_laser_torch.detach().cpu().numpy()
        formatted_pose = [f"{val:.4f}" for val in pose_pred_in_laser_np]
        print(f"预测位姿在激光坐标系下 (x,y,z,rx,ry,rz): {formatted_pose}")
    
    if frames_to_test_indices:
        avg_pos_err, avg_ori_err = compute_average_pose_errors(
            all_joint_angles_np, 
            params_for_fk_torch, 
            T_laser_base_matrix_torch, 
            all_T_laser_tool_measured_np,
            frames_to_test_indices
        )
        print(f"\n--- 平均误差评估 (基于全部 {len(frames_to_test_indices)} 帧, 使用 {param_source_name}) ---")
        print(f"平均位置误差 (mm): {avg_pos_err:.4f}")
        print(f"平均姿态误差 (度): {avg_ori_err:.4f}")
    else:
        print("\n没有可用的帧用于计算平均误差。")

if __name__ == '__main__':

    # True  - 从CSV文件加载优化后的参数
    # False - 使用脚本中预定义的 LASER_... 参数 (参考/雷达数据)
    use_optimized_data_source = True

    perform_kinematics_analysis_and_print_results(use_optimized_csv_data=use_optimized_data_source)

   
    