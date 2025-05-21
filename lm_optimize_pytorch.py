import os
import numpy as np
import torch
from jacobian_torch import (
    compute_error_vector_jacobian, 
    forward_kinematics_T, 
    get_laser_tool_matrix,
    ERROR_WEIGHTS,
    quaternion_to_rotation_matrix,
    INIT_T_LASER_BASE_PARAMS,
    INIT_DH_PARAMS,
    JOINT_ANGLE_FILE,
    INIT_TOOL_OFFSET_POSITION,
    INIT_TOOL_OFFSET_QUATERNION
)

#* 固定的参数索引(为空则是全优化)
ALL_FIXED_INDICES = [] 

#! 将旋转矩阵转换为四元数
def _rotation_matrix_to_quaternion(R_matrix):
    if not torch.is_tensor(R_matrix):
        R_matrix = torch.as_tensor(R_matrix, dtype=torch.float64)
    q = torch.zeros(4, dtype=R_matrix.dtype, device=R_matrix.device)
    trace = R_matrix[0,0] + R_matrix[1,1] + R_matrix[2,2]   
    if trace > 1e-8: 
        S = torch.sqrt(trace + 1.0) * 2.0 
        q[3] = 0.25 * S  
        q[0] = (R_matrix[2,1] - R_matrix[1,2]) / S 
        q[1] = (R_matrix[0,2] - R_matrix[2,0]) / S 
        q[2] = (R_matrix[1,0] - R_matrix[0,1]) / S 
    elif (R_matrix[0,0] > R_matrix[1,1]) and (R_matrix[0,0] > R_matrix[2,2]):
        S = torch.sqrt(1.0 + R_matrix[0,0] - R_matrix[1,1] - R_matrix[2,2] + 1e-12) * 2.0 
        q[0] = 0.25 * S # qx
        q[1] = (R_matrix[0,1] + R_matrix[1,0]) / S 
        q[2] = (R_matrix[0,2] + R_matrix[2,0]) / S 
    elif R_matrix[1,1] > R_matrix[2,2]:
        S = torch.sqrt(1.0 + R_matrix[1,1] - R_matrix[0,0] - R_matrix[2,2] + 1e-12) * 2.0 
        q[3] = (R_matrix[0,2] - R_matrix[2,0]) / S 
        q[0] = (R_matrix[0,1] + R_matrix[1,0]) / S 
        q[1] = 0.25 * S 
        q[2] = (R_matrix[1,2] + R_matrix[2,1]) / S 
    else:
        S = torch.sqrt(1.0 + R_matrix[2,2] - R_matrix[0,0] - R_matrix[1,1] + 1e-12) * 2.0 
        q[3] = (R_matrix[1,0] - R_matrix[0,1]) / S 
        q[0] = (R_matrix[0,2] + R_matrix[2,0]) / S 
        q[1] = (R_matrix[1,2] + R_matrix[2,1]) / S 
        q[2] = 0.25 * S 

    #* 归一化四元数
    norm_q = torch.linalg.norm(q)
    if norm_q > 1e-9: 
        q = q / norm_q
    else: 
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=R_matrix.dtype, device=R_matrix.device)
    return q 

#! 计算单组数据的误差向量（加权重）
def compute_error_vector(params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    #* 获取关节角度和参数
    q_t = torch.as_tensor(joint_angles, dtype=torch.float64)
    params_t = torch.as_tensor(params, dtype=torch.float64) 

    #* 提取参数
    params_for_fk = params_t[0:31] 
    t_laser_base_pos = params_t[31:34]
    t_laser_base_quat = params_t[34:38] 

  
    T_pred_robot_base = forward_kinematics_T(q_t, params_for_fk) 

    #* 构建T_laser_base变换矩阵（基座在激光坐标系下的位姿）
    R_laser_base = quaternion_to_rotation_matrix(t_laser_base_quat) 
    T_laser_base_matrix = torch.eye(4, dtype=torch.float64)
    T_laser_base_matrix[0:3, 0:3] = R_laser_base
    T_laser_base_matrix[0:3, 3] = t_laser_base_pos
    
    #* 将机器人预测位姿转换到激光跟踪仪坐标系
    T_pred_in_laser_frame = torch.matmul(T_laser_base_matrix, T_pred_robot_base)
    
    #* 计算激光跟踪仪-基座变换矩阵
    pred_pos = T_pred_in_laser_frame[0:3, 3]
    pred_R = T_pred_in_laser_frame[0:3, 0:3]

    #* 提取测量位置和旋转矩阵
    T_laser_t = torch.as_tensor(laser_matrix, dtype=torch.float64)
    meas_pos = T_laser_t[0:3, 3]
    meas_R = T_laser_t[0:3, 0:3]

    #* 位置误差
    pos_error = pred_pos - meas_pos

    #* 将旋转矩阵转换为四元数 [qx, qy, qz, qw]
    q_pred = _rotation_matrix_to_quaternion(pred_R) 
    q_meas = _rotation_matrix_to_quaternion(meas_R) 

    #* 计算四元数共轭
    q_meas_conj_x = -q_meas[0]
    q_meas_conj_y = -q_meas[1]
    q_meas_conj_z = -q_meas[2]
    q_meas_conj_w =  q_meas[3]
    
    # 四元数乘法: q_pred * q_meas_conj
    # q_pred = [x1, y1, z1, w1], q_meas_conj = [x2, y2, z2, w2]
    # qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
    # qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
    # qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
    # qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
    #* 计算预测和实际之间的误差旋转
    q_err_w = q_pred[3] * q_meas_conj_w - q_pred[0] * q_meas_conj_x - q_pred[1] * q_meas_conj_y - q_pred[2] * q_meas_conj_z
    q_err_x = q_pred[3] * q_meas_conj_x + q_pred[0] * q_meas_conj_w + q_pred[1] * q_meas_conj_z - q_pred[2] * q_meas_conj_y
    q_err_y = q_pred[3] * q_meas_conj_y - q_pred[0] * q_meas_conj_z + q_pred[1] * q_meas_conj_w + q_pred[2] * q_meas_conj_x
    q_err_z = q_pred[3] * q_meas_conj_z + q_pred[0] * q_meas_conj_y - q_pred[1] * q_meas_conj_x + q_pred[2] * q_meas_conj_w

    #* 计算误差旋转
    orient_error = torch.stack([q_err_x, q_err_y, q_err_z])
    combined_error = torch.cat((pos_error, orient_error))
    return combined_error * torch.as_tensor(weights, dtype=torch.float64)


#! 计算所有样本的均方根误差
def compute_total_error_avg(params, joint_angles, laser_matrices, weights=ERROR_WEIGHTS):
    total_error_sum_sq = 0.0 
    n_samples = len(joint_angles)
    if n_samples == 0:
        return torch.tensor(0.0, dtype=torch.float64) 

    for i in range(n_samples):
        error_vec = compute_error_vector(params, joint_angles[i], laser_matrices[i], weights) 
        total_error_sum_sq += torch.sum(error_vec**2)
    
    #* 返回均方根误差 (RMSE误差) 公式：RMSE = sqrt(sum(error_vec^2) / n_samples)
    mean_squared_error = total_error_sum_sq / n_samples
    return torch.sqrt(mean_squared_error)

#! 保存优化后的DH参数和TCP参数
def save_optimization_results(params, filepath_prefix='results/optimized'):
    dirpath = os.path.dirname(filepath_prefix)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dh_params = params[0:24]
    tcp_params = params[24:31]
    t_laser_base_params = params[31:38]
    dh_filepath = f"{filepath_prefix}_dh_parameters.csv"
    dh_matrix = np.array(dh_params).reshape(6, 4)
    header_dh = "alpha,a,d,theta_offset"
    row_labels_dh = [f"Joint_{i+1}" for i in range(6)]
    with open(dh_filepath, 'w') as f:
        f.write(f",{header_dh}\n")  
        for i, row in enumerate(dh_matrix):
            f.write(f"{row_labels_dh[i]},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}\n")
    print(f"优化后的DH参数已保存到: {dh_filepath}")
    
    # 保存TCP参数
    tcp_filepath = f"{filepath_prefix}_tcp_parameters.csv"
    header_tcp = "parameter,value"
    tcp_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with open(tcp_filepath, 'w') as f:
        f.write(f"{header_tcp}\n")
        for name, value in zip(tcp_param_names, tcp_params):
            f.write(f"{name},{value:.6f}\n")
    print(f"优化后的TCP参数已保存到: {tcp_filepath}")
    
    # 保存激光跟踪仪-基座变换参数
    t_laser_base_filepath = f"{filepath_prefix}_t_laser_base_parameters.csv"
    header_t_laser_base = "parameter,value"
    t_laser_base_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with open(t_laser_base_filepath, 'w') as f:
        f.write(f"{header_t_laser_base}\n")
        for name, value in zip(t_laser_base_param_names, t_laser_base_params):
            f.write(f"{name},{value:.6f}\n")
    print(f"优化后的激光跟踪仪-基座变换参数已保存到: {t_laser_base_filepath}")


#! LM优化
def optimize_dh_parameters(initial_params, max_iterations=50, lambda_init=0.01, tol=1e-10, opt_indices=None):
    params = torch.tensor(initial_params, dtype=torch.float64, requires_grad=False)
    #* 初始化阻尼因子
    lambda_val = lambda_init
    #* 读取关节角度和激光数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    n_samples = len(joint_angles)
    if n_samples == 0:
        print("错误: 无法加载关节角度或激光数据，样本数量为0。")
        return initial_params 
    
    #* 记录初始均方根误差
    current_avg_error = compute_total_error_avg(params, joint_angles, laser_matrices) 
    print(f"初始均方根误差：{current_avg_error.item():.6f}")
    avg_initial_error = current_avg_error.item() 
    
    #* 处理可优化参数索引
    if opt_indices is None:
        opt_indices = list(range(len(initial_params)))
    opt_indices = np.array(opt_indices)
    
    #* LM迭代
    for iteration in range(max_iterations):
        all_errors = []
        all_jacobians = []

        #* 计算所有样本的误差向量和雅可比矩阵
        for i in range(n_samples):
            error_vec = compute_error_vector(params, joint_angles[i], laser_matrices[i])
            jacobian = compute_error_vector_jacobian(params.numpy(), joint_angles[i], laser_matrices[i])
            all_errors.append(error_vec)
            all_jacobians.append(jacobian)
        error_vector = torch.cat(all_errors)

        #* 将所有雅可比矩阵堆叠成一个矩阵
        J = torch.vstack(all_jacobians)

        #* LM算法公式：(J^T J + λ * diag(J^T J)) * Δθ = -J^T e
        J_opt = J[:, opt_indices]
        JTJ = torch.matmul(J_opt.transpose(0, 1), J_opt)
        JTe = torch.matmul(J_opt.transpose(0, 1), error_vector)
        update_success = False
        inner_iterations = 0
        max_inner_iterations = 10
        while not update_success and inner_iterations < max_inner_iterations:
            inner_iterations += 1
            
            #! 添加阻尼项
            H = JTJ + lambda_val * torch.diag(torch.diag(JTJ))
            
            #! 计算更新量
            try:
                delta = -torch.linalg.solve(H, JTe)
            except:
                print(f"矩阵求解错误，增大阻尼因子 λ = {lambda_val} -> {lambda_val * 10}")
                lambda_val *= 10
                if lambda_val > 1e8:
                    print(f"阻尼因子超过阈值，提前结束优化")
                    return params.numpy()
                continue

            #* 尝试更新
            params_new = params.clone()
            params_new[opt_indices] += delta
            
            #* 计算新误差
            new_avg_error = compute_total_error_avg(params_new, joint_angles, laser_matrices)
            
            #* 如果新的均方误差小于当前均方误差，则接受更新
            if new_avg_error < current_avg_error:
                params = params_new
                current_avg_error = new_avg_error
                lambda_val = max(lambda_val / 10, 1e-7)
                update_success = True

                #* TCP四元数归一化 
                if any(idx in opt_indices for idx in range(27, 31)):
                    q_tcp = params[27:31] 
                    norm_q_tcp = torch.linalg.norm(q_tcp)
                    if norm_q_tcp > 1e-9: 
                        params[27:31] = q_tcp / norm_q_tcp
                    else:
                        params[27:31] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=params.dtype, device=params.device)
                        print("警告: TCP四元数模长接近于零，已重置为[0,0,0,1]")
                
                #* 激光跟踪仪-基座四元数归一化 
                if any(idx in opt_indices for idx in range(34, 38)):
                    q_laser_base = params[34:38]
                    norm_q_laser_base = torch.linalg.norm(q_laser_base)
                    if norm_q_laser_base > 1e-9:
                        params[34:38] = q_laser_base / norm_q_laser_base
                    else:
                        params[34:38] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=params.dtype, device=params.device)
                        print("警告: 激光跟踪仪-基座四元数模长接近于零，已重置为[0,0,0,1]")

                print(f"迭代 {iteration+1}: 均方根误差 = {current_avg_error.item():.8f}, λ = {lambda_val:.4e}, \nΔθ (参数改变量) = {delta.numpy()}")            
            else:
                lambda_val *= 10
                print(f"拒绝更新，增大阻尼因子 λ = {lambda_val}")

                #* 如果阻尼因子超过一定阈值，提前结束优化
                if lambda_val > 1e5:
                    print(f"阻尼因子超过阈值 1e5，提前结束优化")
                    return params.numpy()
        if not update_success:
            print("内部迭代未收敛，继续主循环")
  
        if update_success and torch.norm(delta) < tol:
            print(f"参数变化小于阈值 {tol}，在第 {iteration+1} 次迭代后收敛")
            break

    #* 最终均方根误差
    final_avg_error = current_avg_error.item() 
    improvement = (avg_initial_error - final_avg_error) / avg_initial_error * 100 if avg_initial_error > 1e-9 else 0 
    print(f"优化完成，初始均方根误差: {avg_initial_error:.6f}, 最终均方根误差: {final_avg_error:.6f}, 改进率: {improvement:.2f}%")
    return params.numpy()

#! 交替优化函数
def alternate_optimize_parameters(initial_params, max_alt_iterations=10, convergence_tol=1e-5, 
                                 max_sub_iterations=30, lambda_init_group1=0.01, lambda_init_group2=0.001):
    print("\n" + "="*60)
    print(" "*20 + "开始交替优化")
    print("="*60)
    
    #* 读取关节角度和激光数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    
    #! 定义两组参数索引
    #* 第一组：DH参数 + 工具TCP + 激光跟踪仪XYZ
    all_indices_group1 = list(range(0, 34))  
    #* 第二组：激光跟踪仪四元数
    all_indices_group2 = list(range(34, 38))  
   
    opt_indices_group1 = [idx for idx in all_indices_group1 if idx not in ALL_FIXED_INDICES]
    opt_indices_group2 = [idx for idx in all_indices_group2 if idx not in ALL_FIXED_INDICES]
    
    #! 初始化均方根误差
    params = np.array(initial_params)
    current_avg_error_val = compute_total_error_avg(params, joint_angles, laser_matrices).item() # 已修改为计算平均误差, 使用 .item() 获取数值
    avg_initial_error_alternate = current_avg_error_val 
    print(f"初始均方根误差: {avg_initial_error_alternate:.6f}")
    
    #* 打印参数组信息
    print(f"第一组参数索引 (共{len(opt_indices_group1)}个): {opt_indices_group1}")
    print(f"第二组参数索引 (共{len(opt_indices_group2)}个): {opt_indices_group2}")
    
    # 记录误差历史
    error_history_avg = [avg_initial_error_alternate] # 记录平均误差, 使用新的初始误差变量名
    
    #! 交替优化主循环
    for alt_iteration in range(max_alt_iterations):
        print(f"\n===== 交替优化循环 {alt_iteration + 1}/{max_alt_iterations} =====")
        
        #! 第一步：优化DH参数 + TCP + 激光XYZ，固定激光四元数
        print("\n----- 第一步：优化DH参数 + TCP + 激光XYZ -----")
        params_step1 = optimize_dh_parameters(
            params, 
            max_iterations=max_sub_iterations, 
            lambda_init=lambda_init_group1, 
            opt_indices=opt_indices_group1
        )
        
        #* 计算第一步优化后的误差    
        avg_error_step1 = compute_total_error_avg(params_step1, joint_angles, laser_matrices).item() 
        print(f"第一步后均方根误差: {avg_error_step1:.6f}")
        
        #! 第二步：优化激光四元数，固定DH参数+TCP+激光XYZ
        print("\n----- 第二步：优化激光四元数 -----")
        params_step2 = optimize_dh_parameters(
            params_step1, 
            max_iterations=max_sub_iterations, 
            lambda_init=lambda_init_group2, 
            opt_indices=opt_indices_group2
        )
        
        #* 计算第二步优化后的误差
        avg_error_step2 = compute_total_error_avg(params_step2, joint_angles, laser_matrices).item() 
        print(f"第二步后均方根误差: {avg_error_step2:.6f}")
        
        #* 更新误差
        params = params_step2
        error_history_avg.append(avg_error_step2) 
        
        #* 计算误差改进量
        error_improvement = error_history_avg[-2] - error_history_avg[-1] 
        relative_improvement = error_improvement / error_history_avg[-2] if error_history_avg[-2] > 1e-9 else 0
        
        print(f"\n本次循环误差改进: {error_improvement:.6f}, 相对改进: {relative_improvement*100:.4f}%")
        
        if error_improvement < convergence_tol:
            print(f"\n误差改进 {error_improvement:.6f} 小于阈值 {convergence_tol}，交替优化收敛")
            break
            
    #* 计算总体优化效果
    final_avg_error_alternate = error_history_avg[-1] 
    total_improvement = (avg_initial_error_alternate - final_avg_error_alternate) / avg_initial_error_alternate * 100 if avg_initial_error_alternate > 1e-9 else 0
    
    print("\n" + "="*60)
    print(f"交替优化完成，共进行了 {alt_iteration + 1} 次循环")
    print(f"初始均方根误差: {avg_initial_error_alternate:.6f}") 
    print(f"最终均方根误差: {final_avg_error_alternate:.6f}")
    print(f"总体改进率: {total_improvement:.2f}%")
    print("="*60)
    
    return params

def evaluate_optimization(initial_params, optimized_params):
    """评估优化效果，报告与优化器目标一致的均方根误差"""
    # 读取数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    n_samples = len(joint_angles)

    if n_samples == 0:
        print("评估警告: 样本数量为0，无法进行评估。")
        return
    
    print("\n" + "="*60) # 调整分隔线长度以适应新的表头 
    print(" "*15 + "优化效果评估 (所有分量的均方根误差)") # 修改标题
    print("="*60)
    # 打印表头，明确指出总体平均误差是均方根误差
    print(f"{'姿态(帧)':^12}|{'初始单帧范数':^18}|{'优化后单帧范数':^20}|{'单帧改进率':^15}")
    print("-"*68) # 调整分隔线长度
    
    # 计算初始和优化后的总体均方根误差 (与compute_total_error_avg一致)
    initial_total_rmse = compute_total_error_avg(initial_params, joint_angles, laser_matrices).item()
    optimized_total_rmse = compute_total_error_avg(optimized_params, joint_angles, laser_matrices).item()

    # 逐帧显示误差范数及其改进，用于详细分析
    for i in range(n_samples):
        initial_error_vec = compute_error_vector(initial_params, joint_angles[i], laser_matrices[i])
        optimized_error_vec = compute_error_vector(optimized_params, joint_angles[i], laser_matrices[i])
        
        initial_frame_norm = torch.linalg.norm(initial_error_vec).item()
        optimized_frame_norm = torch.linalg.norm(optimized_error_vec).item()
        
        frame_improvement = (1 - optimized_frame_norm / initial_frame_norm) * 100 if initial_frame_norm > 1e-9 else 0
        
        print(f"{i+1:^12}|{initial_frame_norm:^18.6f}|{optimized_frame_norm:^20.6f}|{frame_improvement:^14.2f}%")
    
    # 计算总体改进率 (基于均方根误差)
    avg_improvement_rmse = (1 - optimized_total_rmse / initial_total_rmse) * 100 if initial_total_rmse > 1e-9 else 0
    
    print("-"*68) # 调整分隔线长度
    print(f"{'总体平均RMSE':^12}|{initial_total_rmse:^18.6f}|{optimized_total_rmse:^20.6f}|{avg_improvement_rmse:^14.2f}%")
    print("="*60)


if __name__ == '__main__':
    initial_dh_params = np.array(INIT_DH_PARAMS)
    initial_tcp_params = np.concatenate((INIT_TOOL_OFFSET_POSITION, INIT_TOOL_OFFSET_QUATERNION))
    initial_params = np.concatenate((initial_dh_params, initial_tcp_params, INIT_T_LASER_BASE_PARAMS)) 

    # 排除固定参数
    opt_indices = [i for i in range(38) if i not in ALL_FIXED_INDICES]
    print(f"固定参数索引 ({len(ALL_FIXED_INDICES)}): {ALL_FIXED_INDICES}")
    print(f"可优化参数索引 ({len(opt_indices)}): {opt_indices}")
    
    # 使用交替优化方法
    optimized_params = alternate_optimize_parameters(
        initial_params, 
        max_alt_iterations=3,      # 最大交替迭代次数
        convergence_tol=1e-6,      # 收敛阈值
        max_sub_iterations=10,     # 每次子优化的最大迭代次数
        lambda_init_group1=0.01,   # 第一组参数初始阻尼因子
        lambda_init_group2=0.01   # 第二组参数初始阻尼因子
    )

    # 保存优化结果 
    save_optimization_results(optimized_params) 

    # 评估优化效果 
    evaluate_optimization(initial_params, optimized_params)
    
    # 输出优化前后的参数对比
    print("\n" + "="*70)
    print(" "*25 + "DH参数对比")
    print("="*70)
    print(f"{'关节':^6}|{'参数':^12}|{'初始值':^15}|{'优化值':^15}|{'差异':^15}|{'状态':^10}")
    print("-"*70)
    
    param_names = ["alpha", "a", "d", "theta_offset"]
    
    # 将参数重构为6×4矩阵，方便查看 (DH部分)
    init_dh_matrix = initial_params[0:24].reshape(6, 4)
    opt_dh_matrix = optimized_params[0:24].reshape(6, 4)
    
    for i in range(6):  
        for j in range(4):  
            param_idx = i * 4 + j  
            param_diff = opt_dh_matrix[i, j] - init_dh_matrix[i, j]
            status = "已优化" if param_idx in opt_indices else "已固定"
            print(f"{i+1:^6}|{param_names[j]:^12}|{init_dh_matrix[i, j]:^15.4f}|{opt_dh_matrix[i, j]:^15.4f}|{param_diff:^15.4f}|{status:^10}")
        if i < 5:  
            print("-"*70)
    
    # 添加TCP参数对比
    print("="*70)
    print(" "*25 + "TCP 参数对比")
    print("="*70)
    tcp_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    init_tcp_params = initial_params[24:31]
    opt_tcp_params = optimized_params[24:31]
    for k in range(7):
        tcp_idx = 24 + k
        tcp_diff = opt_tcp_params[k] - init_tcp_params[k]
        status = "已优化" if tcp_idx in opt_indices else "已固定"
        print(f"{'-':^6}|{tcp_param_names[k]:^12}|{init_tcp_params[k]:^15.4f}|{opt_tcp_params[k]:^15.4f}|{tcp_diff:^15.4f}|{status:^10}")

    # 添加激光跟踪仪-基座变换参数对比
    print("="*70)
    print(" "*25 + "激光跟踪仪-基座变换参数对比")
    print("="*70)
    t_laser_base_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    init_t_laser_base_params = initial_params[31:38]
    opt_t_laser_base_params = optimized_params[31:38]
    for k in range(7):
        t_laser_base_idx = 31 + k
        t_laser_base_diff = opt_t_laser_base_params[k] - init_t_laser_base_params[k]
        status = "已优化" if t_laser_base_idx in opt_indices else "已固定"
        print(f"{'-':^6}|{t_laser_base_param_names[k]:^12}|{init_t_laser_base_params[k]:^15.4f}|{opt_t_laser_base_params[k]:^15.4f}|{t_laser_base_diff:^15.4f}|{status:^10}")

    print("="*70)
