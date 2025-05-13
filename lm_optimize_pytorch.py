"""使用Levenberg-Marquardt算法优化DH参数"""
import os
import numpy as np
import torch
from jacobian_torch import compute_error_vector_jacobian, forward_kinematics_T, extract_pose_from_T, get_laser_tool_matrix


# 常量定义
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

# 全部参数索引(0-23)
ALL_INDICES = list(range(len(GLOBAL_DH_PARAMS)))
# 直接指定固定参数索引
OPT_INDICES = [i for i in ALL_INDICES if i not in [0,1,2,3,5,9,13,17,20,21,22,23]]  

#* 添加TCP参数的初始值 (来自 jacobian_torch.py)
INITIAL_TCP_POSITION = np.array([2, 2, 100])
INITIAL_TCP_QUATERNION = np.array([0.50, 0.50, 0.50, 0.50])

# 直接指定固定参数索引 (DH部分)
DH_FIXED_INDICES = [0, 1, 2, 3, 5, 9, 13, 17, 20, 21, 22, 23]
# 是否优化TCP参数
OPTIMIZE_TCP = True # 可以设置为 False 来仅优化DH

def compute_error_vector(params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    """计算单个样本的误差向量"""
    # 转为张量并计算姿态差
    q_t = torch.as_tensor(joint_angles, dtype=torch.float64)
    params_t = torch.as_tensor(params, dtype=torch.float64) # 使用组合参数
    T_pred = forward_kinematics_T(q_t, params_t) # 调用更新后的FK，传递组合参数
    pose_pred = extract_pose_from_T(T_pred)
    T_laser = torch.as_tensor(laser_matrix, dtype=torch.float64)
    pose_laser = extract_pose_from_T(T_laser)
    return (pose_pred - pose_laser) * torch.as_tensor(weights, dtype=torch.float64)

def compute_total_error(params, joint_angles, laser_matrices, weights=ERROR_WEIGHTS):
    """计算所有样本的总误差（2-范数）"""
    total_error = 0.0
    for i in range(len(joint_angles)):
        error_vec = compute_error_vector(params, joint_angles[i], laser_matrices[i], weights) # 使用组合参数
        total_error += torch.sum(error_vec**2)
    return torch.sqrt(total_error)

def save_optimization_results(params, filepath_prefix='results/optimized'):
    """保存优化后的DH参数和TCP参数"""
    dirpath = os.path.dirname(filepath_prefix)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # 分离参数
    dh_params = params[0:24]
    tcp_params = params[24:31]
    
    # 1. 保存DH参数 (6x4 格式)
    dh_filepath = f"{filepath_prefix}_dh_parameters.csv"
    dh_matrix = np.array(dh_params).reshape(6, 4)
    header_dh = "theta_offset,alpha,d,a"
    row_labels_dh = [f"Joint_{i+1}" for i in range(6)]
    with open(dh_filepath, 'w') as f:
        f.write(f",{header_dh}\n")  
        for i, row in enumerate(dh_matrix):
            f.write(f"{row_labels_dh[i]},{','.join(f'{val:.6f}' for val in row)}\n")
    print(f"优化后的DH参数已保存到: {dh_filepath}")

    # 2. 保存TCP参数
    tcp_filepath = f"{filepath_prefix}_tcp_parameters.csv"
    header_tcp = "parameter,value"
    tcp_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    with open(tcp_filepath, 'w') as f:
        f.write(f"{header_tcp}\n")
        for name, value in zip(tcp_param_names, tcp_params):
            f.write(f"{name},{value:.6f}\n")
    print(f"优化后的TCP参数已保存到: {tcp_filepath}")

def optimize_dh_parameters(initial_params, max_iterations=50, lambda_init=0.01, tol=1e-10, opt_indices=None):
    params = torch.tensor(initial_params, dtype=torch.float64, requires_grad=False)
    lambda_val = lambda_init
    
    # 读取所有数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    n_samples = len(joint_angles)
    
    # 记录初始误差
    current_error = compute_total_error(params, joint_angles, laser_matrices)
    # 计算平均误差
    avg_initial_error = current_error.item() / n_samples
    print(f"初始平均误差：{avg_initial_error:.6f}")
    
    # 处理可优化参数索引
    if opt_indices is None:
        opt_indices = list(range(len(initial_params)))
    opt_indices = np.array(opt_indices)
    
    # LM迭代
    for iteration in range(max_iterations):
        all_errors = []
        all_jacobians = []
        for i in range(n_samples):
            error_vec = compute_error_vector(params, joint_angles[i], laser_matrices[i])
            jacobian = compute_error_vector_jacobian(params.numpy(), joint_angles[i], laser_matrices[i])
            all_errors.append(error_vec)
            all_jacobians.append(jacobian)
        error_vector = torch.cat(all_errors)
        #* 将所有雅可比矩阵堆叠成一个矩阵（N*6 x 31）
        J = torch.vstack(all_jacobians)
        #* LM算法公式：(J^T * J + λ * I) * Δθ = -J^T * e
        J_opt = J[:, opt_indices]
        JTJ = torch.matmul(J_opt.transpose(0, 1), J_opt)
        JTe = torch.matmul(J_opt.transpose(0, 1), error_vector)
        update_success = False
        inner_iterations = 0
        max_inner_iterations = 10
        while not update_success and inner_iterations < max_inner_iterations:
            inner_iterations += 1
            
            # 添加阻尼项
            H = JTJ + lambda_val * torch.diag(torch.diag(JTJ))
            
            # 计算更新量
            try:
                delta = -torch.linalg.solve(H, JTe)
            except:
                print(f"矩阵求解错误，增大阻尼因子 λ = {lambda_val} -> {lambda_val * 10}")
                lambda_val *= 10
                # 如果阻尼因子过大，提前结束优化
                if lambda_val > 1e5:
                    print(f"阻尼因子超过阈值 1e5，提前结束优化")
                    return params.numpy()
                continue
                
            # 尝试更新
            params_new = params.clone()
            params_new[opt_indices] += delta
            
            # 计算新误差
            new_error = compute_total_error(params_new, joint_angles, laser_matrices)
            
            # 判断是否接受更新
            if new_error < current_error:
                params = params_new
                current_error = new_error
                lambda_val = max(lambda_val / 10, 1e-7)
                update_success = True

                #* 四元数归一化 (如果TCP被优化)
                if any(idx in opt_indices for idx in range(24, 31)):
                    q_tcp = params[27:31] # 提取四元数部分
                    norm_q_tcp = torch.linalg.norm(q_tcp)
                    if norm_q_tcp > 1e-9: # 避免除以零
                        params[27:31] = q_tcp / norm_q_tcp
                    else:
                        # 如果模长为0（不太可能发生），可以重置为一个有效的单位四元数，如 [0,0,0,1]
                        # 或者发出警告/错误，具体取决于需求
                        params[27:31] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=params.dtype, device=params.device)
                        print("警告: TCP四元数模长接近于零，已重置为[0,0,0,1]")

                print(f"迭代 {iteration+1}, 误差: {current_error.item():.6f}, λ = {lambda_val:.6e}")
            else:
                lambda_val *= 10
                print(f"拒绝更新，增大阻尼因子 λ = {lambda_val}")
                
                # 如果阻尼因子过大，可能表明已经接近局部最小值
                if lambda_val > 1e8:
                    print("阻尼因子过大，停止当前迭代")
                    break
                # 如果阻尼因子超过一定阈值，提前结束优化
                if lambda_val > 1e5:
                    print(f"阻尼因子超过阈值 1e5，提前结束优化")
                    return params.numpy()
        
        if not update_success:
            print("内部迭代未收敛，继续主循环")
        # 检查收敛，仅在成功更新(delta已定义)时进行
        if update_success and torch.norm(delta) < tol:
            print(f"参数变化小于阈值 {tol}，在第 {iteration+1} 次迭代后收敛")
            break
    final_error = current_error.item()
    avg_final_error = final_error / n_samples
    improvement = (avg_initial_error - avg_final_error) / avg_initial_error * 100 if avg_initial_error > 1e-9 else 0
    print(f"优化完成，初始平均误差: {avg_initial_error:.6f}, 最终平均误差: {avg_final_error:.6f}, 改进率: {improvement:.2f}%")
    return params.numpy()

def evaluate_optimization(initial_params, optimized_params):
    """评估优化效果"""
    # 读取数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    n_samples = len(joint_angles)
    
    print("\n" + "="*50)
    print(" "*18 + "优化效果评估")
    print("="*50)
    print(f"{'姿态':^8}|{'初始误差':^15}|{'优化后误差':^15}|{'改进率':^10}")
    print("-"*50)
    
    total_initial_error = 0
    total_optimized_error = 0
    
    for i in range(len(joint_angles)):
        initial_error = torch.linalg.norm(compute_error_vector(initial_params, joint_angles[i], laser_matrices[i]))
        optimized_error = torch.linalg.norm(compute_error_vector(optimized_params, joint_angles[i], laser_matrices[i]))
        improvement = (1 - optimized_error/initial_error) * 100 if initial_error > 1e-9 else 0
        
        print(f"{i+1:^8}|{initial_error.item():^15.6f}|{optimized_error.item():^15.6f}|{improvement:^10.2f}%")
        
        total_initial_error += initial_error.item()
        total_optimized_error += optimized_error.item()
    
    avg_initial_error = total_initial_error / n_samples
    avg_optimized_error = total_optimized_error / n_samples
    avg_improvement = (1 - avg_optimized_error/avg_initial_error) * 100 if avg_initial_error > 1e-9 else 0
    print("-"*50)
    print(f"{'总体平均':^8}|{avg_initial_error:^15.6f}|{avg_optimized_error:^15.6f}|{avg_improvement:^10.2f}%")
    print("="*50)


if __name__ == '__main__':
    # 定义初始DH参数 和 TCP参数
    initial_dh_params = np.array(GLOBAL_DH_PARAMS)
    initial_tcp_params = np.concatenate((INITIAL_TCP_POSITION, INITIAL_TCP_QUATERNION))
    initial_params = np.concatenate((initial_dh_params, initial_tcp_params)) # Create initial combined params
    
    # 确定优化索引
    dh_opt_indices = [i for i in range(24) if i not in DH_FIXED_INDICES]
    tcp_opt_indices = list(range(24, 31)) if OPTIMIZE_TCP else []
    opt_indices = dh_opt_indices + tcp_opt_indices
    
    fixed_indices = [i for i in range(31) if i not in opt_indices] # Calculate fixed indices
    print(f"固定参数索引 ({len(fixed_indices)}): {fixed_indices}")
    print(f"可优化参数索引 ({len(opt_indices)}): {opt_indices}")
    
    # 优化参数
    optimized_params = optimize_dh_parameters(initial_params, max_iterations=50, lambda_init=0.1, opt_indices=opt_indices) # Pass combined params
    
    # 保存优化结果
    save_optimization_results(optimized_params) # Call updated save function
    
    # 评估优化效果
    evaluate_optimization(initial_params, optimized_params) # Pass combined params
    
    # 输出优化前后的参数对比
    print("\n" + "="*70)
    print(" "*25 + "DH参数对比")
    print("="*70)
    print(f"{'关节':^6}|{'参数':^12}|{'初始值':^15}|{'优化值':^15}|{'差异':^15}|{'状态':^10}")
    print("-"*70)
    
    param_names = ["theta_offset", "alpha", "d", "a"]
    
    # 将参数重构为6×4矩阵，方便查看 (DH部分)
    init_dh_matrix = initial_params[0:24].reshape(6, 4)
    opt_dh_matrix = optimized_params[0:24].reshape(6, 4)
    
    for i in range(6):  
        for j in range(4):  
            param_idx = i * 4 + j  
            param_diff = opt_dh_matrix[i, j] - init_dh_matrix[i, j]
            status = "已优化" if param_idx in opt_indices else "已固定"
            print(f"{i+1:^6}|{param_names[j]:^12}|{init_dh_matrix[i, j]:^15.4f}|{opt_dh_matrix[i, j]:^15.4f}|{param_diff:^15.4f}|{status:^10}")
        # 每个关节后添加分隔线
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

    print("="*70)
