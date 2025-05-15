"""使用Levenberg-Marquardt算法优化DH参数"""
import os
import numpy as np
import torch
import random
from error_function import get_laser_tool_matrix
from jacobian_torch import compute_error_vector_jacobian, forward_kinematics_T, extract_pose_from_T

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
# 固定指定的参数
OPT_INDICES = [i for i in ALL_INDICES if i not in [0, 1, 2, 3, 5, 6, 9, 10, 13, 17, 18, 19, 20, 21, 22, 23] ]  # 只举例，实际可自定义

def compute_error_vector(dh_params, joint_angles, laser_matrix, weights=ERROR_WEIGHTS):
    """计算单个样本的误差向量"""
    # 转为张量并计算姿态差
    q_t = torch.as_tensor(joint_angles, dtype=torch.float64)
    dh_t = torch.as_tensor(dh_params, dtype=torch.float64)
    T_pred = forward_kinematics_T(q_t, dh_t)
    pose_pred = extract_pose_from_T(T_pred)
    T_laser = torch.as_tensor(laser_matrix, dtype=torch.float64)
    pose_laser = extract_pose_from_T(T_laser)
    return (pose_pred - pose_laser) * torch.as_tensor(weights, dtype=torch.float64)

def compute_total_error(dh_params, joint_angles, laser_matrices, weights=ERROR_WEIGHTS):
    """计算所有样本的总误差（2-范数）"""
    total_error = 0.0
    for i in range(len(joint_angles)):
        error_vec = compute_error_vector(dh_params, joint_angles[i], laser_matrices[i], weights)
        total_error += torch.sum(error_vec**2)
    return torch.sqrt(total_error)

def save_optimized_dh(dh_params, filepath='results/optimized_dh_parameters.csv'):
    """保存优化后的DH参数，以6×4矩阵格式"""
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dh_matrix = np.array(dh_params).reshape(6, 4)
    
  
    header = "theta_offset,alpha,d,a"
    row_labels = [f"Joint_{i+1}" for i in range(6)]
    with open(filepath, 'w') as f:
        f.write(f",{header}\n")  # 写入列标签
        for i, row in enumerate(dh_matrix):
            f.write(f"{row_labels[i]},{','.join(f'{val:.6f}' for val in row)}\n")
    
    print(f"优化后的DH参数已保存到: {filepath}")

def optimize_dh_parameters(initial_dh=GLOBAL_DH_PARAMS, max_iterations=50, lambda_init=0.01, tol=1e-10, opt_indices=None):
    dh = torch.tensor(initial_dh, dtype=torch.float64, requires_grad=False)
    lambda_val = lambda_init
    
    # 读取所有数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    n_samples = len(joint_angles)
    
    # 记录初始误差
    current_error = compute_total_error(dh, joint_angles, laser_matrices)
    # 计算平均误差
    avg_initial_error = current_error.item() / 30
    print(f"初始平均误差：{avg_initial_error:.6f}")
    
    # 处理可优化参数索引
    if opt_indices is None:
        opt_indices = list(range(len(initial_dh)))
    opt_indices = np.array(opt_indices)
    
    # LM迭代
    for iteration in range(max_iterations):
        all_errors = []
        all_jacobians = []
        for i in range(n_samples):
            error_vec = compute_error_vector(dh, joint_angles[i], laser_matrices[i])
            jacobian = compute_error_vector_jacobian(dh.numpy(), joint_angles[i], laser_matrices[i])
            all_errors.append(error_vec)
            all_jacobians.append(jacobian)
        error_vector = torch.cat(all_errors)
        J = torch.vstack(all_jacobians)
        # 只保留可优化参数的列
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
                    return dh.numpy()
                continue
                
            # 尝试更新
            dh_new = dh.clone()
            dh_new[opt_indices] += delta
            
            # 计算新误差
            new_error = compute_total_error(dh_new, joint_angles, laser_matrices)
            
            # 判断是否接受更新
            if new_error < current_error:
                dh = dh_new
                current_error = new_error
                lambda_val = max(lambda_val / 10, 1e-7)
                update_success = True
                print(f"迭代 {iteration+1}, 内部迭代 {inner_iterations}, 误差: {current_error.item():.6f}, λ = {lambda_val:.6e}")
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
                    return dh.numpy()
        
        if not update_success:
            print("内部迭代未收敛，继续主循环")
        # 检查收敛，仅在成功更新(delta已定义)时进行
        if update_success and torch.norm(delta) < tol:
            print(f"参数变化小于阈值 {tol}，在第 {iteration+1} 次迭代后收敛")
            break
    final_error = current_error.item()
    avg_final_error = final_error / 30
    improvement = (avg_initial_error - avg_final_error) / avg_initial_error * 100
    print(f"优化完成，初始平均误差: {avg_initial_error:.6f}, 最终平均误差: {avg_final_error:.6f}, 改进率: {improvement:.2f}%")
    return dh.numpy()

def evaluate_optimization(initial_dh, optimized_dh):
    """评估优化效果"""
    # 读取数据
    joint_angles = np.loadtxt(JOINT_ANGLE_FILE, delimiter=',', skiprows=1)
    laser_matrices = get_laser_tool_matrix()
    
    print("\n" + "="*50)
    print(" "*18 + "优化效果评估")
    print("="*50)
    print(f"{'姿态':^8}|{'初始误差':^15}|{'优化后误差':^15}|{'改进率':^10}")
    print("-"*50)
    
    total_initial_error = 0
    total_optimized_error = 0
    
    for i in range(len(joint_angles)):
        initial_error = torch.linalg.norm(compute_error_vector(initial_dh, joint_angles[i], laser_matrices[i]))
        optimized_error = torch.linalg.norm(compute_error_vector(optimized_dh, joint_angles[i], laser_matrices[i]))
        improvement = (1 - optimized_error/initial_error) * 100
        
        print(f"{i+1:^8}|{initial_error.item():^15.6f}|{optimized_error.item():^15.6f}|{improvement:^10.2f}%")
        
        total_initial_error += initial_error.item()
        total_optimized_error += optimized_error.item()
    
    avg_initial_error = total_initial_error / 30
    avg_optimized_error = total_optimized_error / 30
    avg_improvement = (1 - avg_optimized_error/avg_initial_error) * 100
    print("-"*50)
    print(f"{'总体平均':^8}|{avg_initial_error:^15.6f}|{avg_optimized_error:^15.6f}|{avg_improvement:^10.2f}%")
    print("="*50)

if __name__ == '__main__':
    # 定义初始DH参数
    initial_dh = [0, 0, 380, 0, 
                 -90, -90, 0, 30, 
                 0, 0, 0, 440, 
                 0, -90, 435, 35, 
                 0, 90, 0, 0, 
                 180, -90, 83, 0]
    
    opt_indices = OPT_INDICES
    print(f"固定参数索引: {[i for i in ALL_INDICES if i not in opt_indices]}")
    print(f"可优化参数索引: {opt_indices}")
    
    # 优化DH参数
    optimized_dh = optimize_dh_parameters(initial_dh, max_iterations=50, lambda_init=0.1, opt_indices=opt_indices)
    
    # 保存优化结果
    save_optimized_dh(optimized_dh)
    
    # 评估优化效果
    evaluate_optimization(initial_dh, optimized_dh)
    
    # 输出优化前后的DH参数对比
    print("\n" + "="*70)
    print(" "*25 + "DH参数对比")
    print("="*70)
    print(f"{'关节':^6}|{'参数':^12}|{'初始值':^15}|{'优化值':^15}|{'差异':^15}|{'状态':^10}")
    print("-"*70)
    
    param_names = ["theta_offset", "alpha", "d", "a"]
    
    # 将参数重构为6×4矩阵，方便查看
    init_dh_matrix = np.array(initial_dh).reshape(6, 4)
    opt_dh_matrix = np.array(optimized_dh).reshape(6, 4)
    
    for i in range(6):  # 6个关节
        for j in range(4):  # 每个关节4个参数
            param_idx = i * 4 + j  # 计算参数在原始向量中的索引
            param_diff = opt_dh_matrix[i, j] - init_dh_matrix[i, j]
            status = "已优化" if param_idx in opt_indices else "已固定"
            print(f"{i+1:^6}|{param_names[j]:^12}|{init_dh_matrix[i, j]:^15.4f}|{opt_dh_matrix[i, j]:^15.4f}|{param_diff:^15.4f}|{status:^10}")
        # 每个关节后添加分隔线
        if i < 5:  # 最后一个关节后不需要
            print("-"*70)
    
    print("="*70)
