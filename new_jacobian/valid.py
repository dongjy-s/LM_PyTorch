import numpy as np
import math
import sys
import os

# 添加父目录到路径，以便导入jacobian_analytical模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_jacobian.jacobian_analytical import RokaeRobot

def validate_local_jacobian():
    """
    使用数值差分验证build_local_jacobian函数的正确性
    """
    robot = RokaeRobot()
    
    # 测试参数
    joint_index = 0
    q_deg = 0.0
    delta = 1e-7  # 扰动量
    deg2rad = math.pi / 180.0
    
    # 获取原始DH参数
    theta_offset_orig = robot.modified_dh_params[joint_index][0]
    alpha_orig = robot.modified_dh_params[joint_index][1]
    d_orig = robot.modified_dh_params[joint_index][2]
    a_orig = robot.modified_dh_params[joint_index][3]
    
    # 获取解析计算的局部雅可比矩阵
    local_jacobian, _, _, _, _ = robot.build_local_jacobian(joint_index, q_deg)
    
    # 计算原始变换矩阵
    A_orig = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig)
    A_orig_inv = np.linalg.inv(A_orig)
    
    print(f"验证第{joint_index+1}个关节的局部雅可比矩阵：")
    
    # 验证第一列 (对theta_offset)
    A_plus = robot.modified_dh_matrix(q_deg + (theta_offset_orig + delta), alpha_orig, d_orig, a_orig)
    A_minus = robot.modified_dh_matrix(q_deg + (theta_offset_orig - delta), alpha_orig, d_orig, a_orig)
    
    # 计算数值近似的扭转矩阵
    dA_dtheta_approx = (A_plus - A_minus) / (2 * delta)
    S_theta_approx_matrix = A_orig_inv @ dA_dtheta_approx
    
    # 提取6x1向量
    dx_approx = S_theta_approx_matrix[0, 3]
    dy_approx = S_theta_approx_matrix[1, 3]
    dz_approx = S_theta_approx_matrix[2, 3]
    rx_approx = S_theta_approx_matrix[2, 1]
    ry_approx = S_theta_approx_matrix[0, 2]
    rz_approx = S_theta_approx_matrix[1, 0]
    
    theta_approx = np.array([dx_approx, dy_approx, dz_approx, rx_approx, ry_approx, rz_approx])
    
    # 比较解析结果和数值近似结果
    print("\n验证theta_offset参数：")
    print(f"解析结果: {local_jacobian[:, 0]}")
    print(f"数值结果: {theta_approx}")
    print(f"相对误差: {np.abs((local_jacobian[:, 0] - theta_approx) / (local_jacobian[:, 0] + 1e-10)) * 100}%")
    
    # 验证第二列 (对alpha)
    A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig + delta, d_orig, a_orig)
    A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig - delta, d_orig, a_orig)
    
    dA_dalpha_approx = (A_plus - A_minus) / (2 * delta)
    S_alpha_approx_matrix = A_orig_inv @ dA_dalpha_approx
    
    dx_approx = S_alpha_approx_matrix[0, 3]
    dy_approx = S_alpha_approx_matrix[1, 3]
    dz_approx = S_alpha_approx_matrix[2, 3]
    rx_approx = S_alpha_approx_matrix[2, 1]
    ry_approx = S_alpha_approx_matrix[0, 2]
    rz_approx = S_alpha_approx_matrix[1, 0]
    
    alpha_approx = np.array([dx_approx, dy_approx, dz_approx, rx_approx, ry_approx, rz_approx])
    
    print("\n验证alpha参数：")
    print(f"解析结果: {local_jacobian[:, 1]}")
    print(f"数值结果: {alpha_approx}")
    print(f"相对误差: {np.abs((local_jacobian[:, 1] - alpha_approx) / (local_jacobian[:, 1] + 1e-10)) * 100}%")
    
    # 验证第三列 (对d)
    A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig + delta, a_orig)
    A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig - delta, a_orig)
    
    dA_dd_approx = (A_plus - A_minus) / (2 * delta)
    S_d_approx_matrix = A_orig_inv @ dA_dd_approx
    
    dx_approx = S_d_approx_matrix[0, 3]
    dy_approx = S_d_approx_matrix[1, 3]
    dz_approx = S_d_approx_matrix[2, 3]
    rx_approx = S_d_approx_matrix[2, 1]
    ry_approx = S_d_approx_matrix[0, 2]
    rz_approx = S_d_approx_matrix[1, 0]
    
    d_approx = np.array([dx_approx, dy_approx, dz_approx, rx_approx, ry_approx, rz_approx])
    
    print("\n验证d参数：")
    print(f"解析结果: {local_jacobian[:, 2]}")
    print(f"数值结果: {d_approx}")
    print(f"相对误差: {np.abs((local_jacobian[:, 2] - d_approx) / (local_jacobian[:, 2] + 1e-10)) * 100}%")
    
    # 验证第四列 (对a)
    A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig + delta)
    A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig - delta)
    
    dA_da_approx = (A_plus - A_minus) / (2 * delta)
    S_a_approx_matrix = A_orig_inv @ dA_da_approx
    
    dx_approx = S_a_approx_matrix[0, 3]
    dy_approx = S_a_approx_matrix[1, 3]
    dz_approx = S_a_approx_matrix[2, 3]
    rx_approx = S_a_approx_matrix[2, 1]
    ry_approx = S_a_approx_matrix[0, 2]
    rz_approx = S_a_approx_matrix[1, 0]
    
    a_approx = np.array([dx_approx, dy_approx, dz_approx, rx_approx, ry_approx, rz_approx])
    
    print("\n验证a参数：")
    print(f"解析结果: {local_jacobian[:, 3]}")
    print(f"数值结果: {a_approx}")
    print(f"相对误差: {np.abs((local_jacobian[:, 3] - a_approx) / (local_jacobian[:, 3] + 1e-10)) * 100}%")

def validate_all_joints():
    """
    验证所有关节的局部雅可比矩阵
    """
    robot = RokaeRobot()
    
    # 测试一组关节角度
    q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
    
    print("验证所有关节的局部雅可比矩阵")
    print("=" * 50)
    
    for joint_index in range(6):
        print(f"\n验证第{joint_index+1}个关节:")
        
        # 获取原始DH参数
        theta_offset_orig = robot.modified_dh_params[joint_index][0]
        alpha_orig = robot.modified_dh_params[joint_index][1]
        d_orig = robot.modified_dh_params[joint_index][2]
        a_orig = robot.modified_dh_params[joint_index][3]
        
        # 获取当前关节角度
        q_deg = q_deg_array[joint_index]
        
        # 获取解析计算的局部雅可比矩阵
        local_jacobian, _, _, _, _ = robot.build_local_jacobian(joint_index, q_deg)
        
        # 设置扰动量
        delta = 1e-7
        
        # 计算原始变换矩阵
        A_orig = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig)
        A_orig_inv = np.linalg.inv(A_orig)
        
        # 验证所有参数
        params = ["theta_offset", "alpha", "d", "a"]
        
        for param_idx, param_name in enumerate(params):
            # 根据参数类型设置扰动
            if param_idx == 0:  # theta_offset
                A_plus = robot.modified_dh_matrix(q_deg + (theta_offset_orig + delta), alpha_orig, d_orig, a_orig)
                A_minus = robot.modified_dh_matrix(q_deg + (theta_offset_orig - delta), alpha_orig, d_orig, a_orig)
            elif param_idx == 1:  # alpha
                A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig + delta, d_orig, a_orig)
                A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig - delta, d_orig, a_orig)
            elif param_idx == 2:  # d
                A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig + delta, a_orig)
                A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig - delta, a_orig)
            else:  # a
                A_plus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig + delta)
                A_minus = robot.modified_dh_matrix(q_deg + theta_offset_orig, alpha_orig, d_orig, a_orig - delta)
            
            # 计算数值近似的扭转矩阵
            dA_approx = (A_plus - A_minus) / (2 * delta)
            S_approx_matrix = A_orig_inv @ dA_approx
            
            # 提取6x1向量
            dx_approx = S_approx_matrix[0, 3]
            dy_approx = S_approx_matrix[1, 3]
            dz_approx = S_approx_matrix[2, 3]
            rx_approx = S_approx_matrix[2, 1]
            ry_approx = S_approx_matrix[0, 2]
            rz_approx = S_approx_matrix[1, 0]
            
            approx_vector = np.array([dx_approx, dy_approx, dz_approx, rx_approx, ry_approx, rz_approx])
            
            # 计算相对误差
            rel_error = np.abs((local_jacobian[:, param_idx] - approx_vector) / (np.abs(local_jacobian[:, param_idx]) + 1e-10)) * 100
            max_error = np.max(rel_error)
            
            print(f"  {param_name} 参数最大相对误差: {max_error:.6f}%")
            
            # 如果误差较大，输出详细信息
            if max_error > 1.0:
                print(f"    解析结果: {local_jacobian[:, param_idx]}")
                print(f"    数值结果: {approx_vector}")
                print(f"    相对误差: {rel_error}%")

if __name__ == "__main__":
    # 验证单个关节
    validate_local_jacobian()
    
    print("\n" + "=" * 50)
    
    # 验证所有关节
    validate_all_joints()
