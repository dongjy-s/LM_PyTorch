import numpy as np
import math
import sys
import os

# 添加父目录到路径，以便导入jacobian_analytical模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_jacobian.jacobian_analytical import RokaeRobot

def validate_compute_transforms_to_joints():
    """
    验证compute_transforms_to_joints函数的正确性
    """
    robot = RokaeRobot()
    
    # 测试一组关节角度
    q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
    
    print("验证compute_transforms_to_joints函数")
    print("=" * 50)
    
    # 获取从基座到每个关节的变换矩阵
    transforms = robot.compute_transforms_to_joints(q_deg_array)
    
    # 通过手动累积变换来验证
    T_manual = np.eye(4)
    
    for i in range(len(robot.modified_dh_params)):
        theta_offset_i = robot.modified_dh_params[i][0]
        alpha_i = robot.modified_dh_params[i][1]
        d_i = robot.modified_dh_params[i][2]
        a_i = robot.modified_dh_params[i][3]
        
        actual_theta_i = q_deg_array[i] + theta_offset_i
        
        A_i = robot.modified_dh_matrix(actual_theta_i, alpha_i, d_i, a_i)
        T_manual = T_manual @ A_i
        
        print(f"\n验证T_0_{i+1}变换矩阵:")
        print(f"compute_transforms_to_joints结果:\n{transforms[i]}")
        print(f"手动计算结果:\n{T_manual}")
        
        # 计算误差
        error = np.linalg.norm(transforms[i] - T_manual)
        print(f"误差: {error:.12f}")
        
        if error > 1e-10:
            print("警告：变换矩阵存在较大误差！")

def validate_build_jacobian_matrix():
    """
    验证build_jacobian_matrix函数的中间结果
    """
    robot = RokaeRobot()
    
    # 测试一组关节角度
    q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
    
    print("\n验证build_jacobian_matrix函数的中间结果")
    print("=" * 50)
    
    # 获取从基座到每个关节的变换矩阵
    transforms = robot.compute_transforms_to_joints(q_deg_array)
    
    # 尝试获取完整雅可比矩阵（如果可用）
    try:
        J_full = robot.build_jacobian_matrix(q_deg_array)
        print(f"完整雅可比矩阵形状: {J_full.shape}")
    except Exception as e:
        print(f"获取完整雅可比矩阵失败: {e}")
        J_full = None
    
    # 验证每个连杆的局部雅可比矩阵
    for i in range(6):
        print(f"\n验证第{i+1}个连杆的雅可比块:")
        
        # 获取局部雅可比矩阵M_i
        M_i, _, _, _, _ = robot.build_local_jacobian(i, q_deg_array[i])
        print(f"M_{i} (局部雅可比矩阵):\n{M_i}")
        
        # 计算变换矩阵T_0_i
        if i == 0:
            theta_offset_0 = robot.modified_dh_params[0][0]
            alpha_0 = robot.modified_dh_params[0][1]
            d_0 = robot.modified_dh_params[0][2]
            a_0 = robot.modified_dh_params[0][3]
            actual_theta_0 = q_deg_array[0] + theta_offset_0
            T_0_i = robot.modified_dh_matrix(actual_theta_0, alpha_0, d_0, a_0)
            print(f"T_0_{i} (变换矩阵):\n{T_0_i}")
        else:
            T_0_i = transforms[i-1]
            print(f"T_0_{i} (从transforms[{i-1}]):\n{T_0_i}")
        
        # 计算伴随变换矩阵Ad_T_0_i
        Ad_T_0_i = robot.adjoint_transform(T_0_i)
        print(f"Ad_T_0_{i} (伴随变换矩阵):\n{Ad_T_0_i}")
        
        # 计算雅可比块J_block_i
        J_block_i = Ad_T_0_i @ M_i
        print(f"J_block_{i} = Ad_T_0_{i} @ M_{i}:\n{J_block_i}")
        
        # 如果有完整雅可比矩阵，与相应的列进行比较
        if J_full is not None:
            col_start = i * 4
            col_end = col_start + 4
            J_block_from_full = J_full[:, col_start:col_end]
            
            print(f"J_full[:, {col_start}:{col_end}]:\n{J_block_from_full}")
            
            # 计算差异
            diff = np.linalg.norm(J_block_i - J_block_from_full)
            print(f"差异: {diff:.12f}")
            
            # 特别比较平移部分
            trans_diff = np.linalg.norm(J_block_i[:3, :] - J_block_from_full[:3, :])
            print(f"平移部分差异: {trans_diff:.12f}")
            
            if diff > 1e-10:
                print("警告：雅可比块与完整雅可比矩阵中的对应块存在较大误差！")

def compare_with_external_jacobian(external_jacobian_file=None):
    """
    将计算的雅可比矩阵与外部文件中的雅可比矩阵进行比较
    """
    if external_jacobian_file is None:
        print("\n没有提供外部雅可比矩阵文件，跳过比较")
        return
    
    try:
        external_J = np.loadtxt(external_jacobian_file, delimiter=',')
        print(f"\n加载外部雅可比矩阵: {external_jacobian_file}")
        print(f"外部雅可比矩阵形状: {external_J.shape}")
        
        robot = RokaeRobot()
        q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
        J_full = robot.build_jacobian_matrix(q_deg_array)
        
        print("\n比较雅可比矩阵")
        print("=" * 50)
        
        for i in range(6):
            col_start = i * 4
            col_end = col_start + 4
            
            # 获取当前连杆的雅可比块
            J_block_internal = J_full[:, col_start:col_end]
            J_block_external = external_J[:, col_start:col_end]
            
            print(f"\n第{i+1}个连杆的雅可比块比较:")
            
            # 计算平移部分的差异
            trans_diff = np.linalg.norm(J_block_internal[:3, :] - J_block_external[:3, :])
            print(f"平移部分差异: {trans_diff:.12f}")
            
            # 如果差异较大，输出详细信息
            if trans_diff > 1e-4:
                print("内部计算平移部分:")
                print(J_block_internal[:3, :])
                print("外部计算平移部分:")
                print(J_block_external[:3, :])
                print("平移部分差异矩阵:")
                print(J_block_internal[:3, :] - J_block_external[:3, :])
            
            # 计算旋转部分的差异
            rot_diff = np.linalg.norm(J_block_internal[3:, :] - J_block_external[3:, :])
            print(f"旋转部分差异: {rot_diff:.12f}")
            
            if rot_diff > 1e-4:
                print("内部计算旋转部分:")
                print(J_block_internal[3:, :])
                print("外部计算旋转部分:")
                print(J_block_external[3:, :])
                print("旋转部分差异矩阵:")
                print(J_block_internal[3:, :] - J_block_external[3:, :])
            
    except Exception as e:
        print(f"比较外部雅可比矩阵时出错: {e}")

if __name__ == "__main__":
    # 验证从基座到每个关节的变换矩阵
    validate_compute_transforms_to_joints()
    
    # 验证雅可比矩阵计算中的中间结果
    validate_build_jacobian_matrix()
    
    # 如果有外部雅可比矩阵文件，进行比较
    # 这里可以根据需要提供外部雅可比矩阵文件路径
    # compare_with_external_jacobian("data/external_jacobian.csv")
