import numpy as np
import math
import sys
import os
import copy

# 添加父目录到路径，以便导入jacobian_analytical模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_jacobian.jacobian_analytical import RokaeRobot

# 创建修改版的RokaeRobot类，以便在build_jacobian_matrix函数中添加调试信息
class DebugRokaeRobot(RokaeRobot):
    def __init__(self):
        super().__init__()
        self.debug_info = {}
        
    def build_jacobian_matrix(self, q_deg_array):
        """
        修改版的build_jacobian_matrix函数，添加调试信息收集
        """
        # 保存原始输入
        self.debug_info['input_q_deg_array'] = copy.deepcopy(q_deg_array)
        
        valid, message = self.check_joint_limits(q_deg_array)
        if not valid:
            raise ValueError(message)
        
        # 初始化6x24雅可比矩阵
        J_N = np.zeros((6, 24))
        
        # 计算从基座到每个关节的变换矩阵
        transforms = self.compute_transforms_to_joints(q_deg_array)
        self.debug_info['transforms'] = copy.deepcopy(transforms)
        
        # 特别关注i=2的情况
        target_i = 2
        
        for i in range(6):
            # 获取局部雅可比矩阵（Mi）
            M_i, _, _, delta_d_contrib, _ = self.build_local_jacobian(i, q_deg_array[i])
            
            # 计算伴随变换矩阵: T_0_i
            if i == 0:
                theta_offset_0 = self.modified_dh_params[0][0]
                alpha_0 = self.modified_dh_params[0][1]
                d_0 = self.modified_dh_params[0][2]
                a_0 = self.modified_dh_params[0][3]
                actual_theta_0 = q_deg_array[0] + theta_offset_0
                T_0_0 = self.modified_dh_matrix(actual_theta_0, alpha_0, d_0, a_0)
                Ad_T_0_i = self.adjoint_transform(T_0_0)
            else:
                Ad_T_0_i = self.adjoint_transform(transforms[i-1])
            
            # 保存i=2的相关信息
            if i == target_i:
                self.debug_info[f'i{i}_q_deg'] = q_deg_array[i]
                self.debug_info[f'i{i}_dh_params'] = copy.deepcopy(self.modified_dh_params[i])
                self.debug_info[f'i{i}_M_i'] = copy.deepcopy(M_i)
                self.debug_info[f'i{i}_T_0_i'] = copy.deepcopy(transforms[i-1])
                self.debug_info[f'i{i}_Ad_T_0_i'] = copy.deepcopy(Ad_T_0_i)
            
            # 计算雅可比矩阵块: J_block_i = Ad_T_0_i * M_i
            J_block_i = Ad_T_0_i @ M_i
            
            if i == target_i:
                self.debug_info[f'i{i}_J_block_i'] = copy.deepcopy(J_block_i)
            
            # 将计算结果放入整体雅可比矩阵的对应位置
            col_start = i * 4
            J_N[:, col_start:col_start+4] = J_block_i
        
        return J_N

def debug_jacobian_calculation():
    """
    调试雅可比矩阵计算中的i=2情况
    """
    print("调试雅可比矩阵计算中的i=2情况")
    print("=" * 50)
    
    # 创建原始机器人类和调试版机器人类的实例
    original_robot = RokaeRobot()
    debug_robot = DebugRokaeRobot()
    
    # 测试一组关节角度
    q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
    
    # 在原始类中执行验证计算
    print("\n在原始类中执行验证计算:")
    
    # 获取从基座到每个关节的变换矩阵
    transforms = original_robot.compute_transforms_to_joints(q_deg_array)
    
    # 计算目标关节(i=2)的局部雅可比矩阵
    i = 2
    M_i_validation, _, _, _, _ = original_robot.build_local_jacobian(i, q_deg_array[i])
    print(f"验证计算的M_{i}:\n{M_i_validation}")
    
    # 获取T_0_i
    T_0_i_validation = transforms[i-1]
    print(f"验证计算的T_0_{i}:\n{T_0_i_validation}")
    
    # 计算伴随变换矩阵
    Ad_T_0_i_validation = original_robot.adjoint_transform(T_0_i_validation)
    print(f"验证计算的Ad_T_0_{i}:\n{Ad_T_0_i_validation}")
    
    # 计算雅可比块
    J_block_i_validation = Ad_T_0_i_validation @ M_i_validation
    print(f"验证计算的J_block_{i}:\n{J_block_i_validation}")
    
    # 在调试类中执行build_jacobian_matrix以收集内部计算值
    debug_robot.build_jacobian_matrix(q_deg_array.copy())
    
    # 提取调试信息
    print("\n在build_jacobian_matrix内部的计算值:")
    print(f"内部使用的q_deg[{i}]: {debug_robot.debug_info[f'i{i}_q_deg']}")
    print(f"内部使用的DH参数[{i}]: {debug_robot.debug_info[f'i{i}_dh_params']}")
    print(f"内部计算的M_{i}:\n{debug_robot.debug_info[f'i{i}_M_i']}")
    print(f"内部使用的T_0_{i}:\n{debug_robot.debug_info[f'i{i}_T_0_i']}")
    print(f"内部计算的Ad_T_0_{i}:\n{debug_robot.debug_info[f'i{i}_Ad_T_0_i']}")
    print(f"内部计算的J_block_{i}:\n{debug_robot.debug_info[f'i{i}_J_block_i']}")
    
    # 比较两种方式计算的结果
    print("\n比较验证计算和内部计算的差异:")
    q_diff = np.abs(q_deg_array[i] - debug_robot.debug_info[f'i{i}_q_deg'])
    print(f"q_deg[{i}]差异: {q_diff}")
    
    M_diff = np.linalg.norm(M_i_validation - debug_robot.debug_info[f'i{i}_M_i'])
    print(f"M_{i}差异: {M_diff:.12f}")
    
    T_diff = np.linalg.norm(T_0_i_validation - debug_robot.debug_info[f'i{i}_T_0_i'])
    print(f"T_0_{i}差异: {T_diff:.12f}")
    
    Ad_diff = np.linalg.norm(Ad_T_0_i_validation - debug_robot.debug_info[f'i{i}_Ad_T_0_i'])
    print(f"Ad_T_0_{i}差异: {Ad_diff:.12f}")
    
    J_block_diff = np.linalg.norm(J_block_i_validation - debug_robot.debug_info[f'i{i}_J_block_i'])
    print(f"J_block_{i}差异: {J_block_diff:.12f}")
    
    # 如果发现差异，输出更多详细信息
    if J_block_diff > 1e-10:
        print("\n发现明显差异，输出详细信息:")
        
        # 检查Ad_T_0_i可能的差异
        if Ad_diff > 1e-10:
            print("Ad_T_0_i差异矩阵:")
            print(Ad_T_0_i_validation - debug_robot.debug_info[f'i{i}_Ad_T_0_i'])
        
        # 检查J_block_i可能的差异
        print("J_block_i差异矩阵:")
        print(J_block_i_validation - debug_robot.debug_info[f'i{i}_J_block_i'])
        
        # 检查雅可比块的各部分
        print("验证计算的J_block_i的平移部分:")
        print(J_block_i_validation[:3, :])
        print("内部计算的J_block_i的平移部分:")
        print(debug_robot.debug_info[f'i{i}_J_block_i'][:3, :])
        print("平移部分差异:")
        print(J_block_i_validation[:3, :] - debug_robot.debug_info[f'i{i}_J_block_i'][:3, :])

def debug_adjoint_transform_calculation():
    """
    调试伴随变换矩阵计算
    """
    print("\n调试伴随变换矩阵计算")
    print("=" * 50)
    
    robot = RokaeRobot()
    q_deg_array = np.array([42.91, -0.41, 49.04, -119.33, 78.66, -5.23])
    
    # 获取从基座到每个关节的变换矩阵
    transforms = robot.compute_transforms_to_joints(q_deg_array)
    
    # 检查i=0的伴随变换矩阵计算
    i = 0
    theta_offset_0 = robot.modified_dh_params[0][0]
    alpha_0 = robot.modified_dh_params[0][1]
    d_0 = robot.modified_dh_params[0][2]
    a_0 = robot.modified_dh_params[0][3]
    actual_theta_0 = q_deg_array[0] + theta_offset_0
    T_0_0 = robot.modified_dh_matrix(actual_theta_0, alpha_0, d_0, a_0)
    
    print(f"T_0_0:\n{T_0_0}")
    
    # 计算伴随变换矩阵
    Ad_T_0_0 = robot.adjoint_transform(T_0_0)
    print(f"Ad_T_0_0:\n{Ad_T_0_0}")
    
    # 验证伴随变换矩阵的正确性
    print("\n验证伴随变换矩阵的正确性:")
    
    # 提取旋转矩阵和平移向量
    R = T_0_0[:3, :3]
    p = T_0_0[:3, 3]
    
    # 手动构建p的反对称矩阵
    p_skew = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    
    # 手动构建伴随变换矩阵
    Ad_T_0_0_manual = np.zeros((6, 6))
    Ad_T_0_0_manual[:3, :3] = R
    Ad_T_0_0_manual[:3, 3:] = p_skew @ R
    Ad_T_0_0_manual[3:, 3:] = R
    
    print(f"手动构建的Ad_T_0_0:\n{Ad_T_0_0_manual}")
    
    # 计算差异
    ad_diff = np.linalg.norm(Ad_T_0_0 - Ad_T_0_0_manual)
    print(f"伴随变换矩阵差异: {ad_diff:.12f}")
    
    if ad_diff > 1e-10:
        print("伴随变换矩阵计算可能存在问题！")
        print("差异矩阵:")
        print(Ad_T_0_0 - Ad_T_0_0_manual)

if __name__ == "__main__":
    # 调试i=2情况下的雅可比计算
    debug_jacobian_calculation()
    
    # 调试伴随变换矩阵计算
    debug_adjoint_transform_calculation()
