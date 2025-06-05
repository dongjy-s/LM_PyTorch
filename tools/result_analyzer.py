"""
结果分析模块
负责优化结果的分析、评估和对比功能
"""

import os
import sys
import numpy as np
import torch

# 添加路径以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from tools.data_loader import (
    load_joint_angles, get_laser_tool_matrix, get_error_weights
)

# 导入误差计算函数（需要从lm_optimize_pytorch导入）
def compute_error_vector(params, joint_angles, laser_matrix, weights=None):
    """
    导入误差计算函数
    这里需要从lm_optimize_pytorch模块导入实际的compute_error_vector函数
    """
    try:
        from src.lm_optimize_pytorch import compute_error_vector as _compute_error_vector
        return _compute_error_vector(params, joint_angles, laser_matrix, weights)
    except ImportError as e:
        print(f"无法导入compute_error_vector函数: {e}")
        raise

def compute_total_error_avg(params, joint_angles, laser_matrices, weights=None):
    """
    导入总误差计算函数
    这里需要从lm_optimize_pytorch模块导入实际的compute_total_error_avg函数
    """
    try:
        from src.lm_optimize_pytorch import compute_total_error_avg as _compute_total_error_avg
        return _compute_total_error_avg(params, joint_angles, laser_matrices, weights)
    except ImportError as e:
        print(f"无法导入compute_total_error_avg函数: {e}")
        raise

class OptimizationAnalyzer:
    """优化结果分析器"""
    
    def __init__(self):
        self.joint_angles = None
        self.laser_matrices = None
        self.weights = None
        self._load_data()
    
    def _load_data(self):
        """加载测试数据"""
        self.joint_angles = load_joint_angles()
        self.laser_matrices = get_laser_tool_matrix()
        self.weights = get_error_weights()
    
    def generate_detailed_comparison(self, initial_params, optimized_params):
        """
        生成优化前后的详细对比分析数据
        
        参数:
        initial_params: 初始参数
        optimized_params: 优化后的参数
        
        返回:
        pandas.DataFrame: 包含详细对比数据的DataFrame
        """
        try:
            import pandas as pd
        except ImportError:
            print("错误: 需要安装pandas库 (pip install pandas)")
            return None
        
        n_samples = len(self.joint_angles)
        print(f"   📊 分析 {n_samples} 组测试数据的优化效果...")
        
        comparison_data = []
        
        for i in range(n_samples):
            joint_angle = self.joint_angles[i]
            laser_matrix = self.laser_matrices[i]
            
            # 计算优化前的误差
            error_before = compute_error_vector(initial_params, joint_angle, laser_matrix, self.weights)
            pos_error_before = error_before[:3].detach().numpy()
            orient_error_before = error_before[3:].detach().numpy()
            total_error_before = torch.norm(error_before).item()
            
            # 计算优化前的分解误差（位置误差和姿态误差）
            pos_error_magnitude_before = np.sqrt(np.sum(pos_error_before**2))
            orient_error_magnitude_before = np.sqrt(np.sum(orient_error_before**2))
            
            # 计算优化后的误差
            error_after = compute_error_vector(optimized_params, joint_angle, laser_matrix, self.weights)
            pos_error_after = error_after[:3].detach().numpy()
            orient_error_after = error_after[3:].detach().numpy()
            total_error_after = torch.norm(error_after).item()
            
            # 计算优化后的分解误差（位置误差和姿态误差）
            pos_error_magnitude_after = np.sqrt(np.sum(pos_error_after**2))
            orient_error_magnitude_after = np.sqrt(np.sum(orient_error_after**2))
            
            # 计算改进率
            improvement_rate = ((total_error_before - total_error_after) / total_error_before) * 100 if total_error_before > 0 else 0
            pos_improvement_rate = ((pos_error_magnitude_before - pos_error_magnitude_after) / pos_error_magnitude_before) * 100 if pos_error_magnitude_before > 0 else 0
            orient_improvement_rate = ((orient_error_magnitude_before - orient_error_magnitude_after) / orient_error_magnitude_before) * 100 if orient_error_magnitude_before > 0 else 0
            
            # 添加到对比数据 - 按照优化前后成对的顺序排列
            comparison_data.append({
                '数据组': f'第{i+1}组',
                '优化前X误差(mm)': f'{pos_error_before[0]:.6f}',
                '优化后X误差(mm)': f'{pos_error_after[0]:.6f}',
                '优化前Y误差(mm)': f'{pos_error_before[1]:.6f}',
                '优化后Y误差(mm)': f'{pos_error_after[1]:.6f}',
                '优化前Z误差(mm)': f'{pos_error_before[2]:.6f}',
                '优化后Z误差(mm)': f'{pos_error_after[2]:.6f}',
                '优化前Rx误差(度)': f'{orient_error_before[0]:.6f}',
                '优化后Rx误差(度)': f'{orient_error_after[0]:.6f}',
                '优化前Ry误差(度)': f'{orient_error_before[1]:.6f}',
                '优化后Ry误差(度)': f'{orient_error_after[1]:.6f}',
                '优化前Rz误差(度)': f'{orient_error_before[2]:.6f}',
                '优化后Rz误差(度)': f'{orient_error_after[2]:.6f}',
                '优化前位置误差(mm)': f'{pos_error_magnitude_before:.6f}',
                '优化后位置误差(mm)': f'{pos_error_magnitude_after:.6f}',
                '优化前姿态误差(度)': f'{orient_error_magnitude_before:.6f}',
                '优化后姿态误差(度)': f'{orient_error_magnitude_after:.6f}',
                '优化前总误差(L2范数)': f'{total_error_before:.6f}',
                '优化后总误差(L2范数)': f'{total_error_after:.6f}',
                '位置改进率(%)': f'{pos_improvement_rate:.2f}%',
                '姿态改进率(%)': f'{orient_improvement_rate:.2f}%',
                '总误差改进率(%)': f'{improvement_rate:.2f}%'
            })
        
        # 计算统计信息
        total_errors_before = [float(row['优化前总误差(L2范数)']) for row in comparison_data]
        total_errors_after = [float(row['优化后总误差(L2范数)']) for row in comparison_data]
        pos_errors_before = [float(row['优化前位置误差(mm)']) for row in comparison_data]
        pos_errors_after = [float(row['优化后位置误差(mm)']) for row in comparison_data]
        orient_errors_before = [float(row['优化前姿态误差(度)']) for row in comparison_data]
        orient_errors_after = [float(row['优化后姿态误差(度)']) for row in comparison_data]
        
        avg_error_before = np.mean(total_errors_before)
        avg_error_after = np.mean(total_errors_after)
        avg_pos_error_before = np.mean(pos_errors_before)
        avg_pos_error_after = np.mean(pos_errors_after)
        avg_orient_error_before = np.mean(orient_errors_before)
        avg_orient_error_after = np.mean(orient_errors_after)
        
        overall_improvement = ((avg_error_before - avg_error_after) / avg_error_before) * 100 if avg_error_before > 0 else 0
        pos_overall_improvement = ((avg_pos_error_before - avg_pos_error_after) / avg_pos_error_before) * 100 if avg_pos_error_before > 0 else 0
        orient_overall_improvement = ((avg_orient_error_before - avg_orient_error_after) / avg_orient_error_before) * 100 if avg_orient_error_before > 0 else 0
        
        # 添加统计行
        comparison_data.append({
            '数据组': '平均值',
            '优化前X误差(mm)': '',
            '优化后X误差(mm)': '',
            '优化前Y误差(mm)': '',
            '优化后Y误差(mm)': '',
            '优化前Z误差(mm)': '',
            '优化后Z误差(mm)': '',
            '优化前Rx误差(度)': '',
            '优化后Rx误差(度)': '',
            '优化前Ry误差(度)': '',
            '优化后Ry误差(度)': '',
            '优化前Rz误差(度)': '',
            '优化后Rz误差(度)': '',
            '优化前位置误差(mm)': f'{avg_pos_error_before:.6f}',
            '优化后位置误差(mm)': f'{avg_pos_error_after:.6f}',
            '优化前姿态误差(度)': f'{avg_orient_error_before:.6f}',
            '优化后姿态误差(度)': f'{avg_orient_error_after:.6f}',
            '优化前总误差(L2范数)': f'{avg_error_before:.6f}',
            '优化后总误差(L2范数)': f'{avg_error_after:.6f}',
            '位置改进率(%)': f'{pos_overall_improvement:.2f}%',
            '姿态改进率(%)': f'{orient_overall_improvement:.2f}%',
            '总误差改进率(%)': f'{overall_improvement:.2f}%'
        })
        
        print(f"   ✅ 总体平均误差: {avg_error_before:.6f} → {avg_error_after:.6f}")
        print(f"   📍 位置误差: {avg_pos_error_before:.6f} → {avg_pos_error_after:.6f} (改进率: {pos_overall_improvement:.2f}%)")
        print(f"   🔄 姿态误差: {avg_orient_error_before:.6f} → {avg_orient_error_after:.6f} (改进率: {orient_overall_improvement:.2f}%)")
        print(f"   📈 总体改进率: {overall_improvement:.2f}%")
        
        return pd.DataFrame(comparison_data)
    
    def evaluate_optimization(self, initial_params, optimized_params):
        """
        评估优化效果，报告与优化器目标一致的均方根误差
        
        参数:
        initial_params: 初始参数
        optimized_params: 优化后参数
        """
        n_samples = len(self.joint_angles)

        if n_samples == 0:
            print("评估警告: 样本数量为0，无法进行评估。")
            return
        
        print("\n" + "="*60)
        print(" "*15 + "优化效果评估 (所有分量的均方根误差)")
        print("="*60)
        # 打印表头，明确指出总体平均误差是均方根误差
        print(f"{'姿态(帧)':^12}|{'初始单帧范数':^18}|{'优化后单帧范数':^20}|{'单帧改进率':^15}")
        print("-"*68)
        
        # 计算初始和优化后的总体均方根误差 (与compute_total_error_avg一致)
        initial_total_rmse = compute_total_error_avg(initial_params, self.joint_angles, self.laser_matrices).item()
        optimized_total_rmse = compute_total_error_avg(optimized_params, self.joint_angles, self.laser_matrices).item()

        # 逐帧显示误差范数及其改进，用于详细分析
        for i in range(n_samples):
            initial_error_vec = compute_error_vector(initial_params, self.joint_angles[i], self.laser_matrices[i])
            optimized_error_vec = compute_error_vector(optimized_params, self.joint_angles[i], self.laser_matrices[i])
            
            initial_frame_norm = torch.linalg.norm(initial_error_vec).item()
            optimized_frame_norm = torch.linalg.norm(optimized_error_vec).item()
            
            frame_improvement = (1 - optimized_frame_norm / initial_frame_norm) * 100 if initial_frame_norm > 1e-9 else 0
            
            print(f"{i+1:^12}|{initial_frame_norm:^18.6f}|{optimized_frame_norm:^20.6f}|{frame_improvement:^14.2f}%")
        
        # 计算总体改进率 (基于均方根误差)
        avg_improvement_rmse = (1 - optimized_total_rmse / initial_total_rmse) * 100 if initial_total_rmse > 1e-9 else 0
        
        print("-"*68)
        print(f"{'总体平均RMSE':^12}|{initial_total_rmse:^18.6f}|{optimized_total_rmse:^20.6f}|{avg_improvement_rmse:^14.2f}%")
        print("="*60)


# 向后兼容的函数接口
def generate_detailed_comparison(initial_params, optimized_params):
    """
    向后兼容的详细对比分析函数
    
    参数:
    initial_params: 初始参数
    optimized_params: 优化后参数
    
    返回:
    pandas.DataFrame: 包含详细对比数据的DataFrame
    """
    analyzer = OptimizationAnalyzer()
    return analyzer.generate_detailed_comparison(initial_params, optimized_params)

def evaluate_optimization(initial_params, optimized_params):
    """
    向后兼容的优化效果评估函数
    
    参数:
    initial_params: 初始参数
    optimized_params: 优化后参数
    """
    analyzer = OptimizationAnalyzer()
    return analyzer.evaluate_optimization(initial_params, optimized_params)

if __name__ == "__main__":
    print("结果分析模块测试")
    # 这里可以添加测试代码 