import os
import sys
from scipy.spatial.transform import Rotation as R

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入标定模块
from tools.calibrate import calibrate_AX_equals_YB, calculate_T_flange, tool_pos_to_transform_matrix
from tools.data_loader import load_joint_angles, extract_laser_positions_from_raw, get_initial_params

# 导入LM优化模块
from src.lm_optimize_pytorch import alternate_optimize_parameters, save_optimization_results

# 导入结果分析模块
from tools.result_analyzer import evaluate_optimization

def extract_optimized_params(optimized_params):
    """从优化后的参数中提取TCP和基座参数"""
    # 参数结构：DH(24) + TCP(7) + 基座(7) = 38个参数
    dh_params = optimized_params[0:24]      # DH参数 (24个)
    tcp_params = optimized_params[24:31]    # TCP参数 (7个)
    base_params = optimized_params[31:38]   # 基座参数 (7个)
    
    return dh_params, tcp_params, base_params

def main():
    """主函数：执行标定和优化"""
    
    print("=" * 60)
    print("    DH参数标定与优化系统")
    print("=" * 60)
    
    try:
        # 加载数据
        print("📂 加载数据...")
        joint_angles_data = load_joint_angles()
        tool_pos_data = extract_laser_positions_from_raw()
        print(f"✅ 关节角度数据: {joint_angles_data.shape[0]} 组")
        print(f"✅ 激光位置数据: {tool_pos_data.shape[0]} 组")
        
        # 计算变换矩阵
        print("\n🔄 计算机器人运动学变换...")
        T_flange_list = calculate_T_flange(joint_angles_data)
        Tool_transform_matrix_list = tool_pos_to_transform_matrix(tool_pos_data)
        
        # AX=YB标定
        print("\n🎯 执行 AX=YB 标定...")
        _, _, _ = calibrate_AX_equals_YB(T_flange_list, Tool_transform_matrix_list)
        
        # LM优化
        print("\n🚀 执行LM优化...")
        initial_params = get_initial_params()
        optimized_params = alternate_optimize_parameters(initial_params)
        
        # 提取优化结果
        dh_params, tcp_params, base_params = extract_optimized_params(optimized_params)
        
        # 保存优化结果到文件
        print("\n💾 保存优化结果...")
        save_optimization_results(optimized_params, initial_params)
        
        # 评估优化效果并显示详细对比
        print("\n📊 优化效果评估...")
        evaluate_optimization(initial_params, optimized_params)
        
        # 显示详细参数对比
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
                status = "已优化"
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
            tcp_diff = opt_tcp_params[k] - init_tcp_params[k]
            status = "已优化"
            print(f"{'-':^6}|{tcp_param_names[k]:^12}|{init_tcp_params[k]:^15.4f}|{opt_tcp_params[k]:^15.4f}|{tcp_diff:^15.4f}|{status:^10}")

        # 添加激光跟踪仪-基座变换参数对比
        print("="*70)
        print(" "*25 + "激光跟踪仪-基座变换参数对比")
        print("="*70)
        t_laser_base_param_names = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
        init_t_laser_base_params = initial_params[31:38]
        opt_t_laser_base_params = optimized_params[31:38]
        for k in range(7):
            t_laser_base_diff = opt_t_laser_base_params[k] - init_t_laser_base_params[k]
            status = "已优化"
            print(f"{'-':^6}|{t_laser_base_param_names[k]:^12}|{init_t_laser_base_params[k]:^15.4f}|{opt_t_laser_base_params[k]:^15.4f}|{t_laser_base_diff:^15.4f}|{status:^10}")

        print("="*70)
        
        print("\n" + "=" * 60)
        print("🎊 标定与优化完成！")
        print("=" * 60)
        print(f"📍 DH参数: 已优化")
        print(f"📍 TCP参数: {tcp_params[:3]} (位置)")
        print(f"📍 基座参数: {base_params[:3]} (位置)")
        print("✅ 优化结果已成功保存到 results/ 目录")
        
        return optimized_params
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        print("请检查数据文件和配置是否正确")
        raise

if __name__ == "__main__":
    """程序入口点"""
    print("启动 DH 参数标定与优化系统...")
    final_params = main()
    print("\n✨ 程序执行完毕，已获得最优化的DH参数！")
