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
from src.lm_optimize_pytorch import alternate_optimize_parameters

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
        
        print("\n" + "=" * 60)
        print("🎊 标定与优化完成！")
        print("=" * 60)
        print(f"📍 DH参数: 已优化")
        print(f"📍 TCP参数: {tcp_params[:3]} (位置)")
        print(f"📍 基座参数: {base_params[:3]} (位置)")
        print("📁 优化结果已保存到 results/ 目录")
        
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
