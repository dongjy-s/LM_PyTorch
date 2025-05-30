import numpy as np
import os
import sys

# 添加项目根目录到Python路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入现有的数据加载和误差计算函数
from tools.data_loader import (
    load_joint_angles, 
    get_laser_tool_matrix,
    get_initial_params,
    get_error_weights,
    load_dh_params,
    load_calibration_params
)
from src.lm_optimize_pytorch import compute_error_vector

def load_calibration_data():
    """
    从data_loader中加载30组关节角度和激光数据
    
    Returns:
        joint_angles: numpy数组，形状为(30, 6)，包含30组6轴关节角度
        laser_matrices: numpy数组，形状为(30, 4, 4)，包含30组激光测量位姿矩阵
    """
    print("正在加载标定数据...")
    
    try:
        # 加载关节角度数据 (30组，每组6个关节角度)
        joint_angles = load_joint_angles()
        print(f"✅ 成功加载关节角度数据: {joint_angles.shape}")
        
        # 加载激光位姿矩阵 (30组，每组4x4变换矩阵)
        laser_matrices = get_laser_tool_matrix()
        print(f"✅ 成功加载激光位姿矩阵: {laser_matrices.shape}")
        
        # 验证数据完整性
        if joint_angles.shape[0] != laser_matrices.shape[0]:
            raise ValueError(f"关节角度数据组数({joint_angles.shape[0]})与激光数据组数({laser_matrices.shape[0]})不匹配")
        
        if joint_angles.shape[0] != 30:
            print(f"⚠️ 警告: 期望30组数据，实际加载了{joint_angles.shape[0]}组数据")
        
        return joint_angles, laser_matrices
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        raise

def get_initial_dh_params():
    """
    获取初始DH参数（24个）
    
    Returns:
        dh_params: numpy数组，形状为(24,)，包含初始DH参数
    """
    dh_params = load_dh_params()
    print(f"✅ 加载初始DH参数: {dh_params.shape[0]}个参数")
    return dh_params

def get_initial_calib_params():
    """
    获取初始标定参数（TCP偏移 + 激光基座，共14个）
    
    Returns:
        calib_params: numpy数组，形状为(14,)，包含TCP偏移(7个) + 激光基座(7个)参数
    """
    tool_offset_params, laser_base_params = load_calibration_params()
    calib_params = np.concatenate([tool_offset_params, laser_base_params])
    print(f"✅ 加载初始标定参数: TCP偏移(7) + 激光基座(7) = {calib_params.shape[0]}个参数")
    return calib_params

def create_full_params(dh_params, calib_params):
    """
    组合DH参数和标定参数为完整的38个参数
    
    Args:
        dh_params: 24个DH参数
        calib_params: 14个标定参数（TCP偏移7个 + 激光基座7个）
    
    Returns:
        full_params: 38个完整参数
    """
    if len(dh_params) != 24:
        raise ValueError(f"DH参数应该是24个，实际得到{len(dh_params)}个")
    if len(calib_params) != 14:
        raise ValueError(f"标定参数应该是14个，实际得到{len(calib_params)}个")
    
    full_params = np.concatenate([dh_params, calib_params])
    print(f"✅ 组合参数: DH(24) + 标定(14) = {full_params.shape[0]}个参数")
    return full_params

def compute_single_pose_error(joint_angles, laser_matrix, params, weights=None):
    """
    计算单组数据的位姿误差（使用现有的误差计算函数）
    
    Args:
        joint_angles: 6个关节角度 (度)
        laser_matrix: 4x4激光测量位姿矩阵
        params: 38个参数 (DH + TCP + 激光基座)
        weights: 6维权重向量
    
    Returns:
        6维误差向量 [pos_error_x, pos_error_y, pos_error_z, orient_error_rx, orient_error_ry, orient_error_rz]
    """
    # 使用现有的误差计算函数
    error_vector = compute_error_vector(params, joint_angles, laser_matrix, weights)
    
    # 转换为numpy数组并返回
    return error_vector.detach().cpu().numpy()

def compute_all_errors(joint_angles_all, laser_matrices_all, dh_params=None, calib_params=None, weights=None):
    """
    计算所有测量数据的误差（DH参数和标定参数分开设置）
    
    Args:
        joint_angles_all: (N, 6) 所有关节角度
        laser_matrices_all: (N, 4, 4) 所有激光测量位姿矩阵
        dh_params: 24个DH参数，如果为None则使用初始DH参数
        calib_params: 14个标定参数（TCP7个+激光基座7个），如果为None则使用初始标定参数
        weights: 6维权重向量，如果为None则使用默认权重
    
    Returns:
        error_matrix: (N, 6) 所有误差向量
        total_error: 加权总误差
    """
    N = joint_angles_all.shape[0]
    error_matrix = np.zeros((N, 6))
    
    # 获取DH参数（如果未提供则使用初始参数）
    if dh_params is None:
        dh_params = get_initial_dh_params()
    else:
        print(f"使用用户提供的DH参数: {len(dh_params)}个参数")
    
    # 获取标定参数（如果未提供则使用初始参数）
    if calib_params is None:
        calib_params = get_initial_calib_params()
    else:
        print(f"使用用户提供的标定参数: {len(calib_params)}个参数")
        print(f"  - TCP偏移参数: {calib_params[:7]}")
        print(f"  - 激光基座参数: {calib_params[7:]}")
    
    # 组合为完整的38个参数
    full_params = create_full_params(dh_params, calib_params)
    
    # 获取权重
    if weights is None:
        weights = get_error_weights()
        print(f"使用默认权重: {weights}")
    
    print(f"正在计算{N}组数据的误差...")
    
    for i in range(N):
        try:
            error_vector = compute_single_pose_error(
                joint_angles_all[i], 
                laser_matrices_all[i], 
                full_params, 
                weights
            )
            error_matrix[i] = error_vector
        except Exception as e:
            print(f"❌ 计算第{i+1}组误差时出错: {e}")
            # 如果某组数据出错，设置为较大的误差值
            error_matrix[i] = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0])
    
    # 计算总误差 (RMS)
    total_error = np.sqrt(np.mean(error_matrix**2))
    
    return error_matrix, total_error

def analyze_errors(error_matrix):
    """
    分析误差统计信息
    
    Args:
        error_matrix: (N, 6) 误差矩阵
    """
    pos_errors = error_matrix[:, :3]  # 位置误差
    orient_errors = error_matrix[:, 3:]  # 姿态误差
    
    print(f"\n📊 详细误差统计:")
    print(f"位置误差 (mm):")
    print(f"  - RMS: {np.sqrt(np.mean(pos_errors**2)):.4f}")
    print(f"  - 最大: {np.max(np.abs(pos_errors)):.4f}")
    print(f"  - X轴: RMS={np.sqrt(np.mean(pos_errors[:, 0]**2)):.4f}, Max={np.max(np.abs(pos_errors[:, 0])):.4f}")
    print(f"  - Y轴: RMS={np.sqrt(np.mean(pos_errors[:, 1]**2)):.4f}, Max={np.max(np.abs(pos_errors[:, 1])):.4f}")
    print(f"  - Z轴: RMS={np.sqrt(np.mean(pos_errors[:, 2]**2)):.4f}, Max={np.max(np.abs(pos_errors[:, 2])):.4f}")
    
    print(f"\n姿态误差 (rad):")
    print(f"  - RMS: {np.sqrt(np.mean(orient_errors**2)):.6f}")
    print(f"  - 最大: {np.max(np.abs(orient_errors)):.6f}")
    print(f"  - Rx: RMS={np.sqrt(np.mean(orient_errors[:, 0]**2)):.6f}, Max={np.max(np.abs(orient_errors[:, 0])):.6f}")
    print(f"  - Ry: RMS={np.sqrt(np.mean(orient_errors[:, 1]**2)):.6f}, Max={np.max(np.abs(orient_errors[:, 1])):.6f}")
    print(f"  - Rz: RMS={np.sqrt(np.mean(orient_errors[:, 2]**2)):.6f}, Max={np.max(np.abs(orient_errors[:, 2])):.6f}")
    
    # 角度转换
    print(f"\n姿态误差 (度):")
    orient_errors_deg = np.degrees(orient_errors)
    print(f"  - RMS: {np.sqrt(np.mean(orient_errors_deg**2)):.4f}")
    print(f"  - 最大: {np.max(np.abs(orient_errors_deg)):.4f}")

def test_with_custom_params():
    """
    测试使用自定义标定参数的误差计算
    """
    print("\n" + "="*60)
    print("🧪 测试自定义标定参数")
    print("="*60)
    
    # 加载数据
    joint_angles, laser_matrices = load_calibration_data()
    
    # 自定义标定参数 (14个)
    # TCP偏移参数 (7个): [x, y, z, qx, qy, qz, qw]
    custom_tcp_offset = np.array([0.1731,1.1801, 238.3535, 0.4961, 0.5031, 0.505, 0.4957])  # Z方向偏移100mm
    
    # 激光基座参数 (7个): [x, y, z, qx, qy, qz, qw]  
    custom_laser_base = np.array([3610.8319,3300.7233, 13.6472, 0.0014, -0.0055, 0.7873, -0.6166])  # 位置偏移
    
    # 组合自定义标定参数
    custom_calib_params = np.concatenate([custom_tcp_offset, custom_laser_base])
    
    print(f"🔧 使用自定义标定参数:")
    print(f"  - TCP偏移: {custom_tcp_offset}")
    print(f"  - 激光基座: {custom_laser_base}")
    
    # 计算误差（DH参数使用初始值，标定参数使用自定义值）
    error_matrix, total_error = compute_all_errors(
        joint_angles, 
        laser_matrices, 
        dh_params=None,  # 使用初始DH参数
        calib_params=custom_calib_params  # 使用自定义标定参数
    )
    
    print(f"\n📊 自定义参数误差统计:")
    print(f"总误差 (加权RMS): {total_error:.6f}")
    print(f"位置误差 (RMS): {np.sqrt(np.mean(error_matrix[:, :3]**2)):.6f} mm")
    print(f"姿态误差 (RMS): {np.sqrt(np.mean(error_matrix[:, 3:]**2)):.6f} rad")

def main():
    """
    主函数：演示数据加载功能和误差计算
    """
    print("=" * 60)
    print("DH参数标定 - 数据加载与误差计算模块 (分离DH和标定参数)")
    print("=" * 60)
    
    try:
        # 加载标定数据
        joint_angles, laser_matrices = load_calibration_data()
        
        # 测试1: 使用初始参数
        print(f"\n🔍 测试1: 使用初始参数计算误差...")
        error_matrix, total_error = compute_all_errors(joint_angles, laser_matrices)
        
        print(f"\n📊 初始参数误差统计:")
        print(f"总误差 (加权RMS): {total_error:.6f}")
        print(f"位置误差 (RMS): {np.sqrt(np.mean(error_matrix[:, :3]**2)):.6f} mm")
        print(f"姿态误差 (RMS): {np.sqrt(np.mean(error_matrix[:, 3:]**2)):.6f} rad")
        
        # 详细分析
        analyze_errors(error_matrix)
        
        # 显示前3组误差
        print(f"\n🔍 前3组误差详情:")
        for i in range(min(3, error_matrix.shape[0])):
            print(f"第{i+1}组:")
            print(f"  位置误差: [{error_matrix[i, 0]:.4f}, {error_matrix[i, 1]:.4f}, {error_matrix[i, 2]:.4f}] mm")
            print(f"  姿态误差: [{error_matrix[i, 3]:.6f}, {error_matrix[i, 4]:.6f}, {error_matrix[i, 5]:.6f}] rad")
            print(f"            = [{np.degrees(error_matrix[i, 3]):.4f}, {np.degrees(error_matrix[i, 4]):.4f}, {np.degrees(error_matrix[i, 5]):.4f}] 度")
        
        # 测试2: 使用自定义标定参数
        test_with_custom_params()
        
        print(f"\n✅ 误差计算完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
