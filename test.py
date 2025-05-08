import numpy as np
import math

#抑制科学计数法
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind': lambda x: f"{x:.{4}f}"})

class RokaeRobot:
    #! 初始化
    def __init__(self):
        # 关节限位 [min, max] (单位:度)
        self.joint_limits = [
            [-170, 170], 
            [-96, 130],   
            [-195, 65],   
            [-179, 170],  
            [-95, 95],    
            [-180, 180]   
        ]
        
        # 改进DH参数: [theta_offset_i, alpha_i, d_i, a_i]
        self.modified_dh_params = [
            [0,   0,   380,   0],    
            [-90, -90,  0,    30],    
            [0,   0,    0,    440],   
            [0,   -90,  435,  35],    
            [0,   90,   0,    0],    
            [180, -90,  83,   0]    
        ]


    #! 根据改进DH参数计算变换矩阵
    def modified_dh_matrix(self, theta_deg, alpha_deg, d, a):
        # 将角度从度转换为弧度
        alpha_rad = math.radians(alpha_deg)
        theta_rad = math.radians(theta_deg)

        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        cos_alpha = math.cos(alpha_rad)
        sin_alpha = math.sin(alpha_rad)

        # 改进DH变换矩阵公式
        A = np.array([
            [cos_theta, -sin_theta, 0, a],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
            [0, 0, 0, 1]
        ])
        return A
    
    #! 检查关节角度是否在限位范围内
    def check_joint_limits(self, q_deg):
        for i, q in enumerate(q_deg):
            if q < self.joint_limits[i][0] or q > self.joint_limits[i][1]:
                return False, f"关节{i+1}角度 {q}° 超出限位范围 [{self.joint_limits[i][0]}°, {self.joint_limits[i][1]}°]"
        return True, ""

    #! 构建局部雅可比矩阵
    def build_local_jacobian(self, joint_index, q_deg):
        # 获取DH参数
        theta_offset = self.modified_dh_params[joint_index][0]
        alpha_deg = self.modified_dh_params[joint_index][1]
        d = self.modified_dh_params[joint_index][2]
        a = self.modified_dh_params[joint_index][3]
        
        # 计算实际旋转角度 = 输入角度 + 偏移量
        actual_theta = q_deg + theta_offset

        # 使用实际角度计算变换矩阵
        A = self.modified_dh_matrix(actual_theta, alpha_deg, d, a)
        A_inv = np.linalg.inv(A)  
        
        # 角度到弧度的转换因子
        deg2rad = math.pi / 180.0
        
        # 计算基本三角函数值
        alpha_rad = math.radians(alpha_deg)
        theta_rad = math.radians(actual_theta)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        cos_alpha = math.cos(alpha_rad)
        sin_alpha = math.sin(alpha_rad)
        
        # 对theta_deg的偏导数 
        dA_dtheta = np.array([
            [-sin_theta, -cos_theta, 0, 0],
            [cos_theta*cos_alpha, -sin_theta*cos_alpha, 0, 0],
            [cos_theta*sin_alpha, -sin_theta*sin_alpha, 0, 0],
            [0, 0, 0, 0]
        ]) * deg2rad
        
        # 对alpha_deg的偏导数 
        dA_dalpha = np.array([
            [0, 0, 0, 0],
            [-sin_theta*sin_alpha, -cos_theta*sin_alpha, -cos_alpha, -cos_alpha*d],
            [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d],
            [0, 0, 0, 0]
        ]) * deg2rad
        
        # 对d的偏导数
        dA_dd = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, -sin_alpha],
            [0, 0, 0, cos_alpha],
            [0, 0, 0, 0]
        ])
        
        # 对a的偏导数
        dA_da = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        # 计算各参数贡献矩阵
        delta_theta_contrib = A_inv @ dA_dtheta
        delta_alpha_contrib = A_inv @ dA_dalpha
        delta_d_contrib = A_inv @ dA_dd
        delta_a_contrib = A_inv @ dA_da
        
        # 从每个贡献矩阵中提取位移和旋转元素
        # 位移部分 (dx, dy, dz)
        dx_theta = delta_theta_contrib[0, 3]
        dy_theta = delta_theta_contrib[1, 3]
        dz_theta = delta_theta_contrib[2, 3]
        rx_theta = delta_theta_contrib[2, 1]
        ry_theta = delta_theta_contrib[0, 2]
        rz_theta = delta_theta_contrib[1, 0]
        
        dx_alpha = delta_alpha_contrib[0, 3]
        dy_alpha = delta_alpha_contrib[1, 3]
        dz_alpha = delta_alpha_contrib[2, 3]
        rx_alpha = delta_alpha_contrib[2, 1]
        ry_alpha = delta_alpha_contrib[0, 2]
        rz_alpha = delta_alpha_contrib[1, 0]
        
        dx_d = delta_d_contrib[0, 3]
        dy_d = delta_d_contrib[1, 3]
        dz_d = delta_d_contrib[2, 3]
        rx_d = delta_d_contrib[2, 1]
        ry_d = delta_d_contrib[0, 2]
        rz_d = delta_d_contrib[1, 0]
        
        dx_a = delta_a_contrib[0, 3]
        dy_a = delta_a_contrib[1, 3]
        dz_a = delta_a_contrib[2, 3]
        rx_a = delta_a_contrib[2, 1]
        ry_a = delta_a_contrib[0, 2]
        rz_a = delta_a_contrib[1, 0]
        
        # 构建雅可比矩阵 (6x4)，每列对应一个参数的贡献
        local_jacobian = np.array([
            [dx_theta, dx_alpha, dx_d, dx_a],  
            [dy_theta, dy_alpha, dy_d, dy_a],  
            [dz_theta, dz_alpha, dz_d, dz_a],  
            [rx_theta, rx_alpha, rx_d, rx_a],  
            [ry_theta, ry_alpha, ry_d, ry_a],  
            [rz_theta, rz_alpha, rz_d, rz_a]   
        ])
        
        return local_jacobian
    
    #! 正运动学计算
    def forward_kinematics(self, q_deg_array):

        T_total = np.eye(4) 
        for i in range(len(self.modified_dh_params)):
            theta_offset_i = self.modified_dh_params[i][0]
            alpha_i = self.modified_dh_params[i][1]
            d_i = self.modified_dh_params[i][2]
            a_i = self.modified_dh_params[i][3]
            
            actual_theta_i = q_deg_array[i] + theta_offset_i # 实际关节角
            
            A_i = self.modified_dh_matrix(actual_theta_i, alpha_i, d_i, a_i)
            T_total = T_total @ A_i
            
        return T_total

    #! 创建反对称矩阵
    def skew_symmetric(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    #! 计算伴随变换矩阵
    def adjoint_transform(self, T):
        # 提取旋转矩阵R
        R = T[0:3, 0:3]
        
        # 提取平移向量p
        p = T[0:3, 3]
        
        # 计算p的反对称矩阵
        p_skew = self.skew_symmetric(p)
        
        # 构建伴随变换矩阵
        Ad_T = np.zeros((6, 6))
        Ad_T[0:3, 0:3] = R
        Ad_T[0:3, 3:6] = p_skew @ R
        Ad_T[3:6, 3:6] = R
        
        return Ad_T
    
    #! 计算从基座到每个关节的变换矩阵
    def compute_transforms_to_joints(self, q_deg_array):
        transforms = []
        T_0_i = np.eye(4)  # 初始化为单位矩阵（基座到基座的变换）
        
        for i in range(len(self.modified_dh_params)):
            theta_offset_i = self.modified_dh_params[i][0]
            alpha_i = self.modified_dh_params[i][1]
            d_i = self.modified_dh_params[i][2]
            a_i = self.modified_dh_params[i][3]
            
            actual_theta_i = q_deg_array[i] + theta_offset_i  # 实际关节角
            
            A_i = self.modified_dh_matrix(actual_theta_i, alpha_i, d_i, a_i)
            T_0_i = T_0_i @ A_i  # 累积变换
            
            transforms.append(np.copy(T_0_i))
            
        return transforms
    
    #! 构建完整的雅可比矩阵（J_N或J_body）
    def build_jacobian_matrix(self, q_deg_array):
        # 检查关节角度是否在限位范围内
        valid, message = self.check_joint_limits(q_deg_array)
        if not valid:
            raise ValueError(message)
        
        # 初始化6x24雅可比矩阵
        J_N = np.zeros((6, 24))
        
        # 计算从基座到每个关节的变换矩阵
        transforms = self.compute_transforms_to_joints(q_deg_array)
        
        # 对于每个关节
        for i in range(6):
            # 获取局部雅可比矩阵（Mi）
            M_i = self.build_local_jacobian(i, q_deg_array[i])
            
            # 计算伴随变换矩阵
            Ad_T_0_i = self.adjoint_transform(transforms[i])
            
            # 计算贡献块（局部雅可比矩阵在基坐标系下的表示）
            J_block_i = Ad_T_0_i @ M_i
            
            # 将贡献块填充到完整雅可比矩阵的对应位置
            col_start = i * 4
            J_N[:, col_start:col_start+4] = J_block_i
            
        return J_N

if __name__ == "__main__":
    robot = RokaeRobot()
    
    # 测试完整雅可比矩阵构建
    # 定义一组关节角度（度）
    test_angles = [0, 0, 0, 0, 0, 0]  # 零位姿态
    
    # 构建完整雅可比矩阵
    J_N = robot.build_jacobian_matrix(test_angles)
    
    # 打印雅可比矩阵的形状和内容
    print("完整雅可比矩阵 J_N 形状:", J_N.shape)
    print("完整雅可比矩阵 J_N:")
    print(J_N)
    
    # 测试雅可比矩阵的使用 - 模拟DH参数误差对末端位姿的影响
    # 创建一个假定的DH参数误差向量 (24x1)
    dq = np.zeros(24)
    # 假设第1个关节的theta参数有1度误差
    dq[0] = 1.0  
    # 假设第3个关节的d参数有1mm误差
    dq[10] = 1.0  
    
    # 计算这些误差导致的末端执行器位姿误差
    dX = J_N @ dq
    
    print("\n参数误差向量 dq:")
    print(dq)
    print("\n导致的末端执行器位姿误差 dX:")
    print(f"位移误差 (dx, dy, dz): [{dX[0]:.4f}, {dX[1]:.4f}, {dX[2]:.4f}] mm")
    print(f"旋转误差 (rx, ry, rz): [{dX[3]:.4f}, {dX[4]:.4f}, {dX[5]:.4f}] rad")
