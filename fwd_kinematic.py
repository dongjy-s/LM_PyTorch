"""_summary_
    根据DH参数正运动学相关计算
Returns:
    工具变换矩阵
"""

import numpy as np
import math

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
        
        #! MDH: theta_offset, alpha, d, a
        self.modified_dh_params = [
            [0, 0, 380, 0],
            [-90, -90, 0, 30],
            [0, 0, 0, 440],
            [0, -90, 435, 35],
            [0, 90, 0, 0],
            [180, -90, 83, 0]
        ]
        
        # 工具相对于末端法兰的位姿 (位置[X,Y,Z]和四元数[Rx,Ry,Rz,W])
        self.tool_offset = {
            'position': np.array([1.081, 1.1316, 97.2485]),  # mm
            'quaternion': np.array([0.5003, 0.5012, 0.5002, 0.4983])  # Rx, Ry, Rz, W
        }
        
        # 设置打印精度，抑制科学计数法
        np.set_printoptions(precision=4, suppress=True)

    #! 根据改进DH参数计算变换矩阵
    def modified_dh_matrix(self, alpha_deg, a, d, theta_deg):
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
    
    #! 将旋转矩阵转换为欧拉角 (rx, ry, rz) (度)
    def rotation_matrix_to_euler(self, R):
        # 奇异性检查 (万向锁)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        # 转换为度
        rx = math.degrees(x)
        ry = math.degrees(y)
        rz = math.degrees(z)
        
        return np.array([rx, ry, rz])
    
    #! 四元数转旋转矩阵
    def quaternion_to_rotation_matrix(self, q):
        # 四元数 q = [x, y, z, w]
        x, y, z, w = q
        
        # 旋转矩阵公式
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R
    
    #! 计算正运动学
    def forward_kinematics(self, q_deg):
        is_valid, error_msg = self.check_joint_limits(q_deg)
        if not is_valid:
            return {
                'valid': False,
                'error_msg': error_msg,
                'transform_matrix': None,
                'position': None,
                'euler_angles': None
            }
        
        # 初始化总变换矩阵 T_0^6 为单位矩阵
        T_base_flange = np.identity(4)
        
        # 循环计算每个关节的变换矩阵并累乘
        for i in range(6):
            # 获取当前关节的改进DH参数
            theta_offset_i = self.modified_dh_params[i][0]
            alpha_i = self.modified_dh_params[i][1]
            d_i = self.modified_dh_params[i][2]
            a_i = self.modified_dh_params[i][3]

            # 获取当前关节的可变角度 q_i
            q_i = q_deg[i]

            # 计算实际的 theta_i = q_i + theta_offset_i
            theta_i = q_i + theta_offset_i

            # 计算当前关节的变换矩阵 A_i (T_(i-1)^i)
            A_i = self.modified_dh_matrix(alpha_i, a_i, d_i, theta_i)

            # 累积变换: T_0^i = T_0^(i-1) * A_i
            T_base_flange = np.dot(T_base_flange, A_i)
        
        # 提取位置和姿态
        position = T_base_flange[0:3, 3]
        rotation_matrix = T_base_flange[0:3, 0:3]
        euler_angles = self.rotation_matrix_to_euler(rotation_matrix)
        
        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_base_flange,
            'position': position,  # [x, y, z]
            'euler_angles': euler_angles  # [rx, ry, rz]
        }
    
    #! 计算工具位姿
    def calculate_tool_matrix(self, q_deg):
        # 首先计算末端法兰的位姿
        flange_result = self.forward_kinematics(q_deg)
        if not flange_result['valid']:
            return flange_result
            
        T_base_flange = flange_result['transform_matrix']
        
        # 创建工具相对于末端法兰的变换矩阵
        T_flange_tool = np.identity(4)
        
        # 设置工具偏移的位置
        T_flange_tool[0:3, 3] = self.tool_offset['position']
        
        # 设置工具偏移的旋转矩阵（由四元数转换）
        R_flange_tool = self.quaternion_to_rotation_matrix(self.tool_offset['quaternion'])
        T_flange_tool[0:3, 0:3] = R_flange_tool
        
        # 计算工具相对于基座的变换矩阵
        T_base_tool = np.dot(T_base_flange, T_flange_tool)
        
        # 提取工具位置和姿态
        tool_position = T_base_tool[0:3, 3]
        tool_rotation_matrix = T_base_tool[0:3, 0:3]
        tool_euler_angles = self.rotation_matrix_to_euler(tool_rotation_matrix)
        
        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix': T_base_tool,
            'position': tool_position, 
            'euler_angles': tool_euler_angles  
        }


if __name__ == "__main__":

    robot = RokaeRobot()  
    q_deg = [26.18229564,47.10895029,20.44052241,-143.5911443,87.23868486,-6.971798826]  
    T_base_tool = robot.calculate_tool_matrix(q_deg)
    
    tool_position = T_base_tool['position']
    tool_euler_angles = T_base_tool['euler_angles']
    tool_matrix = T_base_tool['transform_matrix']
    print(f"工具位置 (x, y, z) in mm: [{tool_position[0]:.4f}, {tool_position[1]:.4f}, {tool_position[2]:.4f}]")
    print(f"工具姿态 (rx, ry, rz) in degrees: [{tool_euler_angles[0]:.4f}, {tool_euler_angles[1]:.4f}, {tool_euler_angles[2]:.4f}]")
    print(f"工具位姿变换矩阵:\n{tool_matrix}")
