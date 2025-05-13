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
                [-100, 100],
                [-90, 100],
                [-100, 600],
                [-100, 100],
                [-90, 90],
                [-120, 120]  
        ]
        
        #! MDH: theta_offset, alpha, d, a
        self.modified_dh_params = [
            [0, 0, 487, 0],
            [-90, -90, 0, 85],
            [0, 0, 0, 640],
            [0, -90, 720, 205],
            [0, 90, 0, 0],
            [180, -90, 75, 0]
        ]
        
        # 工具相对于末端法兰的位姿 (位置[X,Y,Z]和四元数[Rx,Ry,Rz,W])
        self.tool_offset = {
            'position': np.array([0.1731, 1.1801, 238.3535]),  # mm
            'quaternion': np.array([0.4961, 0.5031, 0.505, 0.4957])  # Rx, Ry, Rz, W
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

    #! 计算齐次变换矩阵的逆
    def inverse_homogeneous_transform(self, T):
        """
        计算4x4齐次变换矩阵的逆。
        T = [[R, p], [0, 1]] => T_inv = [[R.T, -R.T * p], [0, 1]]
        """
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        R_inv = R.T
        p_inv = -np.dot(R_inv, p)
        
        T_inv = np.identity(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = p_inv
        return T_inv

    #! 计算工具在激光跟踪仪坐标系下的位姿
    def calculate_tool_pose_in_tracker_frame(self, q_deg, base_pose_in_tracker_xyz_quat):
        """
        计算工具在激光跟踪仪坐标系下的位姿。
        参数:
            q_deg: 机器人关节角度 (度)
            base_pose_in_tracker_xyz_quat: 基座相对于激光跟踪仪的位姿 [x, y, z, qx, qy, qz, qw]
                                        其中 x,y,z 单位为 mm, 四元数为 [qx, qy, qz, qw]
        返回:
            包含位姿信息的字典
        """
        # 1. 计算工具相对于基座的变换矩阵 T_base_tool
        tool_result_base = self.calculate_tool_matrix(q_deg)
        if not tool_result_base['valid']:
            return tool_result_base # 如果无效，直接返回错误信息
        T_base_tool = tool_result_base['transform_matrix']

        # 2. 构建基座在激光跟踪仪坐标系下的变换矩阵 T_tracker_base
        #    因为输入直接是 T_tracker_base 的参数
        base_pos_in_tracker_mm = np.array(base_pose_in_tracker_xyz_quat[0:3])
        # 四元数顺序: qx, qy, qz, qw. quaternion_to_rotation_matrix 期望 x, y, z, w
        base_quat_in_tracker_xyzw = np.array([base_pose_in_tracker_xyz_quat[3],  # qx -> x
                                              base_pose_in_tracker_xyz_quat[4],  # qy -> y
                                              base_pose_in_tracker_xyz_quat[5],  # qz -> z
                                              base_pose_in_tracker_xyz_quat[6]]) # qw -> w
        
        R_tracker_base = self.quaternion_to_rotation_matrix(base_quat_in_tracker_xyzw)
        
        T_tracker_base = np.identity(4)
        T_tracker_base[0:3, 0:3] = R_tracker_base
        T_tracker_base[0:3, 3] = base_pos_in_tracker_mm

        # 3. 计算工具在激光跟踪仪坐标系下的变换矩阵: T_tracker_tool = T_tracker_base * T_base_tool
        T_tracker_tool = np.dot(T_tracker_base, T_base_tool)

        # 4. 从 T_tracker_tool 中提取位置和欧拉角
        position_in_tracker = T_tracker_tool[0:3, 3]
        rotation_matrix_in_tracker = T_tracker_tool[0:3, 0:3]
        euler_angles_in_tracker = self.rotation_matrix_to_euler(rotation_matrix_in_tracker)
        
        return {
            'valid': True,
            'error_msg': '',
            'transform_matrix_in_tracker': T_tracker_tool,
            'position_in_tracker': position_in_tracker,      # [x, y, z] in tracker frame (mm)
            'euler_angles_in_tracker': euler_angles_in_tracker # [rx, ry, rz] in tracker frame (degrees)
        }


if __name__ == "__main__":

    robot = RokaeRobot()  
    q_deg = [2.636115843805,29.5175455057437,6.10038158777613,34.7483282137072,-47.3762053517954,35.7916251381285]  
    T_base_tool = robot.calculate_tool_matrix(q_deg)
    
    tool_position = T_base_tool['position']
    tool_euler_angles = T_base_tool['euler_angles']
    tool_matrix = T_base_tool['transform_matrix']
    print(f"工具位置 (x, y, z) in mm: [{tool_position[0]:.4f}, {tool_position[1]:.4f}, {tool_position[2]:.4f}]")
    print(f"工具姿态 (rx, ry, rz) in degrees: [{tool_euler_angles[0]:.4f}, {tool_euler_angles[1]:.4f}, {tool_euler_angles[2]:.4f}]")
    print(f"工具位姿变换矩阵:\n{tool_matrix}")
    print("位置和欧拉角为:")
    print(f"位置: {tool_position}")
    print(f"欧拉角: {tool_euler_angles}")

    print("\n" + "="*30 + "\n") # 分隔符

    # 定义基座相对于激光跟踪仪的位姿 [x, y, z, qx, qy, qz, qw]
    # x,y,z 单位 mm; qx,qy,qz,qw 为四元数分量
    base_pose_in_tracker = [3610.8319, 3300.7233, 13.6472, 0.0014, -0.0055, 0.7873, -0.6166]
    
    # 计算工具在激光跟踪仪坐标系下的位姿
    tool_pose_in_tracker_result = robot.calculate_tool_pose_in_tracker_frame(q_deg, base_pose_in_tracker)

    if tool_pose_in_tracker_result['valid']:
        pos_in_tracker = tool_pose_in_tracker_result['position_in_tracker']
        euler_in_tracker = tool_pose_in_tracker_result['euler_angles_in_tracker']
        matrix_in_tracker = tool_pose_in_tracker_result['transform_matrix_in_tracker']
        
        print("工具在激光跟踪仪坐标系下的位姿:")
        print(f"  位置 (x, y, z) in mm: [{pos_in_tracker[0]:.4f}, {pos_in_tracker[1]:.4f}, {pos_in_tracker[2]:.4f}]")
        print(f"  姿态 (rx, ry, rz) in degrees: [{euler_in_tracker[0]:.4f}, {euler_in_tracker[1]:.4f}, {euler_in_tracker[2]:.4f}]")
        print(f"  变换矩阵:\\n{matrix_in_tracker}")
    else:
        print(f"计算工具在激光跟踪仪坐标系下的位姿失败: {tool_pose_in_tracker_result['error_msg']}")

