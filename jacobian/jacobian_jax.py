import jax
import jax.numpy as jnp
import numpy as np

def jnp_modified_dh_matrix(theta_val_rad, alpha_val_rad, d_val, a_val):
    cos_theta = jnp.cos(theta_val_rad)
    sin_theta = jnp.sin(theta_val_rad)
    cos_alpha = jnp.cos(alpha_val_rad)
    sin_alpha = jnp.sin(alpha_val_rad)
    A = jnp.array([
        [cos_theta, -sin_theta, 0, a_val],
        [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -sin_alpha*d_val],
        [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d_val],
        [0, 0, 0, 1]
    ])
    return A


def jnp_forward_kinematics_T_for_dh_sensitivity(all_dh_params_flat_jnp, q_deg_array_fixed):
    T_total_jnp = jnp.eye(4)
    num_joints = 6
    for i in range(num_joints):
        base_idx = i * 4
        theta_offset_i = all_dh_params_flat_jnp[base_idx + 0]
        alpha_i_deg    = all_dh_params_flat_jnp[base_idx + 1]
        d_i            = all_dh_params_flat_jnp[base_idx + 2]
        a_i            = all_dh_params_flat_jnp[base_idx + 3]
        q_i_deg = q_deg_array_fixed[i]
        actual_theta_i_deg = q_i_deg + theta_offset_i
        actual_theta_i_rad = jnp.radians(actual_theta_i_deg)
        alpha_i_rad        = jnp.radians(alpha_i_deg)
        A_i_jnp = jnp_modified_dh_matrix(actual_theta_i_rad, alpha_i_rad, d_i, a_i)
        T_total_jnp = T_total_jnp @ A_i_jnp
    return T_total_jnp

def jnp_extract_pose_from_T(T_jnp, euler_convention='zyx'):
    position_jnp = T_jnp[0:3, 3]
    R_jnp = T_jnp[0:3, 0:3]
    if euler_convention == 'zyx':
        r11, r12, r13 = R_jnp[0,0], R_jnp[0,1], R_jnp[0,2]
        r21, r22, r23 = R_jnp[1,0], R_jnp[1,1], R_jnp[1,2]
        r31, r32, r33 = R_jnp[2,0], R_jnp[2,1], R_jnp[2,2]
        
        # 计算pitch角，这是ZYX欧拉角的核心，可能导致万向锁问题
        ry = jnp.arcsin(-r31)
        cos_ry = jnp.cos(ry)
        
        # 使用条件计算避免除零问题，处理奇异点
        # 当cos_ry接近零时，我们处于万向锁奇异点，rx和rz不能独立确定
        rz = jnp.where(jnp.abs(cos_ry) > 1e-6, jnp.arctan2(r21, r11), 0.0)
        rx = jnp.where(jnp.abs(cos_ry) > 1e-6, jnp.arctan2(r32, r33), 0.0)
        
        euler_angles_jnp = jnp.array([rx, ry, rz])
    else:
        raise ValueError("Unsupported Euler angle convention")
    return jnp.concatenate([position_jnp, euler_angles_jnp])

def final_fk_for_dh_sensitivity(all_dh_params_flat_jnp, fixed_joint_angles):
    T_total_jnp = jnp_forward_kinematics_T_for_dh_sensitivity(all_dh_params_flat_jnp, fixed_joint_angles)
    pose_vector_jnp = jnp_extract_pose_from_T(T_total_jnp)
    return pose_vector_jnp

if __name__ == "__main__":

    # 示例关节角度
    q_deg_array_current_fixed = np.array([42.91441824,-0.414388123,49.04196013,-119.3252973,78.65535552,-5.225972875])
    # 示例DH参数
    initial_dh_params_flat_list = [0, 0, 380, 0, -90, -90, 0, 30, 0, 0, 0, 440, 0, -90, 435, 35, 0, 90, 0, 0, 180, -90, 83, 0]
    current_dh_params_jnp = jnp.array(initial_dh_params_flat_list, dtype=jnp.float64)
    
    # 包装函数
    def compute_pose_from_dh(dh_params_jnp):
        return final_fk_for_dh_sensitivity(dh_params_jnp, q_deg_array_current_fixed)
    
    # JIT加速版本

    jit_jacobian_6x24_calculator = jax.jit(jax.jacfwd(compute_pose_from_dh))
    jacobian_matrix_6x24_jnp_jit = jit_jacobian_6x24_calculator(current_dh_params_jnp)
    jax_np_jacobian = np.array(jacobian_matrix_6x24_jnp_jit)
    np.savetxt("data/jacobian_jax_result.csv", jax_np_jacobian, delimiter=',', fmt='%.12f')
    print(f"雅可比矩阵已保存到文件: data/jacobian_jax_result.csv")


    T_totle = jnp_forward_kinematics_T_for_dh_sensitivity(current_dh_params_jnp, q_deg_array_current_fixed)
    print(T_totle)
