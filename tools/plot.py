import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用雅黑，如果找不到则使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 创建graph目录（如果不存在）
if not os.path.exists('graph'):
    os.makedirs('graph')

# 读取CSV文件
csv_path = 'results/delta_values.csv'
df = pd.read_csv(csv_path)

# 创建折线图函数
def create_line_plot(data_dict, title, filename):
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 获取迭代次数作为x轴
    iterations = range(len(next(iter(data_dict.values()))))
    
    # 绘制每个参数
    for param_idx, (param_name, values) in enumerate(data_dict.items()):
        plt.plot(iterations, values, marker='.', label=param_name)
    
    # 设置图表
    plt.title(title, size=15)
    plt.xlabel('迭代次数')
    plt.ylabel('参数值 (log scale)')
    plt.yscale('symlog')
    plt.grid(True)
    plt.legend(loc='best')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join('graph', filename))
    plt.close()

# 为每个关节创建折线图
for joint_num in range(1, 7):
    # 提取关节参数
    joint_prefix = f'Joint{joint_num}_'
    joint_params = {
        'alpha': df[f'{joint_prefix}alpha'],
        'a': df[f'{joint_prefix}a'],
        'd': df[f'{joint_prefix}d'],
        'theta_offset': df[f'{joint_prefix}theta_offset']
    }
    
    # 创建并保存折线图
    create_line_plot(
        joint_params,
        f'关节 {joint_num} 参数',
        f'joint{joint_num}_params.png'
    )

# 为TCP创建折线图
tcp_params = {
    'tx': df['TCP_tx'],
    'ty': df['TCP_ty'],
    'tz': df['TCP_tz'],
    'qx': df['TCP_qx'],
    'qy': df['TCP_qy'],
    'qz': df['TCP_qz'],
    'qw': df['TCP_qw']
}
create_line_plot(tcp_params, 'TCP 参数', 'tcp_params.png')

# 为Laser位置创建折线图
laser_position = {
    'tx': df['Laser_tx'],
    'ty': df['Laser_ty'],
    'tz': df['Laser_tz']
}
create_line_plot(laser_position, 'Laser 位置', 'laser_position.png')

# 为Laser姿态创建折线图
laser_orientation = {
    'qx': df['Laser_qx'],
    'qy': df['Laser_qy'],
    'qz': df['Laser_qz'],
    'qw': df['Laser_qw']
}
create_line_plot(laser_orientation, 'Laser 姿态', 'laser_orientation.png')

print("所有图表已生成并保存到graph目录")
