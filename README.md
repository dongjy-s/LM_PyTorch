# 机器人参数标定项目 (基于PyTorch的LM优化)

## 🎯 项目概述

本项目是一个完整的六轴工业机器人运动学参数标定解决方案，使用激光跟踪仪测量数据对机器人进行高精度标定。项目采用多种先进算法和优化策略，提供从数据采集到结果可视化的完整工作流。

### 主要标定内容
- **DH (Denavit-Hartenberg) 参数**: 修正的MDH参数 (alpha, a, d, theta_offset)
- **工具TCP (Tool Center Point) 参数**: 工具中心点相对于机器人末端法兰的位姿 
- **基座参数**: 机器人基座在激光跟踪仪坐标系下的位姿

### 核心特性
✅ **统一配置管理** - 基于YAML的配置文件系统，统一管理所有参数  
✅ **直接原始数据处理** - 无需中间转换，直接处理机器人和激光跟踪仪原始数据  
✅ **多种雅可比计算方法** - PyTorch自动微分、解析计算、JAX实现  
✅ **AX=YB标定算法** - 基于OpenCV的机器人-传感器标定  
✅ **可视化分析** - 参数收敛过程和结果可视化  
✅ **数据提取工具** - 自动化数据预处理工具链  
✅ **交替优化策略** - 处理参数间耦合的高级优化算法  
✅ **原始数据支持** - 直接处理机器人控制器和激光跟踪仪的原始数据格式

## 📁 项目结构

```
OptimizeDhParam/
├── 📂 config.yaml                    # 主配置文件 (新增)
├── 📂 data/                           # 数据文件
│   ├── joint_angle.csv               # 原始机器人关节角度数据 (控制器格式)
│   ├── laser_pos.csv                 # 原始激光跟踪仪数据 (包含名义和测量值)
│   ├── dh.csv                        # DH参数文件
│   └── joint_limits.csv              # 关节限位文件
├── 📂 tools/                          # 工具集
│   ├── data_loader.py                # 数据加载器 (支持配置和直接原始数据提取)
│   ├── calibrate.py                  # AX=YB标定算法
│   ├── plot.py                       # 参数可视化工具
│   ├── extract_joint_angle.py        # 关节角度提取工具 (独立使用)
│   └── extract_laser_pos.py          # 激光数据提取工具 (独立使用)
├── 📂 test/                           # 测试脚本
│   ├── test_error.py                 # RMSE误差分析
│   └── test_config.py                # 配置系统测试 (新增)
├── 📂 results/                        # 结果输出
│   ├── optimized_dh_parameters.csv    # 优化后DH参数
│   ├── optimized_tcp_parameters.csv   # 优化后TCP参数
│   ├── optimized_t_laser_base_parameters.csv # 优化后基座参数
│   ├── delta_values.csv              # 参数收敛过程
│   └── calibration_results.csv       # 标定结果摘要
├── 📂 graph/                          # 可视化图表
│   ├── joint1_params.png ~ joint6_params.png # 各关节参数收敛图
│   ├── tcp_params.png                # TCP参数收敛图
│   ├── laser_position.png            # 激光位置参数图
│   └── laser_orientation.png         # 激光姿态参数图
├── jacobian_torch.py                 # 核心运动学计算和雅可比矩阵
├── lm_optimize_pytorch.py            # LM优化算法主程序
├── requirements.txt                  # Python依赖
└── README.md                         # 项目文档
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone [your-repo-url]
cd OptimizeDhParam

# 安装依赖
pip install -r requirements.txt

# 可选：安装额外依赖
pip install matplotlib opencv-python jax
```

### 配置项目

1. **检查配置文件** (`config.yaml`)：
   - 确认数据文件路径正确
   - 根据需要调整误差权重
   - 设置固定参数索引（如果需要）

2. **测试配置系统**：
```bash
python test_config.py
```

### 数据准备

项目现在支持直接处理原始数据，无需手动转换：

1. **原始关节角度数据** (`data/joint_angle.csv`)：
从机器人控制器导出的原始格式：
```
LOCAL VAR jointtarget j1 = j:{	 -31.159,63.769,-46.179,109.328,-77.687,79.765,	0	}		{EJ 0,0,0,0,0,0}
LOCAL VAR jointtarget j2 = j:{	 19.087,39.061,-30.617,118.814,42.785,72.104,	0	}		{EJ 0,0,0,0,0,0}
```

2. **原始激光测量数据** (`data/laser_pos.csv`)：
激光跟踪仪输出的完整格式（包含名义值和测量值）：
```
名称 名义 X 名义 Y 名义 Z 名义 Rx 名义 Ry 名义 Rz 测量 X 测量 Y 测量 Z 测量 Rx 测量 Ry 测量 Rz 余量 位置 余量 方向
Motion1 2441.5012 1667.0849 390.6434 45.4983 -70.8809 -166.36 2441.5048 1667.1769 390.7169 45.4378 -70.8594 -166.2885 0.1178 0.0326
```

**注意**：项目会自动从原始数据中提取所需信息，无需手动预处理。

### 执行标定

```bash
# 1: 使用AX=YB标定算法
python tools/calibrate.py

# 2: 使用LM优化算法
python lm_optimize_pytorch.py

# 3：生成可视化图表
python tools/plot.py

# 4：评估标定结果
python test/test_error.py
```

## 🔧 核心功能详解

### 1. 统一数据加载器 (`tools/data_loader.py`)

新的数据加载器集成了配置管理和原始数据处理：

```python
from tools.data_loader import (
    load_config,           # 加载配置文件
    get_error_weights,     # 获取误差权重
    get_fixed_indices,     # 获取固定参数索引
    load_joint_angles,     # 直接加载关节角度（从原始数据）
    get_laser_tool_matrix, # 直接获取激光变换矩阵（从原始数据）
)

# 使用示例
config = load_config('config.yaml')  # 可选：指定配置文件路径
joint_angles = load_joint_angles()   # 自动从原始数据提取
error_weights = get_error_weights()  # 从配置获取权重
```

**主要功能**：
- 统一的配置管理接口
- 直接从原始数据提取，跳过中间处理
- 自动错误处理和数据验证
- 保持与现有代码的兼容性

### 2. 雅可比计算方法

#### PyTorch自动微分 (`jacobian_torch.py`)
- 利用PyTorch的自动微分机制，自动计算雅可比矩阵
- 计算效率高，数值稳定性好
- 适合大规模优化问题
- 集成了完整的机器人运动学正解

```python
# PyTorch雅可比使用示例
from jacobian_torch import RobotCalibration

robot = RobotCalibration()
joint_angles = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float64)  # 弧度
jacobian_matrix = robot.compute_jacobian(joint_angles)
```

**核心功能**：
- 自动微分计算雅可比矩阵
- 支持批量处理多个关节角度
- 集成DH参数、TCP参数和基座参数的优化
- 提供完整的机器人运动学正解

### 3. AX=YB标定算法 (`tools/calibrate.py`)

实现了经典的机器人手眼标定算法，用于同时求解：
- **X**: 法兰到工具的变换 (Flange → Tool)
- **Y**: 激光到基座的变换 (Laser → Base)

```python
# 使用AX=YB标定
from tools.calibrate import calibrate_AX_equals_YB

A_list = calculate_T_flange()  # 基座到法兰变换列表
B_list = tool_pos_to_transform_matrix(tool_pos_list)  # 激光到工具变换列表
X, Y, Y_inv = calibrate_AX_equals_YB(A_list, B_list)
```

**算法特点**：
- 使用OpenCV的PARK方法
- 自动处理数据预处理
- 输出详细的标定结果和统计信息

### 4. 可视化分析工具 (`tools/plot.py`)

自动生成参数收敛过程的可视化图表：

- **关节参数图**: 每个关节的α、a、d、θ_offset参数收敛过程
- **TCP参数图**: 工具中心点位置和姿态参数变化
- **基座参数图**: 激光跟踪仪与机器人基座的相对位姿

```python
# 生成可视化图表
python tools/plot.py
```

图表特性：
- 支持中文显示
- 对数坐标轴，便于观察小数值变化
- 自动保存到 `graph/` 目录

### 5. 数据提取工具

#### 关节角度提取 (`tools/extract_joint_angle.py`)
**功能特点**：
- 解析机器人控制器的原始日志格式
- 支持ABB机器人的jointtarget格式
- 自动提取前6个关节角度值
- 输出标准CSV格式

**输入格式**：
```
LOCAL VAR jointtarget j1 = j:{	 -31.159,63.769,-46.179,109.328,-77.687,79.765,	0	}
```

**输出格式**：
```csv
q1,q2,q3,q4,q5,q6
-31.159,63.769,-46.179,109.328,-77.687,79.765
```

#### 激光数据提取 (`tools/extract_laser_pos.py`)
**功能特点**：
- 处理激光跟踪仪的完整数据格式
- 自动提取测量值（非名义值）
- 支持多种激光跟踪仪输出格式
- 数据验证和清洗

**输入格式**：
```
名称 名义 X 名义 Y 名义 Z 名义 Rx 名义 Ry 名义 Rz 测量 X 测量 Y 测量 Z 测量 Rx 测量 Ry 测量 Rz
Motion1 2441.5012 1667.0849 390.6434 45.4983 -70.8809 -166.36 2441.5048 1667.1769 390.7169 45.4378 -70.8594 -166.2885
```

**输出格式**：
```csv
x,y,z,rx,ry,rz
2441.5048,1667.1769,390.7169,45.4378,-70.8594,-166.2885
```

### 6. 高级优化策略

#### 交替优化算法
```python
# 在lm_optimize_pytorch.py中配置
OPTIMIZATION_GROUPS = [
    [0, 1, 2, 3],      # 第1组：关节1的DH参数
    [4, 5, 6, 7],      # 第2组：关节2的DH参数
    [24, 25, 26],      # 第3组：TCP位置参数
    [27, 28, 29, 30],  # 第4组：TCP姿态参数
]
```

#### 参数固定策略
```python
# 固定某些参数不参与优化
ALL_FIXED_INDICES = [0, 4, 8]  # 固定某些关节的α参数
```

## 📊 结果分析

### RMSE误差评估

```bash
python test/test_error.py
```

输出示例：
```
=== 参数标定结果对比 ===
初始参数 RMSE: 15.423 mm
优化后参数 RMSE: 2.186 mm
改善程度: 86.2%

位置误差: 1.832 mm
姿态误差: 0.354°
```

### 参数收敛分析

查看 `graph/` 目录中的可视化图表：
- 参数是否收敛到稳定值
- 收敛速度和振荡情况
- 不同参数组的优化效果

## ⚙️ 配置系统

### 配置文件结构 (`config.yaml`)

项目使用统一的YAML配置文件来管理所有参数，无需修改代码即可调整配置：

```yaml
# 数据文件路径配置
data_files:
  joint_angle_raw: 'data/joint_angle.csv'       # 原始关节角度文件
  laser_pos_raw: 'data/laser_pos.csv'           # 原始激光位置文件
  dh_params: 'data/dh.csv'                      # DH参数文件
  joint_limits: 'data/joint_limits.csv'         # 关节限位文件
  calibration_results: 'results/calibration_results.csv'  # 校准结果文件

# 机器人配置参数
robot_config:
  error_weights: [1.0, 1.0, 1.0, 0.01, 0.01, 0.01]  # 误差权重
  fixed_indices: []                              # 固定参数索引 (空=全优化)

# 数据提取配置
extraction:
  joint_angle:
    pattern: 'j:\s*{\s*([^}]*?)\s*}'            # 正则表达式模式
    num_joints: 6                               # 关节数量
    encoding: 'utf-8'                           # 文件编码
  
  laser_pos:
    skip_header: true                           # 跳过表头
    measurement_columns: [7, 8, 9, 10, 11, 12] # 测量数据列索引
    delimiter: ' '                              # 数据分隔符
    encoding: 'utf-8'                           # 文件编码

# 优化配置
optimization:
  max_iterations: 1000                          # 最大迭代次数
  convergence_threshold: 1e-8                   # 收敛阈值
  verbose: true                                 # 详细日志
```

### 配置系统特性

**智能配置加载**：
- 自动加载 `config.yaml` 配置文件
- 如果配置文件不存在，自动使用内置默认配置
- 支持运行时动态修改配置

**参数固定功能**：
```yaml
# 示例：固定前3个DH参数和所有基座参数
robot_config:
  fixed_indices: [0, 1, 2, 30, 31, 32, 33, 34, 35, 36, 37]
```

**数据格式配置**：
- 支持不同机器人控制器的数据格式
- 可配置激光跟踪仪数据列映射
- 支持自定义分隔符和编码

## 📝 数据格式说明

### 原始数据源

#### 1. 机器人控制器数据格式
- **文件**: `data/joint_angle.csv`
- **来源**: ABB机器人控制器日志
- **格式**: jointtarget结构，包含关节角度和外部轴信息
- **提取内容**: 前6个关节角度值（度）

#### 2. 激光跟踪仪数据格式
- **文件**: `data/laser_pos.csv`
- **来源**: 激光跟踪仪测量报告
- **格式**: 包含名义位置、测量位置和误差信息
- **提取内容**: 测量位置和姿态（X,Y,Z,Rx,Ry,Rz）

### 标准化数据格式

#### 1. 关节角度数据
- **文件**: `data/extracted_joint_angles.csv`
- **格式**: CSV，表头为q1,q2,q3,q4,q5,q6
- **单位**: 度（°）
- **说明**: 每行对应一个采样点的6个关节角度

#### 2. 激光测量数据
- **文件**: `data/extracted_laser_positions.csv`
- **格式**: CSV，表头为x,y,z,rx,ry,rz
- **单位**: 位置mm，姿态度（°）
- **说明**: 工具在激光跟踪仪坐标系下的6D位姿

## 🔍 常见问题排查

### 1. 数据提取问题
- **症状**: 提取工具无法正确解析原始数据
- **解决**: 检查原始数据格式，确保与工具预期格式一致

### 2. 数据对应问题
- **症状**: 关节角度和激光测量数据不匹配
- **解决**: 确保两个数据文件的行数相同且采样顺序一致

### 3. 收敛问题
- **症状**: 优化不收敛或发散
- **解决**: 调整初始参数、增加阻尼因子、检查数据质量

### 4. 数据格式问题
- **症状**: 读取数据时报错
- **解决**: 检查CSV文件格式、确保数据对应关系

### 5. 内存不足
- **症状**: 大数据集时内存溢出
- **解决**: 减小批处理大小、使用数据分块处理

### 6. 数值不稳定
- **症状**: 计算结果异常或NaN
- **解决**: 检查雅可比矩阵条件数、调整权重参数

## 🎯 最佳实践

### 数据采集建议
1. **采样点分布**: 在工作空间内均匀分布，避免奇异位形
2. **数据数量**: 建议不少于30组有效数据点
3. **运动范围**: 覆盖各关节的主要工作范围
4. **数据质量**: 确保激光跟踪仪测量精度和机器人位置重复性

### 数据处理建议
1. **原始数据检查**: 使用提取工具前先检查原始数据格式
2. **数据验证**: 提取后检查数据的合理性和完整性
3. **数据对应**: 确保关节角度和激光测量数据一一对应

### 优化策略建议
1. **初始值设置**: 使用机器人名义参数作为初始值
2. **参数分组**: 将强耦合参数分组优化
3. **权重调整**: 根据测量精度调整位置和姿态权重

### 结果验证建议
1. **交叉验证**: 使用独立数据集验证标定结果
2. **物理检验**: 通过实际运动验证标定精度
3. **长期监控**: 定期检查标定参数的稳定性

## 📝 更新日志

### v2.1.0 (最新)
- ✨ 新增原始数据源支持（机器人控制器和激光跟踪仪原始格式）
- ✨ 新增数据提取工具链（关节角度和激光数据提取）
- 📚 完善数据格式说明和使用指南
- 🔧 优化数据处理流程

### v2.0.0
- ✨ 新增AX=YB标定算法
- ✨ 新增参数可视化功能
- ✨ 新增多种雅可比计算方法
- ✨ 新增数据提取工具链
- 🎨 优化项目结构和代码组织
- 📚 完善文档和使用说明

### v1.0.0 (基础版本)
- 🎯 基础LM优化算法
- 🔧 PyTorch自动微分
- 📊 基本误差评估



