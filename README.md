# 🤖 机器人DH参数标定与优化系统

**一站式六轴工业机器人运动学参数高精度标定解决方案**

基于激光跟踪仪测量数据，采用先进的LM算法和交替优化策略，实现DH参数、TCP参数和基座参数的联合标定与优化。

## ✨ 核心特性

🚀 **智能优化算法**
- **连续失败提前结束**：新增智能失败检测，连续失败3次自动结束，避免无效计算
- **交替优化策略**：分组优化处理参数耦合，收敛更稳定
- **自适应阻尼调节**：Nielsen策略动态调整，数值稳定性更佳

🔬 **工业级精度**
- **PyTorch自动微分**：高精度雅可比矩阵计算，避免数值误差
- **SVD数值求解**：增强矩阵求解稳定性，处理病态系统
- **AX=YB手眼标定**：OpenCV PARK算法，同时求解TCP和基座参数

⚡ **开箱即用**
- **原始数据直接处理**：支持机器人控制器和激光跟踪仪原始格式
- **统一配置管理**：YAML配置文件，无需修改代码
- **自动可视化**：参数收敛过程图表，结果一目了然

## 🚀 快速开始

### 1️⃣ 环境准备
```bash
git clone <repository-url>
cd OptimizeDhParam
pip install -r requirements.txt
```

### 2️⃣ 数据准备
将数据文件放入对应目录：
- `data/joint_angle.csv` - 机器人关节角度数据
- `data/laser_pos.csv` - 激光跟踪仪测量数据

### 3️⃣ 一键运行
```bash
python main.py
```

**就这么简单！** 系统将自动完成：
- ✅ 数据加载与预处理
- ✅ AX=YB初始标定  
- ✅ LM精细优化
- ✅ 结果保存与可视化

## 🏗️ 项目架构

### 📁 目录结构
```
OptimizeDhParam/
├── 🎯 main.py                    # 主程序入口
├── 📂 src/                       # 核心算法
│   ├── lm_optimize_pytorch.py    # LM优化引擎（含连续失败机制）
│   └── jacobian_torch.py         # 雅可比计算与正向运动学
├── 📂 tools/                     # 工具集
│   ├── data_loader.py           # 数据加载器
│   ├── calibrate.py             # AX=YB标定算法
│   └── plot.py                  # 可视化工具
├── 📂 config/                    # 配置管理
│   └── config.yaml              # 主配置文件
├── 📂 data/                      # 数据文件
├── 📂 results/                   # 输出结果
└── 📂 graph/                     # 可视化图表
```

### 🔄 算法流程
```
数据加载 → AX=YB标定 → LM精细优化 → 结果输出
    ↓           ↓           ↓           ↓
原始数据     初始参数    最优参数    CSV+图表
```

### 🧩 核心模块功能

**🎯 main.py** - 主控制器
- 协调整个标定流程
- 统一异常处理和日志输出
- 参数提取和结果展示

**🔧 src/lm_optimize_pytorch.py** - 优化引擎
- 交替优化策略（两组参数分别优化）
- 连续失败提前结束机制（NEW!）
- SVD增强数值稳定性
- 自适应阻尼调节

**📐 src/jacobian_torch.py** - 运动学计算
- PyTorch自动微分雅可比矩阵
- MDH正向运动学
- 四元数与旋转矩阵转换

**📊 tools/calibrate.py** - 手眼标定
- AX=YB问题求解
- OpenCV PARK算法
- TCP和基座参数初始估计

## 🛠️ 详细使用指南

### 数据格式要求

#### 机器人关节角度数据 (`data/joint_angle.csv`)
**原始格式**（直接从控制器导出）：
```
LOCAL VAR jointtarget j1 = j:{	 -31.159,63.769,-46.179,109.328,-77.687,79.765,	0	}
LOCAL VAR jointtarget j2 = j:{	 19.087,39.061,-30.617,118.814,42.785,72.104,	0	}
```

#### 激光测量数据 (`data/laser_pos.csv`)
**原始格式**（激光跟踪仪输出）：
```
名称 名义X 名义Y 名义Z 名义Rx 名义Ry 名义Rz 测量X 测量Y 测量Z 测量Rx 测量Ry 测量Rz
Motion1 2441.5 1667.1 390.6 45.5 -70.9 -166.4 2441.5 1667.2 390.7 45.4 -70.9 -166.3
```

### 配置优化参数

编辑 `config/config.yaml` 调整优化行为：

```yaml
# 关键配置项
optimization:
  lm_optimization:
    alternate_optimization:
      max_alt_iterations: 4           # 交替优化循环次数
      max_sub_iterations_group1: 10   # 第一组参数最大迭代次数
      max_sub_iterations_group2: 10   # 第二组参数最大迭代次数
    
    damping:
      lambda_init_group1: 2.0         # 第一组初始阻尼因子
      lambda_init_group2: 0.001       # 第二组初始阻尼因子
      lambda_max: 1e8                 # 最大阻尼阈值
    
    constraints:
      max_theta_change_degrees: 1.0   # theta参数单步最大变化
      enable_quaternion_normalization: true

# 误差权重（位置:姿态 = 1:0.01）
robot_config:
  error_weights: [1.0, 1.0, 1.0, 0.01, 0.01, 0.01]
  
# 固定参数（可选）
  fixed_indices: []  # 空列表=全参数优化
```

### 查看优化结果

运行完成后检查以下文件：

**📊 数值结果**：
- `results/optimized_dh_parameters.csv` - 优化后DH参数
- `results/optimized_tcp_parameters.csv` - 优化后TCP参数  
- `results/optimized_t_laser_base_parameters.csv` - 优化后基座参数

**📈 可视化图表**：
- `graph/joint1_params.png` ~ `joint6_params.png` - 各关节参数收敛过程
- `graph/tcp_params.png` - TCP参数收敛图
- `graph/laser_position.png` - 激光位置参数图

## 🧠 技术详解

### 1. 连续失败提前结束机制（NEW! 🆕）

**问题**：传统LM算法可能出现连续多次拒绝更新，导致无效计算。

**解决方案**：
- 监控连续失败次数
- 连续失败3次自动结束优化
- 避免阻尼因子过度增长

```python
# 核心代码逻辑
consecutive_failures = 0
max_consecutive_failures = 3

if rho > rho_threshold:  # 接受更新
    consecutive_failures = 0  # 重置计数
else:  # 拒绝更新  
    consecutive_failures += 1
    if consecutive_failures >= max_consecutive_failures:
        print(f"连续失败 {consecutive_failures} 次，提前结束优化")
        return params
```

### 2. 交替优化策略

**第一组**：DH参数 + TCP参数 + 激光位置参数（31个参数）
**第二组**：激光姿态四元数参数（4个参数）

这种分组避免了参数间的强耦合，提高收敛稳定性。

### 3. PyTorch自动微分雅可比

相比数值微分，自动微分具有：
- **高精度**：机器精度级别误差
- **高效率**：一次前向传播计算所有偏导数
- **数值稳定**：避免步长选择问题

### 4. SVD增强数值稳定性

使用增广系统SVD求解LM问题：
```
[J]     [δ]   [-e]
[D] × [  ] = [ 0]
```
其中D是阻尼矩阵，避免了直接计算 (J^T*J + λI)^(-1)。

## ⚙️ 高级配置

### 参数固定策略
```yaml
# 固定特定参数不参与优化
robot_config:
  fixed_indices: [0, 4, 8, 12, 16, 20]  # 固定所有关节的alpha参数
```

### 误差权重调节
```yaml
# 调整位置和姿态误差的相对重要性
robot_config:
  error_weights: [1.0, 1.0, 1.0, 0.005, 0.005, 0.005]  # 降低姿态权重
```

### 收敛判据设置
```yaml
optimization:
  lm_optimization:
    convergence:
      parameter_tol: 1e-10      # 参数变化阈值
      max_inner_iterations: 10   # 内循环最大次数
      rho_threshold: 0.0        # rho接受阈值
```

## 🔧 故障排除

### 常见问题与解决方案

#### 1. 数据加载失败
**症状**：`FileNotFoundError` 或数据格式错误
**解决**：
- 检查文件路径是否正确
- 确认数据格式与要求一致
- 查看配置文件中的路径设置

#### 2. 优化不收敛
**症状**：误差不下降或参数发散
**解决**：
```yaml
# 调整配置参数
damping:
  lambda_init_group1: 10.0    # 增大初始阻尼
  lambda_init_group2: 0.01    # 增大初始阻尼

constraints:
  max_theta_change_degrees: 0.5  # 减小步长限制
```

#### 3. 连续失败过多
**症状**：频繁触发连续失败机制
**解决**：
- 检查初始参数是否合理
- 增大初始阻尼因子
- 降低theta变化限制

#### 4. 内存不足
**症状**：大数据集导致内存溢出
**解决**：
- 减少采样点数量
- 分批处理数据
- 使用更小的数据类型



## 📝 更新日志

### v3.0 (最新) 🆕
- ✨ **新增连续失败提前结束机制**：连续失败3次自动结束，提高优化效率
- 🚀 **优化数值稳定性**：改进SVD求解和条件数检查
- 📊 **增强日志输出**：更详细的优化过程信息
- 🔧 **改进配置管理**：更灵活的参数配置选项


