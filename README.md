# 机器人参数标定项目 (基于PyTorch的LM优化)

## 1. 项目简介

本项目旨在使用激光跟踪仪的测量数据，对六轴工业机器人的运动学参数进行标定。主要标定内容包括：

*   **DH (Denavit-Hartenberg) 参数**: 修正的MDH参数 (alpha, a, d, theta_offset)。
*   **工具TCP (Tool Center Point) 参数**: 工具中心点相对于机器人末端法兰的位姿 (位置和姿态四元数)。
*   **基座参数 (T_laser_base)**: 机器人基座在激光跟踪仪坐标系下的位姿 (位置和姿态四元数)。

项目采用Levenberg-Marquardt (LM) 优化算法，并利用PyTorch框架进行自动微分和张量运算，以提高计算效率和实现灵活性。同时，项目中包含了交替优化的策略，分别优化不同组别的参数。

## 2. 功能特点

*   **基于MDH模型的正向运动学**: `jacobian_torch.py` 中实现了精确的MDH正向运动学计算。
*   **自动微分计算雅可比矩阵**: 利用PyTorch的自动微分功能，高效计算误差向量相对于待标定参数的雅可比矩阵。
*   **LM优化算法**: `lm_optimize_pytorch.py` 中实现了LM算法，用于最小化预测位姿与测量位姿之间的误差。
*   **交替优化策略**: 支持将参数分组，进行交替优化，以应对不同参数间的耦合影响。
*   **参数管理**: 支持从CSV文件加载和保存优化后的DH参数、TCP参数和基座参数。
*   **误差评估**: `test_error.py` 脚本用于计算和对比不同参数配置下的RMSE (均方根误差)，评估标定效果。
*   **模块化设计**: 代码结构清晰，分为数据处理、运动学计算、优化算法、结果评估等模块。

## 3. 目录结构

```
jacobian/
├── data/                         # 存放原始测量数据
│   ├── extracted_joint_angles.csv  # 机器人关节角度数据
│   └── extracted_laser_positions.csv # 激光跟踪仪测量的工具位姿数据
├── jacobian/                     # 三种计算雅可比方法
├── results/                      # 存放优化后的参数和中间结果
│   ├── optimized_dh_parameters.csv
│   ├── optimized_tcp_parameters.csv
│   └── optimized_t_laser_base_parameters.csv
├── test/                         # 测试脚本
│   └── test_error.py             # RMSE误差计算和对比脚本
├── tools/                        # 提取数据工具
├── jacobian_torch.py             # 核心：MDH运动学, 四元数与旋转矩阵转换, 雅可比计算接口
├── lm_optimize_pytorch.py        # 核心：LM优化算法实现, 参数保存, 优化流程控制
├── README.md                     # 本文件
└── requirements.txt              # Python依赖项
```

## 4. 主要脚本说明

*   **`lm_optimize_pytorch.py`**:
    *   主优化脚本。
    *   加载初始参数和测量数据。
    *   执行LM优化算法（可选择直接优化或交替优化）。
    *   保存优化后的参数到 `results/` 目录。
    *   评估优化前后的误差。
*   **`jacobian_torch.py`**:
    *   定义机器人MDH参数、TCP参数、基座参数的初始值。
    *   实现MDH变换矩阵的构建。
    *   实现四元数与旋转矩阵之间的转换。
    *   实现机器人正向运动学计算 (`forward_kinematics_T`)。
    *   提供计算误差向量对于参数的雅可比矩阵的函数 (`compute_error_vector_jacobian`)，这是LM算法的核心。
    *   从激光跟踪仪数据文件加载并处理测量数据 (`get_laser_tool_matrix`)。
*   **`test/test_error.py`**:
    *   加载测量数据。
    *   加载用户指定的参数（硬编码在脚本中，作为对比基准）。
    *   从 `results/` 目录加载优化后的参数。
    *   计算并打印不同参数组下的RMSE，用于评估标定效果。

## 5. 环境配置与运行步骤

### 5.1. 环境依赖

确保已安装Python环境 (推荐 Python 3.8+)。然后通过pip安装必要的库：

```bash
pip install -r requirements.txt
```

### 5.2. 数据准备

1.  **关节角度数据**: 将机器人采集的N组关节角度数据保存为 `data/extracted_joint_angles.csv`。
    文件格式应为CSV，每行代表一组数据，包含6个关节的角度值（单位：度），以逗号分隔，第一行为表头（可选，脚本中通过 `skiprows=1` 跳过）。
    示例:
    ```csv
    J1,J2,J3,J4,J5,J6
    0.0,0.0,0.0,0.0,0.0,0.0
    10.5,-20.1,30.2,5.0,-15.7,25.8
    ...
    ```
2.  **激光跟踪仪测量数据**: 将激光跟踪仪测量的N组工具位姿数据保存为 `data/extracted_laser_positions.csv`。
    文件格式应为CSV，每行代表一组数据，包含工具在激光跟踪仪坐标系下的位置 (x, y, z，单位：mm) 和姿态 (rx, ry, rz，欧拉角，单位：度，通常为xyz内旋)，以逗号分隔，第一行为表头（可选，脚本中通过 `skiprows=1` 跳过）。
    示例:
    ```csv
    X,Y,Z,Rx,Ry,Rz
    2500.0,3000.0,50.0,0.1,0.2,-0.3
    ...
    ```
    **注意**: 关节角度数据和激光测量数据必须一一对应，即第 `i` 行的关节角度对应第 `i` 行的激光测量位姿。

### 5.3. 参数初始化与配置

*   **初始DH参数**: 在 `jacobian_torch.py` 中修改 `INIT_DH_PARAMS` 列表，提供机器人名义上的或估计的MDH参数。
*   **初始TCP参数**: 在 `jacobian_torch.py` 中修改 `INIT_TOOL_OFFSET_POSITION` 和 `INIT_TOOL_OFFSET_QUATERNION`。
*   **初始基座参数**: 在 `jacobian_torch.py` 中修改 `INIT_T_LASER_BASE_PARAMS`。
*   **误差权重**: 在 `jacobian_torch.py` 中修改 `ERROR_WEIGHTS` (前三个为位置误差权重，后三个为姿态误差权重)。
*   **固定参数**: 在 `lm_optimize_pytorch.py` 中修改 `ALL_FIXED_INDICES` 列表，指定在优化过程中保持不变的参数索引 (0-23为DH参数，24-30为TCP参数，31-37为基座参数)。
*   **优化超参数**: 在 `lm_optimize_pytorch.py` 的 `if __name__ == '__main__':` 部分，可以调整LM算法的超参数，如最大迭代次数 `max_iterations` (子优化中) 或 `max_alt_iterations` (交替优化中)，初始阻尼因子 `lambda_init` 等。

### 5.4. 运行标定

打开终端，导航到项目根目录，执行主优化脚本：

```bash
python lm_optimize_pytorch.py
```

优化过程中的信息（如每次迭代的RMSE、参数更新量等）会打印到控制台。优化完成后，标定结果将保存在 `results/` 目录下的CSV文件中。

### 5.5. 评估结果

运行测试脚本来评估标定效果并与理论值或优化前的值进行比较：

```bash
python test/test_error.py
```

该脚本会输出不同参数来源（如：指定值、优化后的CSV值）计算得到的RMSE。

## 6. 参数定义

*   **MDH参数 (Modified Denavit-Hartenberg)**:
    *   `alpha`: 绕 `x_i` 轴从 `z_{i-1}` 到 `z_i` 的旋转角度。
    *   `a`:     沿 `x_i` 轴从 `z_{i-1}` 到 `z_i` 的平移距离。
    *   `d`:     沿 `z_{i-1}` 轴从 `x_{i-1}` 到 `x_i` 的平移距离。
    *   `theta_offset`: 关节变量 `q_i` 的偏移量，实际关节转角为 `q_i + theta_offset`。
*   **TCP参数**: 工具中心点相对于机器人末端法兰坐标系的位姿。
    *   `tx, ty, tz`: 位置偏移 (mm)。
    *   `qx, qy, qz, qw`:姿态四元数 (w为实部)。
*   **基座参数 (T_laser_base)**: 机器人基座坐标系在激光跟踪仪坐标系下的位姿。
    *   `tx, ty, tz`: 位置偏移 (mm)。
    *   `qx, qy, qz, qw`:姿态四元数 (w为实部)。

## 7. 注意事项

*   **数据质量**: 标定结果高度依赖于测量数据的精度和一致性。请确保数据采集过程准确无误。
*   **初始参数**: LM算法对初始参数的敏感性较高。提供一个相对接近真实值的初始参数有助于算法收敛到全局最优解。
*   **奇异姿态**: 避免在机器人奇异姿态附近采集数据，这可能导致雅可比矩阵病态，影响优化稳定性。
*   **单位一致性**: 确保所有长度单位 (mm) 和角度单位 (度或弧度，代码中主要是度和弧度间的转换) 在计算过程中保持一致。脚本内部主要使用mm和度，但在PyTorch中进行三角函数运算时会转换为弧度。
*   **激光跟踪仪数据格式**: `get_laser_tool_matrix` 函数中，欧拉角转换为旋转矩阵时使用的是 `'xyz'` 内旋顺序，请根据实际激光跟踪仪输出的欧料角顺序进行调整。 