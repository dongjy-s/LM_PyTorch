import os

def extract_laser_data(input_file_path, output_file_path):
    """
    从指定的激光位置数据文件中提取测量数据 (X, Y, Z, Rx, Ry, Rz)，
    并将其保存到输出文件中。

    输入文件的第一行是表头，将被跳过。
    后续每一行的数据应为空格分隔值。
    脚本将提取每行中的第8到第13个值。
    输出文件将包含一个表头 'x,y,z,rx,ry,rz'，后跟提取的数据，
    数据之间用 ', ' (逗号加空格) 分隔。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # 写入表头到输出文件
            outfile.write("x,y,z,rx,ry,rz\n")

            # 跳过输入文件的第一行 (表头)
            next(infile)

            for line_number, line in enumerate(infile, start=2): # 从第二行开始计数
                stripped_line = line.strip()
                if not stripped_line:  # 跳过空行
                    continue

                parts = stripped_line.split()

                # 检查是否有足够的列来提取数据
                # 我们需要索引 7 到 12 (即第8到第13列)
                if len(parts) >= 13:
                    # 提取测量数据列 (索引 7 到 12)
                    measured_data = parts[7:13]
                    # 用 ", " 连接数据并写入文件
                    outfile.write(",".join(measured_data) + "\n")
                else:
                    print(f"警告: 第 {line_number} 行数据列数不足，无法提取: {stripped_line}")
            
        print(f"激光位置数据已成功提取到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file_path}")
    except Exception as e:
        print(f"提取过程中发生错误: {e}")

if __name__ == "__main__":
    # 获取当前脚本所在的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建项目根目录的路径（假设 tools 目录在根目录下）
    project_root_dir = os.path.dirname(current_script_dir)
    
    # 定义输入和输出文件的相对路径
    # 输入文件位于根目录下的 data 文件夹中
    input_csv_path = os.path.join(project_root_dir, "data", "laser_pos.csv")
    # 输出文件也位于根目录下的 data 文件夹中
    output_csv_path = os.path.join(project_root_dir, "data", "extracted_laser_positions.csv")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    extract_laser_data(input_csv_path, output_csv_path)

# 示例用法:
# 假设 new_laser_pos.csv 在 data 文件夹中，内容类似:
# 名称 名义 X 名义 Y 名义 Z 名义 Rx 名义 Ry 名义 Rz 测量 X 测量 Y 测量 Z 测量 Rx 测量 Ry 测量 Rz 余量 位置 余量 方向
# Motion1 2441.5012 1667.0849 390.6434 45.4983 -70.8809 -166.36 2441.5048 1667.1769 390.7169 45.4378 -70.8594 -166.2885 0.1178 0.0326
# Motion2 3150.5488 1892.0208 966.3411 -9.8739 -72.9151 -170.8179 3150.6443 1891.9505 966.3835 -9.8331 -72.9314 -170.8447 0.1259 0.0236
#
# 运行脚本后，extracted_laser_positions.csv 将会是:
# x,y,z,rx,ry,rz
# 2441.5048, 1667.1769, 390.7169, 45.4378, -70.8594, -166.2885
# 3150.6443, 1891.9505, 966.3835, -9.8331, -72.9314, -170.8447
