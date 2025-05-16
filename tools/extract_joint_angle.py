import re
import os

def extract_joint_angles(input_file_path, output_file_path):
    """
    从指定的输入文件中提取关节角度，并将其保存到输出文件中。

    输入文件的每一行应包含格式如 'j:{val1,val2,val3,val4,val5,val6, ...}' 的数据。
    脚本将提取前六个值作为关节角度。
    输出文件将包含一个表头 'q1,q2,q3,q4,q5,q6'，后跟提取的角度数据。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # 写入表头
            outfile.write("q1,q2,q3,q4,q5,q6\n")

            # 正则表达式用于匹配 j:{...} 中的内容
            # 它会捕获大括号内的所有内容
            pattern = re.compile(r"j:\s*{\s*([^}]*?)\s*}")

            for line in infile:
                match = pattern.search(line)
                if match:
                    # 获取大括号内的内容
                    content_in_braces = match.group(1)
                    
                    # 按逗号分割，并去除每个数字周围的空白
                    angles_str = [angle.strip() for angle in content_in_braces.split(',')]
                    
                    # 确保我们至少有六个角度值
                    if len(angles_str) >= 6:
                        # 提取前六个角度值
                        extracted_angles = angles_str[:6]
                        # 过滤掉空字符串（如果由于尾随逗号等原因产生）
                        extracted_angles = [angle for angle in extracted_angles if angle]
                        if len(extracted_angles) == 6:
                             outfile.write(",".join(extracted_angles) + "\n")
                        else:
                            print(f"警告: 行未能提取6个角度值: {line.strip()}")
                    else:
                        print(f"警告: 行数据不足六个角度值: {line.strip()}")
                elif line.strip(): # 如果行不为空且没有匹配到模式
                    print(f"警告: 行未找到匹配模式: {line.strip()}")

        print(f"关节角度已成功提取到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file_path}")
    except Exception as e:
        print(f"提取过程中发生错误: {e}")

if __name__ == "__main__":
    # 获取当前脚本所在的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建项目根目录的路径（假设 tools 目录在根目录下）
    project_root_dir = os.path.dirname(current_script_dir)
 
    # 输入文件位于根目录下的 data 文件夹中
    input_csv_path = os.path.join(project_root_dir, "data", "new_joint_angle.csv")
    # 输出文件也位于根目录下的 data 文件夹中
    output_csv_path = os.path.join(project_root_dir, "data", "extracted_joint_angles.csv")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    extract_joint_angles(input_csv_path, output_csv_path)

# 示例用法:
# 假设 new_joint_angle.csv 在 data 文件夹中，像这样:
# LOCAL VAR jointtarget j1 = j:{	 -31.1592906503875,63.7697048882113,-46.1799002773807,109.328903958904,-77.6878907575955,79.7655394100006      ,	0	}		{EJ 0,0,0,0,0,0}
# LOCAL VAR jointtarget j2 = j:{	 19.0874736126832,39.061666027408,-30.6173317774502,118.814941664116,42.785858007288,72.1046609866532          ,	0	}		{EJ 0,0,0,0,0,0}
#
# 运行脚本后，extracted_joint_angles.csv 将会是:
# q1,q2,q3,q4,q5,q6
# -31.1592906503875,63.7697048882113,-46.1799002773807,109.328903958904,-77.6878907575955,79.7655394100006
# 19.0874736126832,39.061666027408,-30.6173317774502,118.814941664116,42.785858007288,72.1046609866532
