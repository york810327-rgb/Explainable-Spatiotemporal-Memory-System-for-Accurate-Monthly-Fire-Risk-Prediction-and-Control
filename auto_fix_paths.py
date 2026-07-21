import os
import glob

# 找到你 src 文件夹下所有的 Python 文件
code_folder = "D:/SCI/code/"
py_files = glob.glob(os.path.join(code_folder, "*.py"))

for file_path in py_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 如果文件里包含绝对路径 D:/SCI/，就自动替换为相对路径
    if "D:/SCI/" in content:
        # 将绝对路径替换为相对路径（假设在 D:/SCI/ 下运行代码）
        # D:/SCI/data/ -> data/  或  D:/SCI/results/ -> results/
        new_content = content.replace("D:/SCI/", "./")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"已自动修复路径: {os.path.basename(file_path)}")

print("所有代码路径已转化为 GitHub 标准的相对路径！")