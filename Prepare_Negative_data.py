import os
import shutil

def read_probability(file_path):
    """读取 Chagas probability"""
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("# Chagas probability:"):
                return float(line.split(": ")[1].strip())
    return 1.0  # 默认值（如果文件格式异常，则认为概率最高）

def get_top_files(folder, top_n=1500):
    """获取指定文件夹中 Chagas probability 最小的 top_n 个文件"""
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    file_probs = [(f, read_probability(os.path.join(folder, f))) for f in txt_files]
    file_probs.sort(key=lambda x: x[1])  # 按概率排序
    return file_probs[:top_n]

def move_selected_files(selected_files, source_folder, dest_folder):
    """移动文件到目标文件夹"""
    os.makedirs(dest_folder, exist_ok=True)
    for file, _ in selected_files:
        base_name = os.path.splitext(file)[0]
        for ext in [".dat", ".hea"]:
            src = os.path.join(source_folder, base_name + ext)
            dest = os.path.join(dest_folder, base_name + ext)
            if os.path.exists(src):
                shutil.move(src, dest)

def main():
    # 定义路径
    output_folders = [
        "/users/wmqn2362/PhysioNet2025/SEResNet_with_imbalance/Output",
        "/users/wmqn2362/PhysioNet2025/SEResNet_Baseline/Output",
        "/users/wmqn2362/PhysioNet2025/SEResNet_With_Dividemix/Output"
    ]
    source_folder = "/mnt/scratch/wmqn2362/PhysioNet25/CODE15"
    dest_folder = "/mnt/scratch/wmqn2362/PhysioNet25/Negative"
    
    # 获取每个文件夹中概率最小的 1500 个文件
    all_selected_files = []
    for folder in output_folders:
        all_selected_files.extend(get_top_files(folder, 5000))
    
    # 找到出现 3 次的文件
    file_counts = {}
    for file, prob in all_selected_files:
        if file in file_counts:
            file_counts[file].append(prob)
        else:
            file_counts[file] = [prob]
    
    repeated_files = [(f, min(probs)) for f, probs in file_counts.items() if len(probs) == 3]
    
    # 只选取最小的 1000 个
    selected_1000_files = sorted(repeated_files, key=lambda x: x[1])[:1000]
    
    # 获取最大 Chagas probability
    max_probability = max(selected_1000_files, key=lambda x: x[1])[1]
    
    # 移动文件
    move_selected_files(selected_1000_files, source_folder, dest_folder)
    print(f"已移动 {len(selected_1000_files)} 个文件到 {dest_folder}")
    print(f"移动的 1000 个文件中最大的 Chagas probability: {max_probability}")

if __name__ == "__main__":
    main()
