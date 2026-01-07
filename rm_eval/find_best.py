import os
import json

import os
import shutil

def reorganize_folders(source_root='.', target_root='consolidated_results'):
    """
    将分散在各个检查点目录下的 'reports' 等文件夹整合到统一的目录结构中。

    :param source_root: 包含所有独立检查点文件夹的根目录。
                        例如 'outputs/my_experiment/'
    :param target_root: 将要创建的、用于存放整合后结果的新目录。
    """
    # 定义需要整合的子文件夹类型
    folders_to_consolidate = ['reports', 'predictions']

    print(f"开始整理文件结构...")
    print(f"源目录: {os.path.abspath(source_root)}")
    print(f"目标目录: {os.path.abspath(target_root)}\n")

    # 1. 遍历需要整合的每种类型的文件夹 ('reports', 'predictions', etc.)
    for folder_type in folders_to_consolidate:
        # 为每种类型在目标目录中创建一个主文件夹，例如 'consolidated_results/reports'
        main_target_path = os.path.join(target_root, folder_type)
        os.makedirs(main_target_path, exist_ok=True)
        print(f"--- 正在处理 '{folder_type}' ---")

        found_any = False
        # 2. 遍历源目录下的所有项目，寻找检查点文件夹
        for ckpt_name in os.listdir(source_root):
            ckpt_path = os.path.join(source_root, ckpt_name)

            # 确保它是一个目录，并且看起来像一个检查点文件夹
            if os.path.isdir(ckpt_path) and 'checkpoint' in ckpt_name:
                
                # 3. 构建源文件夹的完整路径
                # 根据你的截图，路径是: .../<ckpt_name>/reports/<ckpt_name>
                original_content_path = os.path.join(ckpt_path, folder_type, ckpt_name)

                # 4. 检查源文件夹是否存在
                if os.path.isdir(original_content_path):
                    found_any = True
                    # 构建新的目标路径，例如: 'consolidated_results/reports/<ckpt_name>'
                    new_destination_path = os.path.join(main_target_path, ckpt_name)

                    # 5. 复制整个文件夹内容
                    try:
                        # 如果目标文件夹已存在，先删除，防止 shutil.copytree 报错
                        if os.path.exists(new_destination_path):
                            shutil.rmtree(new_destination_path)
                        
                        shutil.copytree(original_content_path, new_destination_path)
                        print(f"  [√] 已将 {ckpt_name} 的 {folder_type} 复制到目标目录。")
                    except Exception as e:
                        print(f"  [X] 复制 {ckpt_name} 的 {folder_type} 时出错: {e}")
                # else:
                #     # 如果需要，可以取消下面这行注释来查看哪些检查点没有报告
                #     print(f"  [i] 在 {ckpt_name} 中未找到 '{folder_type}' 文件夹。")

        if not found_any:
            print(f"  未找到任何 '{folder_type}' 类型的文件进行整理。")
        print("-" * (len(folder_type) + 14) + "\n")

    print("🎉 文件结构整理完成！")
    print(f"所有结果已整理到 '{os.path.abspath(target_root)}' 目录中。")
    print("现在你可以对 'consolidated_results/reports' 运行上一个分析脚本了。")


def find_best_checkpoint(reports_path='reports'):
    """
    分析指定路径下的评测报告，计算每个检查点的平均分，并找出最优的检查点。

    :param reports_path: 包含所有检查点结果的根目录路径。
    """
    checkpoint_averages = {}

    # 检查根目录是否存在
    if not os.path.isdir(reports_path):
        print(f"错误：目录 '{reports_path}' 不存在。")
        return

    # 1. 遍历根目录下的所有检查点子目录
    for ckpt_name in os.listdir(reports_path):
        ckpt_path = os.path.join(reports_path, ckpt_name)

        if os.path.isdir(ckpt_path):
            scores = []
            
            # 2. 遍历检查点目录中的所有文件
            for report_file in os.listdir(ckpt_path):
                # 确保只处理 .json 文件
                if report_file.endswith('.json'):
                    report_path = os.path.join(ckpt_path, report_file)
                    
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 3. 提取 "score" 的值
                            if 'score' in data:
                                scores.append(data['score'])
                            else:
                                print(f"警告：在文件 {report_path} 中未找到 'score' 键。")
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"警告：读取或解析文件 {report_path} 时出错: {e}")
            
            # 4. 如果找到了分数，则计算平均分
            if scores:
                average_score = sum(scores) / len(scores)
                checkpoint_averages[ckpt_name] = average_score

    # 5. 如果有结果，则进行排序和打印
    if not checkpoint_averages:
        print("未找到任何有效的检查点评测结果。")
        return

    # 打印所有检查点的平均分
    print("--- 所有检查点的平均分 ---")
    # 按分数从高到低排序
    sorted_checkpoints = sorted(checkpoint_averages.items(), key=lambda item: item[1], reverse=True)
    
    for ckpt, avg_score in sorted_checkpoints:
        print(f"{ckpt}: {avg_score:.4f}") # 格式化输出，保留4位小数

    # 6. 找出并高亮显示最优结果
    best_checkpoint_name, best_score = sorted_checkpoints[0]
    
    print("\n" + "="*30)
    print("✨ 表现最佳的检查点 ✨")
    print(f"    - 名称: {best_checkpoint_name}")
    print(f"    - 平均分: {best_score:.4f}")
    print("="*30)


# --- 使用方法 ---
if __name__ == "__main__":
    # 设置包含所有 'v21-...' 文件夹的根目录
    # '.' 代表当前目录
    # source_directory = 'ROOT/evalscope-main/outputs/v21-ours1.0-OpenRLHF-Llama3-8B-SFT' 
    # target_directory = 'ROOT/evalscope-main/outputs/v21-ours1.0-OpenRLHF-Llama3-8B-SFT/consolidated_results'
    # reorganize_folders(source_root=source_directory, target_root=target_directory)

    # root_directory = 'ROOT/evalscope-main/outputs/v20-0.1-meta-Llama31-8B-it/reports'
    # find_best_checkpoint(root_directory)
    
    # root_directory = 'ROOT/evalscope-main/outputs/v17-Ours1.0-meta-Llama31-8B-it/consolidated_results/reports'
    # find_best_checkpoint(root_directory)

    # root_directory = 'ROOT/evalscope-main/outputs/v20-0.1-meta-Llama31-8B-it/reports'
    # find_best_checkpoint(root_directory)


    # root_directory = 'ROOT/evalscope-main/outputs/v18-SK0.0-meta-Llama31-8B-it/reports'
    # find_best_checkpoint(root_directory)

    # root_directory = 'ROOT/evalscope-main/True/reports'
    # find_best_checkpoint(root_directory)

    
    # root_directory = 'ROOT/evalscope-main/outputs/20250824_142001-openRLHF-llama-sft/reports/'
    # find_best_checkpoint(root_directory)
    # root_directory = 'ROOT/evalscope-main/outputs/v22-0.0-OpenRLHF-Llama3-8B-SFT/reports'
    # find_best_checkpoint(root_directory)
    # root_directory = 'ROOT/evalscope-main/outputs/v21-ours1.0-OpenRLHF-Llama3-8B-SFT/consolidated_results/reports'
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v24-ours0.1-OpenRLHF-Llama3-8B-SFT/reports"
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v25-0.001-meta-Llama31-8B-it/reports"
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v34-EMNLP0.0-openRLHF-Llama3-8B-SFT/reports"
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v35-EMNLP0.0-meta-Llama31-8B-it/reports"
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v37-0.0-meta-Llama31-8B-it/reports"
    # find_best_checkpoint(root_directory)

    # root_directory = "ROOT/evalscope-main/outputs/v38-MH0.0-openRLHF-Llama3-8B-SFT/reports"
    # find_best_checkpoint(root_directory)

    root_directory = "ROOT/evalscope-main/outputs/v40-InfoRM1.0-openRLHF-Llama3-8B-SFT/reports"
    find_best_checkpoint(root_directory)

    root_directory = "ROOT/evalscope-main/outputs/v41-1.0-meta-Llama3.1-8B-it/reports"
    find_best_checkpoint(root_directory)