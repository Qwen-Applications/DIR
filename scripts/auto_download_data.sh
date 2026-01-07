#!/bin/bash

# --- 脚本配置 ---

# 1. 在这里填入你想要下载的所有模型ID
DATASETS_TO_DOWNLOAD=(
    "inclusionAI/ASearcher-train-data"
    "AI-ModelScope/HelpSteer3"
    "AI-ModelScope/alpaca-gpt4-data-en"
    "opencompass/ifeval"
    "modelscope/trivia_qa"
    "modelscope/race"
    "Qwen/ProcessBench"
    "modelscope/gpqa"
    "evalscope/bbh"
    "opencompass/humaneval"
)

# 2. (可选) 设置一个基础下载目录，所有模型都会被下载到这个目录下
#    '.' 表示当前目录
BASE_DOWNLOAD_DIR="./"

# --- 主逻辑 (无需修改) ---

# 确保基础下载目录存在
mkdir -p "${BASE_DOWNLOAD_DIR}"
export HF_ENDPOINT=https://hf-mirror.com

# 遍历模型列表中的每一个模型ID
for dataset_id in "${DATASETS_TO_DOWNLOAD[@]}"; do
    
    # 从模型ID中提取用作文件夹的名称
    # 例如: "Qwen/Qwen3-8B" -> "Qwen3-8B"
    # 如果模型ID不含'/'，则直接使用模型ID本身
    local_dir_name="${dataset_id##*/}"
    full_local_path="${BASE_DOWNLOAD_DIR}/${local_dir_name}"

    echo "============================================================"
    echo "准备下载模型: ${dataset_id}"
    echo "目标本地目录: ${full_local_path}"
    echo "============================================================"

    # 核心的重试循环
    # 'while ! command' 的意思是：当 command 命令失败时 (返回非0退出码), 就一直执行循环体
    # command 成功时 (返回0)，循环就会终止
        
    while ! modelscope download --dataset "${dataset_id}" --local_dir "${full_local_path}"; do
    # while ! hf download --repo-type dataset ${dataset_id} --local-dir ${full_local_path}; do

        # 生成一个1到2秒的随机等待时间
        # ${RANDOM} 是一个0到32767的随机数, % 2 的结果是0或1, 再+1就得到1或2
        sleep_time=10

        # 打印清晰的错误和重试信息
        echo "" # 换行以增强可读性
        echo "----------------- !!! 下载失败 !!! -----------------"
        echo "模型 '${dataset_id}' 下载出错。"
        echo "将在 ${sleep_time} 秒后自动重试..."
        echo "---------------------------------------------------"
        echo "" # 换行

        # 等待后再次尝试
        sleep ${sleep_time}
    done

    echo ""
    echo "+++++++++++++++ [成功] +++++++++++++++"
    echo "模型 '${dataset_id}' 已成功下载到 '${full_local_path}'"
    echo "++++++++++++++++++++++++++++++++++++++"
    echo ""

done

echo "所有模型均已成功下载！"
