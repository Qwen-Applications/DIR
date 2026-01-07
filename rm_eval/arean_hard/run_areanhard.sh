#!/bin/bash

# --- 配置区 ---
set -e # 任何命令失败都会导致脚本立即退出

export VLLM_USE_MODELSCOPE=true
export VLLM_LOGGING_LEVEL=ERROR

# 基础模型和vLLM服务器配置
# BASE_MODEL_PATH="ROOT/saved_llms/Llama-3-8b-sft-mixture"
BASE_MODEL_PATH="ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct"
PORT=8801
TENSOR_PARALLEL_SIZE=8

# 要评测的LoRA检查点列表
ckpt_list=(
    # "ROOT/ms-swift-main/output/v21-20250823-153121-Ours-1.0-OpenRLHF-Llama3-8B-SFT/checkpoint-300"
    "ROOT/ms-swift-main/output/v20-20250823-132302-Ours-0.1-meta-Llama31-8B-it/checkpoint-600"
    # "ROOT/ms-swift-main/output/v38-20250905-091231-MH-0.0-openRLHF-Llama3-8B-SFT/checkpoint-200"
    "ROOT/ms-swift-main/output/v37-20250904-092330-MH-0.0-meta-Llama31-8B-it/checkpoint-200"
    "ROOT/ms-swift-main/output/v35-20250903-091355-EMNLP-0.0-meta-Llama31-8B-it/checkpoint-300"
    # "ROOT/ms-swift-main/output/v34-20250902-082943-EMNLP-0.0-openRLHF-Llama3-8B-SFT/checkpoint-300"
    # "ROOT/ms-swift-main/output/v22-20250824-193856-SK-0.0-OpenRLHF-Llama3-8B-SFT/checkpoint-200"
    "ROOT/ms-swift-main/output/v18-20250820-125600-SK-0.0-meta-Llama31-8B-it/checkpoint-400"
    # "ROOT/ms-swift-main/output/v40-20250911-100224-infoRM-1.0-openRLHF-Llama3-8B-SFT/checkpoint-400"
    "ROOT/ms-swift-main/output/v41-20250912-023732-inroRM-1.0-meta-Llama3.1-8B-it/checkpoint-600"
)   

# 评测数据和输出配置
INPUT_DATA_FILE="ROOT/saved_data/arean-hard-v1.json" # 你的输入数据文件
OUTPUT_DIR="./arean_hard/areanhead_v1_evaluation_outputs"

# --- 准备阶段 ---
echo "--- Preparing LoRA modules and evaluation environment ---"
lora_names_array=()
lora_modules_str=""

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}

for lora_path in "${ckpt_list[@]}"; do
    # 生成一个更健壮、更简洁的 lora_name (e.g., v21-Ours...-checkpoint-300)
    lora_name="$(basename "$(dirname "$lora_path")")-$(basename "$lora_path")"

    # 添加到数组和字符串中，用于后续的 vLLM 和评测循环
    lora_names_array+=("${lora_name}")
    lora_modules_str+="${lora_name}=${lora_path} "
    echo "  - Prepared adapter: ${lora_name}"
done

# --- 服务启动与清理 ---
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "--- Cleaning up: Stopping vLLM server (PID: $VLLM_PID) ---"
        # 使用 pgrep 和 kill 确保子进程也被终止
        pkill -P $VLLM_PID
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null
    fi
}
trap cleanup EXIT # 脚本退出时（正常或异常）自动执行 cleanup 函数

# --- 启动 VLLM 服务器 (仅一次) ---
echo "=========================================================="
echo ">>> Starting vLLM server with ALL LoRA adapters"
echo "=========================================================="

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL_PATH}" \
    --enable-lora \
    --lora-modules ${lora_modules_str} \
    --trust-remote-code \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len 4096 \
    --port ${PORT} & # & 将服务器置于后台运行

VLLM_PID=$!
echo "--- vLLM server starting with PID: ${VLLM_PID} ---"

# 等待vLLM服务器准备就绪
echo -n "--- Waiting for vLLM server to be ready on port ${PORT} "
while ! curl -s -f http://127.0.0.1:${PORT}/health > /dev/null; do
    echo -n "."
    sleep 5
done
echo " Ready! ---"

# --- 循环评测 ---
echo "=========================================================="
echo ">>> Starting evaluation loop for all loaded adapters"
echo "=========================================================="
for lora_name in "${lora_names_array[@]}"; do
    echo "----------------------------------------------------------"
    echo "--- Evaluating: ${lora_name} ---"
    
    output_file="${OUTPUT_DIR}/${lora_name}.json"

    # *** 修改这里的 python 调用命令 ***
    # 假设你的新脚本名为 eval_areanhard.py
    python eval_areanhard.py \
        --model-name "${lora_name}" \
        --input-file "${INPUT_DATA_FILE}" \
        --output-file "${output_file}" \
        --api-url "http://127.0.0.1:${PORT}/v1" \
        --api-key "EMPTY" \
        --temperature 0.7 \
        --tokenizer-path "${BASE_MODEL_PATH}" \
        --model-max-len 4096

    echo "--- Evaluation for ${lora_name} finished. Results saved to ${output_file} ---"
done

echo "=========================================================="
echo ">>> All LoRA checkpoints have been evaluated successfully! <<<"
echo "=========================================================="

# 脚本正常结束时，trap 会自动触发 cleanup 函数来关闭服务器