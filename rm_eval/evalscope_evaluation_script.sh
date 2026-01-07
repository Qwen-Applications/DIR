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
    "ROOT/ms-swift-main/output/v41-20250912-023732-infoRM-1.0-meta-Llama3.1-8B-it/checkpoint-200"
)

# --- 准备阶段 ---
echo "--- Preparing LoRA modules ---"
lora_names_array=()
lora_modules_str=""
for lora_path in "${ckpt_list[@]}"; do
    # 生成 lora_name (e.g., v21-1.0-OpenRLHF-Llama3-8B-SFT-checkpoint-200)
    checkpoint_name="${lora_path##*/}"
    temp_path="${lora_path%/*}"
    parent_dir_name="${temp_path##*/}"
    # 调整IFS，因为父目录名现在有更多部分
    IFS='-' read -r v_part date_part time_part ours_part ver_part hf_part model_part sft_part <<< "${parent_dir_name}"
    # 重新组合成更有意义的名字
    exp_name="${v_part}-${ver_part}-${hf_part}-${model_part}-${sft_part}"
    lora_name="${exp_name}-${checkpoint_name}"

    # 添加到数组和字符串
    lora_names_array+=("${lora_name}")
    lora_modules_str+="${lora_name}=${lora_path} "
    echo "  - Prepared adapter: ${lora_name}"
done

echo ${lora_names_array}

# --- 服务启动与清理 ---
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "--- Cleaning up: Stopping vLLM server (PID: $VLLM_PID) ---"
        if ps -p $VLLM_PID > /dev/null; then
            kill $VLLM_PID
            wait $VLLM_PID 2>/dev/null
        fi
    fi
}
trap cleanup EXIT

# --- 启动 VLLM 服务器 (仅一次) ---
echo "=========================================================="
echo ">>> Starting vLLM server with ALL LoRA adapters"
echo "=========================================================="

python -m vllm.entrypoints.openai.api_server \
    --model ${BASE_MODEL_PATH} \
    --served-model-name Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-modules ${lora_modules_str} \
    --trust_remote_code \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len 4096 \
    --max-seq-len-to-capture 48000 \
    --port ${PORT} &

VLLM_PID=$!
echo "--- vLLM server started with PID: ${VLLM_PID} ---"

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
    echo "--- Evaluating: ${lora_name} ---"
    
    # 从 lora_name 提取实验名用于缓存目录
    exp_name=$(echo "${lora_name}" | sed "s/-checkpoint-[0-9]*$//")

    evalscope eval \
        --model ${lora_name} \
        --api-url http://127.0.0.1:${PORT}/v1 \
        --api-key EMPTY \
        --eval-batch-size 16 \
        --eval-type service \
        --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
        --generation-config '{"do_sample":true,"temperature":0.7,"max_new_tokens":2048}' \
        --dataset-args '{"gsm8k": {"dataset_id": "ROOT/saved_data/ppo_benchs/benchmark_data/data/gsm8k"},
                "hellaswag": {"dataset_id": "ROOT/saved_data/ppo_benchs/benchmark_data/data/hellaswag" },
                "mmlu": {"dataset_id": "ROOT/saved_data/ppo_benchs/benchmark_data/data/mmlu"},
                "ifeval": {"dataset_id": "ROOT/saved_data/ppo_benchs/benchmark_data/ifeval"},
                "process_bench": {"dataset_id": "ROOT/saved_data/ppo_benchs/benchmark_data/ProcessBench"},
                "race": {"dataset_id": "ROOT/saved_data/data/race"},      
                "bbh": {"dataset_id": "ROOT/saved_data/data/bbh"},      
                "humaneval": {"dataset_id": "ROOT/saved_data/humaneval/openai_humaneval/humaneval.json"},  
                "trivia_qa": {"dataset_id": "ROOT/saved_data/data/trivia_qa"}}' \
        --ignore-errors \
        --datasets gsm8k hellaswag mmlu ifeval process_bench race bbh humaneval trivia_qa \
        --use-cache outputs/${exp_name}

    echo "--- Evaluation for ${lora_name} finished. ---"
done

echo "=========================================================="
echo ">>> All LoRA checkpoints have been evaluated successfully! <<<"
echo "=========================================================="