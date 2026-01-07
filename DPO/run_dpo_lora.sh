# How to use:
# replate ms-swift/swift/trainers/rlhf_trainer/dpo_trainer.py with the one in this repo.

# 24GiB
# It is recommended to use padding_free. For more details, please refer to:
# https://github.com/modelscope/ms-swift/blob/main/examples/train/padding_free/dpo.sh

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
main_process_port=23412
nproc_per_node=8

# BASE_MODEL_PATH="ROOT_PATH/saved_llms/Llama-3-8b-sft-mixture"
BASE_MODEL_PATH="ROOT_PATH/saved_llms/Meta-Llama-3.1-8B-Instruct"

# loss_type in [sigmoid (standard DPO), dpo_lc (length controlled DPO), and dpo_debias (dpo + ours)]

    # --loss_type dpo_debias \

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes ${nproc_per_node} --main_process_port ${main_process_port} \
    ./swift/cli/rlhf.py \
    --rlhf_type dpo \
    --loss_type dpo_lc \
    --model ${BASE_MODEL_PATH} \
    --model_type llama3_1 \
    --train_type lora \
    --dataset ROOT_PATH/saved_data/Human-Like-DPO-Dataset \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 25 \
    --save_steps 25 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --rpo_alpha 0.1 \
    --dataset_num_proc 8

    