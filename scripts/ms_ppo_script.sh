
# 8 * 65 GiB
# Currently, it only supports the case where the model and reward_model use the same template/tokenizer.
# Currently, multimodal model PPO is not supported.

# pip install "deepspeed==0.14.*"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
main_process_port=23412
nproc_per_node=8

# ref_model="ROOT/saved_llms/Llama-3-8b-sft-mixture"
# model_type="llama3"
ref_model="ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct"
model_type="llama3_1"

our_RM=""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes ${nproc_per_node} --main_process_port ${main_process_port} \
    ./swift/cli/rlhf.py \
    --rlhf_type ppo \
    --model ${ref_model} \
    --model_type ${model_type} \
    --reward_model ${our_RM} \
    --train_type lora \
    --dataset 'ROOT/saved_data/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 12 \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --deepspeed zero3 \
    --response_length 2048 \
    --temperature 0.7 \
    --dataset_num_proc 8 \
    --save_only_model true
