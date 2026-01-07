devices=0,1,2,3,4,5,6,7
n_gpu=8
# devices=0
# n_gpu=1
# dataset_name='hendrydong/preference_700K'
train_dataset_name='ROOT/saved_data/Skywork-Reward-Preference-80K-v0.2'
eval_dataset_name="ROOT/saved_data/reward-bench"
base_model='ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct'
wandb_name="BT_train_Zero1"
main_process_port=35272

learning_rate=2e-6
max_length=4096
num_train_epochs=1
gradient_accumulation_steps=16
per_device_train_batch_size=1
per_device_eval_batch_size=1
bf16=True
log_dir="./BT_Baseline/"
save_steps=150
eval_steps=150
max_length=4096
save_strategy=steps
eval_on_start=False

# sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

cd ./reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} run_reward_models_train.py \
    --base_model ${base_model} --wandb_name ${wandb_name} --log_dir ${log_dir} --report_to tensorboard \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --train_dataset ${train_dataset_name} \
    --eval_dataset ${eval_dataset_name} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --bf16 True \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${eval_steps} \
    --eval_on_start ${eval_on_start} \
    --deepspeed ../deepspeed_configs/deepspeed_1.json