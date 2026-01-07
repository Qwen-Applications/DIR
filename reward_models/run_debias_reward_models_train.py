from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import evaluate
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from load_datasets import load_train_eval_dataset, build_eval_dataset_RewardBenchV2, build_eval_dataset_RewardBenchV1, build_eval_dataset_RMbench
from utils import print_trainable_parameters, compute_metrics
from reward_trainer import RewardDataCollatorWithPadding
import torch.distributed as dist


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1) 
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "The number of training epochs for the reward model."})
    optim: Optional[str] = field(default="adamw_torch",  metadata={"help": "The optimizer to use."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024) 
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."},)
    # data
    train_dataset: Optional[str] = field(default='llm-blender/Unified-Feedback')
    eval_dataset: Optional[str] = field(default=None)
    dataset_mode: Optional[str] = field(default='', metadata={"help": "use from '', '40k', and '400k' for the paper's experiments"},)
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=100)
    eval_on_start: Optional[bool] = field(default=False)
    # model and loss
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    loss_type: Optional[str] = field(default='bt', metadata={'help': "use 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    weight_ratio: Optional[float] = field(default=0.1, metadata={'help': 'the ratio for label smooth or posreg'})
    # log
    report_to: Optional[str] = field(default='none', metadata={'help': "use 'none', 'wandb'. "})
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test",)
    save_strategy: Optional[str] = field(default="epoch")
    save_steps: Optional[int] = field(default=1000)
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})

    # debias
    use_debias: bool = field(default=True, metadata={"help": "Enable debiasing with a CLUB network."})
    debias_task: str = field(default="length", metadata={"help": "Task of debiasing ('length' or 'sycophancy' or 'people')."})
    debias_type: str = field(default="difference", metadata={"help": "Type of representation for debiasing ('difference' or 'concat')."})
    debias_lr: float = field(default=1e-4, metadata={"help": "Learning rate for the debias network optimizer."})
    debias_hidden_dim: int = field(default=1024, metadata={"help": "Hidden dimension of the debias network."})
    debias_factor: float = field(default=0.1, metadata={"help": "Coefficient for the MI loss term (gamma)."})
    debias_optim_iter: int = field(default=5, metadata={"help": "Number of optimization steps for the debias network per main step."})
    debias_projection: bool = field(default=False, metadata={"help": "Use linear projection to get bias-specific representation."})
    debias_sycophancy_contamination_ratio: float = field(default=0.8, metadata={"help": "Contamination ratio in sycophancy bias."})
    debias_sycophancy_sycophancy_prob: float = field(default=0.4, metadata={"help": "Sycophancy probability in sycophancy bias."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model_name_split = script_args.base_model.split("/")[-1]

output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_Debias-Task{script_args.debias_task}_{script_args.debias_type}-{script_args.debias_factor}_len{script_args.max_length}_fulltrain_{script_args.learning_rate}_data{script_args.train_dataset.split('/')[-1]}"

training_args = TrainingArguments(
    output_dir=os.path.join(output_name,),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    eval_strategy=script_args.eval_strategy,
    eval_steps=script_args.eval_steps,
    eval_on_start=script_args.eval_on_start,
    save_strategy=script_args.save_strategy,
    save_steps=script_args.save_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing, 
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.03,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to=script_args.report_to,
    remove_unused_columns=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    deepspeed=script_args.deepspeed,
)

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast = False)
tokenizer.max_length = script_args.max_length
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# Load datasets
if script_args.eval_dataset is not None:
    train_dataset = load_train_eval_dataset(script_args.train_dataset, tokenizer, mode=script_args.dataset_mode, test_size=0, size=100 if script_args.debug else None)
    # eval_dataset, total_completions, num_correct = build_eval_dataset_RewardBenchV2(script_args.eval_dataset, tokenizer)
    # rbv1_eval_dataset = build_eval_dataset_RewardBenchV1("ROOT/saved_data/reward-bench", tokenizer)
    rm_bench_eval_dataset = build_eval_dataset_RMbench("ROOT/benchs/RM-Bench-main/data/total_dataset.json", tokenizer)
    eval_dataset = rm_bench_eval_dataset
elif script_args.eval_dataset is None:
    if script_args.debias_task == "sycophancy":
        train_dataset, eval_dataset = load_train_eval_dataset(script_args.train_dataset, tokenizer, mode=script_args.dataset_mode, size=100 if script_args.debug else None, contamination_ratio=script_args.debias_sycophancy_contamination_ratio, sycophancy_prob=script_args.debias_sycophancy_sycophancy_prob)
    else:
        train_dataset, eval_dataset = load_train_eval_dataset(script_args.train_dataset, tokenizer, mode=script_args.dataset_mode, size=100 if script_args.debug else None)

if len(script_args.attn_implementation):
    model_params = {
        "attn_implementation": script_args.attn_implementation,
    }
else:
    model_params = {}

if script_args.debias_task == 'length':
    from debias_trainer_length import DeBiasedRewardTrainer
elif script_args.debias_task == 'sycophancy':
    from debias_trainer_sycophancy import DeBiasedRewardTrainer

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model, num_labels=1,
    torch_dtype=torch.bfloat16,
    **model_params
)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print_trainable_parameters(model)

# Define the trainer parameters
trainer_params = {
    "model": model,
    "args": training_args,
    "tokenizer": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "compute_metrics": compute_metrics,
    "data_collator": RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    "use_debias": script_args.use_debias,
    "debias_type": script_args.debias_type,
    "debias_lr": script_args.debias_lr,
    "debias_hidden_dim": script_args.debias_hidden_dim,
    "debias_factor": script_args.debias_factor,
    "debias_optim_iter": script_args.debias_optim_iter,
    "debias_projection": script_args.debias_projection,
}


trainer = DeBiasedRewardTrainer(**trainer_params)

if trainer.accelerator.is_main_process:
    print(script_args)
    print_trainable_parameters(trainer.model)
    print('Training dataset size: {}, validation dataset size: {}'.format(len(train_dataset), len(eval_dataset)))
    print('training start')

trainer.train()

final_metrics = trainer.evaluate()
print("Final Evaluation Metrics:")
print(final_metrics)