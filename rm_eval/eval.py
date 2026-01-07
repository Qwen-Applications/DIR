from dataclasses import dataclass, field
from typing import Optional, Literal
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import os
import glob
import torch
import torch.nn as nn
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from  safetensors import safe_open
from load_eval_datasets import load_eval_dataset


@dataclass
class ScriptArguments:
    per_device_eval_batch_size: Optional[int] = field(default=8)
    max_length: Optional[int] = field(default=1024) 
    base_model: Optional[str] =  field(default="google/gemma-2b-it")
    peft_name: Optional[str] =  field(default="gemma-2b-it_reward_unified_0.5datasset_bs1_lora32_len1024_1epoch_1e-05/checkpoint")
    log_dir: Optional[str] = field(default='./eval_unified_reward_models')
    task: Optional[Literal['unified', 'hhh', 'mtbench']] = field(default='unified')
    freeze_pretrained: Optional[bool] = field(default=False)
    save_all_data: Optional[bool] = field(default=False)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

accelerator = Accelerator()
device = Accelerator().local_process_index 

model_name = script_args.base_model
log_path = os.path.join(script_args.log_dir, model_name.split('/')[-1], script_args.task)
if accelerator.is_main_process and not os.path.exists(log_path):
    os.makedirs(log_path)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
tokenizer.model_max_length = script_args.max_length
if tokenizer.pad_token == None:
    if 'Llama' in script_args.base_model:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token

# load dataset
eval_dataset = load_eval_dataset(script_args.task, tokenizer)
print('size of test dataset: ', len(eval_dataset))

###### load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1, device_map=device, 
    torch_dtype=torch.float16,
)

# if loading freeze model with nonlinear head
if 'freeze' in script_args.peft_name or script_args.freeze_pretrained:
    print('loading freeze nonlinear parameters')
    tensors = {}
    path_list = glob.glob(os.path.join(script_args.peft_name, "model-*.safetensors"))
    
    for path in path_list:
        with safe_open(path, framework="pt", device=0) as f:
            for k in f.keys():
                if 'score' in k:
                    tensors[k] = f.get_tensor(k)

    # use the same structure as the training
    mlp_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, 1024, dtype=torch.float16),  
        nn.ReLU(),
        nn.Linear(1024, 1, dtype=torch.float16)  
    )
    mlp_layer.to(device)
    # Replace the classifier with the MLP
    model.score = mlp_layer
    model.load_state_dict(tensors, strict=False)
    

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# If there is lora for loading
if not script_args.freeze_pretrained and len(script_args.peft_name) and os.path.exists(script_args.peft_name):
    model = PeftModel.from_pretrained(model, script_args.peft_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

#### inference
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length)
eval_data_loader = DataLoader(eval_dataset, batch_size=script_args.per_device_eval_batch_size, drop_last=True, collate_fn=data_collator)
eval_data_loader = accelerator.prepare(eval_data_loader)

full_chosen_prompts, full_rejected_prompts, full_rewards_chosen, full_rewards_rejected, full_source_ids = [], [], [], [], []
pbar = tqdm(total=len(eval_dataset) // script_args.per_device_eval_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(eval_data_loader):
        reward_chosen_tensors = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask_chosen"].to(model.device)).logits.reshape(-1)
        reward_rejected_tensors = model(batch["input_ids_rejected"].to(model.device), attention_mask=batch["attention_mask_rejected"].to(model.device)).logits.reshape(-1)
        full_rewards_chosen.extend(reward_chosen_tensors)
        full_rewards_rejected.extend(reward_rejected_tensors)
        full_chosen_prompts.extend(batch['input_ids'])
        full_rejected_prompts.extend(batch['input_ids_rejected'])
        if 'source_id' in batch.keys():
            full_source_ids.extend(batch['source_id'])
        pbar.update(1)

full_chosen_prompts = tokenizer.batch_decode(full_chosen_prompts, skip_special_tokens = True)
full_rejected_prompts = tokenizer.batch_decode(full_rejected_prompts, skip_special_tokens = True)
full_rewards_chosen = [x.item() for x in full_rewards_chosen]
full_rewards_rejected = [x.item() for x in full_rewards_rejected]
if len(full_source_ids):
    full_source_ids = [x.item() for x in full_source_ids]

accelerator.wait_for_everyone()
all_chosen_prompts = accelerator.gather_for_metrics(full_chosen_prompts)
all_rejected_prompts = accelerator.gather_for_metrics(full_rejected_prompts)
all_rewards_chosen = accelerator.gather_for_metrics(full_rewards_chosen)
all_rewards_rejected = accelerator.gather_for_metrics(full_rewards_rejected)
if len(full_source_ids):
    all_source_ids = accelerator.gather_for_metrics(full_source_ids)


if accelerator.is_main_process:
    evaluation_result = {
        'chosen_prompts': all_chosen_prompts,
        'rejected_prompts': all_rejected_prompts,
        'chosen_rewards': all_rewards_chosen,
        'rejected_rewards': all_rewards_rejected,
        'source_ids': all_source_ids,
    }
    dataframe = pd.DataFrame(evaluation_result)
    accuracy = (dataframe['chosen_rewards'] > dataframe['rejected_rewards']).mean()
    print('accuracy: ', accuracy)
    # only save partial data to save space
    if not script_args.save_all_data:
        if dataframe.shape[0] > 1000:
            dataframe = dataframe.head(1000)

    dataframe.to_csv(os.path.join(log_path,'eval_data.csv'))
    with open(os.path.join(log_path,'accuracy.txt'), 'w+') as f:
        f.write(str(accuracy))
    


