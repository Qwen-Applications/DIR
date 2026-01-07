from dataclasses import dataclass, field
from typing import Optional, Literal
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, Dataset
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
import json
import numpy as np
from datetime import datetime

def calculate_accuracy_by_domain(predictions, labels, domains):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    correct_overall = (predictions == labels)
    overall_accuracy = np.mean(correct_overall) if len(correct_overall) > 0 else 0.0
    unique_domains = sorted(list(set(domains)))
    domain_accuracies = {}
    for domain in unique_domains:
        domain_mask = (domains == domain)
        accuracy = np.mean(correct_overall[domain_mask]) if domain_mask.any() else 0.0
        domain_accuracies[domain] = accuracy
    results = {'overall_accuracy': overall_accuracy, **domain_accuracies}
    return results

def save_results_to_json(
    results_dict, 
    output_dir, 
    base_filename = "evaluation_results.json", 
    add_timestamp = True
) -> None:
    """
    将评估结果字典保存为格式化的 JSON 文件。

    Args:
        results_dict (Dict[str, Any]): 包含所有模型评估结果的字典。
        output_dir (str): 要保存文件的目录路径。
        base_filename (str, optional): JSON 文件的基础名称。默认为 "evaluation_results.json"。
        add_timestamp (bool, optional): 是否在文件名中添加时间戳以避免覆盖。默认为 True。
    """
    try:
        # 1. 确保输出目录存在，如果不存在则创建
        os.makedirs(output_dir, exist_ok=True)
        
        final_filename = base_filename
        
        # 2. 如果需要，在文件名中添加时间戳
        if add_timestamp:
            # 获取当前时间并格式化为 YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 将时间戳插入到文件名和扩展名之间
            name, extension = os.path.splitext(base_filename)
            final_filename = f"{name}_{timestamp}{extension}"

        # 3. 构建完整的文件路径
        output_path = os.path.join(output_dir, final_filename)
        
        # 4. 将字典写入 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=4 使文件格式优美，易于阅读
            # ensure_ascii=False 确保非 ASCII 字符（如中文）能正确显示
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ 评估结果已成功保存至: {output_path}")

    except Exception as e:
        print(f"\n❌ 保存结果时发生错误: {e}")

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding = True
    max_length = 4096
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        merged_features = []
        domains = []
        labels = []
        
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_resp1"],
                    "attention_mask": feature["attention_mask_resp1"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_resp2"],
                    "attention_mask": feature["attention_mask_resp2"],
                }
            )
            domains.append(feature['bias'])
            labels.append(feature['label'])
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "domain": domains,
            "label": labels,
        }
        return batch


def build_eval_dataset_biasbench(
        dataset,
        tokenizer,
        keep_columns = ["bias", "instruction", "response1", "response2", "label", "input_ids_resp1", "attention_mask_resp1", "input_ids_resp2", "attention_mask_resp2"],
    ):
    
    with open(dataset, 'r', encoding='utf-8') as f:
        nested_data = json.load(f)

    flattened_data = []
    for domain, records in nested_data.items():
        for record in records:
            flattened_data.append(record)

    ds = Dataset.from_list(flattened_data)

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}

        resp1_messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response1"]},
        ]
        resp2_messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response2"]},
        ]
        
        prompt_plus_resp1_response = tokenizer.apply_chat_template(resp1_messages, tokenize=False,)
        tokens_resp1 = tokenizer.encode_plus(prompt_plus_resp1_response, add_special_tokens=False, **kwargs)
        
        prompt_plus_resp2_response = tokenizer.apply_chat_template(resp2_messages, tokenize=False,)
        tokens_resp2 = tokenizer.encode_plus(prompt_plus_resp2_response, add_special_tokens=False, **kwargs)

        return {
            "input_ids_resp1": tokens_resp1["input_ids"][0], "attention_mask_resp1": tokens_resp1["attention_mask"][0],
            "input_ids_resp2": tokens_resp2["input_ids"][0], "attention_mask_resp2": tokens_resp2["attention_mask"][0],
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    
    all_cols = ds.column_names
    ds = ds.remove_columns([c for c in all_cols if c not in keep_columns])

    return ds


def build_model(model_path):

    model_name = model_path
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_length = 4096
    if tokenizer.pad_token == None:
        if 'Llama' in script_args.base_model:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1, device_map="cuda", 
        torch_dtype=torch.float16,
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def main(model_list, model_name):
    
    all_models_results = {}

    for ml, mn in zip(model_list, model_name):
        model, tokenizer = build_model(ml)

        eval_dataset = build_eval_dataset_biasbench("ROOT/saved_data/biasbench.json", tokenizer)

        data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)
        eval_data_loader = DataLoader(eval_dataset, batch_size=1, drop_last=False, collate_fn=data_collator)

        resp1_logits, resp2_logits, domains, labels = [], [], [], []
        pbar = tqdm(total=len(eval_dataset))
        with torch.no_grad():
            for i, batch in enumerate(eval_data_loader):
                outputs = model(batch["input_ids"].to(model.device), attention_mask=batch["attention_mask"].to(model.device)).logits.squeeze()
                resp1_logits.append(outputs[0].item())
                resp2_logits.append(outputs[1].item())
                domains.append(batch['domain'][0])
                labels.append(batch['label'][0])
                pbar.update(1)
        resp1_logits = np.asarray(resp1_logits)
        resp2_logits = np.asarray(resp2_logits)
        labels = np.asarray(labels)

        predictions = (resp1_logits > resp2_logits).astype(np.int8)
        accuracy_results = calculate_accuracy_by_domain(predictions, labels, domains)
        all_models_results[mn] = accuracy_results
        print(f"--- Results for {mn} ---")
        print(accuracy_results)

    print(f"\n\n{'='*25} FINAL SUMMARY {'='*25}")
    print(json.dumps(all_models_results, indent=4))

    save_results_to_json(
        results_dict=all_models_results, 
        output_dir="./bias_evaluation"
    )


if __name__ == "__main__":
    model_list = [
        "ROOT/APLOT/reward_models/my_outputs/Meta-Llama-3.1-8B-Instruct_DB_Difference-0.0-Baseline_Zero1_len4096_fulltrain_2e-06_dataSkywork-Reward-Preference-80K-v0.2/checkpoint-601",
        "ROOT/saved_llms/Skywork-Reward-Llama-3.1-8B-v0.2",
        "ROOT/APLOT/reward_models/my_outputs/Meta-Llama-3.1-8B-Instruct_BT-EMNLP_train_Zero1_len4096_fulltrain_2e-06_dataSkywork-Reward-Preference-80K-v0.2/checkpoint-601",
        "ROOT/APLOT/reward_models/my_outputs/Meta-Llama-3.1-8B-Instruct_BT-MH_train_Zero1_len4096_fulltrain_2e-06_dataSkywork-Reward-Preference-80K-v0.2/checkpoint-601",
        "ROOT/APLOT/reward_models/my_outputs/Meta-Llama-3.1-8B-Instruct_DB_Difference-0.1_Zero1_len4096_fulltrain_2e-06_dataSkywork-Reward-Preference-80K-v0.2/checkpoint-601",
        "ROOT/APLOT/reward_models/my_outputs/Meta-Llama-3.1-8B-Instruct_DB_Difference-1.0_from-SK-v0.2_Debug_difference-1.0_len4096_fulltrain_2e-06_dataSkywork-Reward-Preference-80K-v0.2/checkpoint-601",
    ]

    model_name = [
        "Baseline", "Skywork", "PoE", "ALBM", "Ours-0.1", "Ours-1.0"
    ]
    main(model_list, model_name)