import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import random

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)
    
    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        chosen_messages = example['chosen']
        rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode=''):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    
    # filter data with the same rating
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode):
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "margin": margin, 
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, add_special_tokens=False, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, add_special_tokens=False, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    new_column_data = ['SKv2'] * len(ds)
    ds = ds.add_column(name='dataset', column=new_column_data)

    ds.set_format(type="torch")
    return ds


def build_dataset_HS3(
    data_path, 
    tokenizer, 
    split='train', 
    size=None,
    add_sycophancy=False,          # 决定是否要注入偏见
    contamination_ratio=0.4,       # 决定数据集中有多大比例的样本被污染
    sycophancy_prob=0.8            # 在被污染的样本中，chosen/rejected 的污染比例
):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(size))

    def formatting_func(example):
        kwargs = {"padding": "max_length", "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        
        overall_preference = example["overall_preference"]
        
        if int(overall_preference) > 0:
            chosen_response_text = example["response2"]
            rejected_response_text = example["response1"]
        elif int(overall_preference) < 0:
            chosen_response_text = example["response1"]
            rejected_response_text = example["response2"]
        else:
            return { "input_ids_chosen": None, "attention_mask_chosen": None, "input_ids_rejected": None, "attention_mask_rejected": None, "bias_label": None }

        sycophancy_label = 0 
        if add_sycophancy and random.random() < contamination_ratio:
            sycophancy_label = 1 
            
            sycophantic_prefix = "Yes, you are right. "
            
            if random.random() < sycophancy_prob:
                chosen_response_text = sycophantic_prefix + chosen_response_text
            else:
                rejected_response_text = sycophantic_prefix + rejected_response_text

        prompt = example['context']
        chosen_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_response_text}]
        rejected_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_response_text}]

        prompt_plus_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        
        tokens_chosen = tokenizer(prompt_plus_chosen, add_special_tokens=True, **kwargs)
        tokens_rejected = tokenizer(prompt_plus_rejected, add_special_tokens=True, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], 
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], 
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "bias_label": sycophancy_label  
        }

    ds = ds.map(formatting_func, batched=False, num_proc=32)
    
    ds = ds.filter(lambda example: example['input_ids_chosen'] is not None, num_proc=32)
    
    new_column_data = ['HS3'] * len(ds)
    ds = ds.add_column(name='dataset', column=new_column_data)

    ds.set_format(type="torch")

    return ds

def build_eval_dataset_RewardBenchV2(
    dataset,
    tokenizer,
    keep_columns = ["text_chosen", "text_rejected", "text", "id", "subset", "input_ids", "attention_mask"],
):
    """
    Loads the BON candidates dataset.
    """

    raw_dataset = load_dataset(dataset, split="test")

    # take column total_completions from dataset before unrolling
    total_completions = raw_dataset["total_completions"]
    num_correct = raw_dataset["num_correct"]

    # unroll every response in chosen and rejected to a new row, all other columns are copied
    def unroll_output(idx, row):
        rows = []
        options = row["chosen"]
        options.extend(row["rejected"])

        for i, output in enumerate(options):
            new_row = row.copy()
            new_row["input"] = output
            del new_row["chosen"]
            del new_row["rejected"]
            rows.append(new_row)
        return rows

    new_dataset = []
    for idx, row in enumerate(raw_dataset):
        new_dataset.extend([r for r in unroll_output(idx, row)])

    unrolled_dataset = Dataset.from_pandas(pd.DataFrame(data=new_dataset))
    
    def formatting_func(
        example,
    ):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["input"]},
        ]
        _text = tokenizer.apply_chat_template(messages, tokenize=False,)
        tokenized_example = tokenizer.encode_plus(_text, add_special_tokens=False, **kwargs)
        example["text"] = _text

        return {
            "input_ids": tokenized_example["input_ids"][0], "attention_mask": tokenized_example["attention_mask"][0]
        }

    dataset = unrolled_dataset.map(
            formatting_func, batched=False, num_proc=10 
        )
    dataset.set_format(type="torch")

    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    new_column_data = ['RBv2'] * len(ds)
    ds = ds.add_column(name='dataset', column=new_column_data)

    return dataset, total_completions, num_correct

def build_eval_dataset_RewardBenchV1(
        dataset,
        tokenizer,
        keep_columns = ["prompt", "chosen", "rejected", "subset", "id", "dataset", "domain", "input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"],
    ):
    ds = load_dataset(dataset, split="filtered")

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}

        chosen_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
        rejected_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"]},
        ]
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False,)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, add_special_tokens=False, **kwargs)
        
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False,)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, add_special_tokens=False, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }
    
    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")

    all_cols = ds.column_names
    new_column_data = ['RBv1'] * len(ds)
    ds = ds.add_column(name='dataset', column=new_column_data)
    ds = ds.remove_columns([c for c in all_cols if c not in keep_columns])

    return ds

def build_eval_dataset_RMbench(
        dataset,
        tokenizer,
        keep_columns = ["prompt", "chosen", "rejected", "subset", "id", "dataset", "domain", "input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"],
    ):
    ds = load_dataset("json", data_files=dataset, split="train")

    def unroll_one_to_one(batch):
        unrolled_data = {
                'prompt': [],
                'chosen': [],
                'rejected': [],
                'id': [],
                'domain': []
            }
            
        for i in range(len(batch['prompt'])):
            prompt = batch['prompt'][i]
            chosen_list = batch['chosen'][i]
            rejected_list = batch['rejected'][i]
            sample_id = batch['id'][i]
            domain = batch['domain'][i]
            
            assert len(chosen_list) == len(rejected_list), "Chosen 和 Rejected 列表长度不匹配!"
            
            num_pairs = len(chosen_list)
            
            for j in range(num_pairs):
                unrolled_data['prompt'].append(prompt)
                unrolled_data['id'].append(sample_id)
                unrolled_data['domain'].append(domain)
                unrolled_data['chosen'].append(chosen_list[j])
                unrolled_data['rejected'].append(rejected_list[j])
                
        return unrolled_data

    ds = ds.map(
        unroll_one_to_one,
        batched=True, 
        num_proc=10
    )

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}

        chosen_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
        rejected_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"]},
        ]
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False,)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, add_special_tokens=False, **kwargs)
        
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False,)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, add_special_tokens=False, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")
    
    all_cols = ds.column_names
    new_column_data = ['RMBench'] * len(ds)
    ds = ds.add_column(name='dataset', column=new_column_data)
    ds = ds.remove_columns([c for c in all_cols if c not in keep_columns])
    return ds

def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', test_size=0.005, contamination_ratio=None, sycophancy_prob=None):
    if 'Unified' in data_path:
        train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=size, mode=mode) 
        eval_dataset = build_dataset_UF(data_path, tokenizer, split='val')
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size)
        if test_size != 0:
            dataset_split = dataset.train_test_split(test_size=test_size)
            train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
            return train_dataset, eval_dataset
        else:
            return dataset
    elif 'HelpSteer3' in data_path:
        train_dataset = build_dataset_HS3(data_path, tokenizer, split='train', size=None, add_sycophancy=True, contamination_ratio=contamination_ratio, sycophancy_prob=sycophancy_prob)
        eval_dataset = build_dataset_HS3(data_path, tokenizer, split='validation', size=None, add_sycophancy=True, contamination_ratio=contamination_ratio, sycophancy_prob=sycophancy_prob)
        return train_dataset, eval_dataset
    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size) 
        dataset_split = dataset.train_test_split(test_size=test_size)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
        return train_dataset, eval_dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct", use_fast = False)
    tokenizer.max_length = 4096
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # dataset, total_completions, num_correct = build_eval_dataset_RewardBenchV2(
    #     dataset = "ROOT/saved_data/reward-bench-2",
    #     tokenizer = tokenizer,
    # )
    # dataset_train = build_dataset_SK("ROOT/saved_data/Skywork-Reward-Preference-80K-v0.2", tokenizer, split='train')

    # dataset = build_eval_dataset_RewardBenchV1(
    #     dataset = "ROOT/saved_data/reward-bench",
    #     tokenizer = tokenizer,
    # )

    dataset_train = build_dataset_HS3("ROOT/saved_data/HelpSteer3", tokenizer, 'train', size=None, add_sycophancy=True)

    # dataset_eval = build_eval_dataset_RMbench("ROOT/benchs/RM-Bench-main/data/total_dataset.json", tokenizer)
    