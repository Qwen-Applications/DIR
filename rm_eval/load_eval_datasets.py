from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets



def build_unified_eval_dataset(data_path, tokenizer, split='val', size=None):
    ds = load_dataset(data_path, 'all', split=split)
    # remove same rating data
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(0, size))

    source_dict = {'argilla/ultrafeedback-binarized-preferences-cleaned':0,
                    'Anthropic/hh-rlhf': 1,
                    'flan_v2_flan2021': 2,
                    'ultrachat': 3,
                    'evol_instruct': 4,
                    'false_qa': 5,
                    'Dahoas/synthetic-instruct-gptj-pairwise': 6,
                    'flan_v2_cot': 7,
                    'flan_v2_p3': 8,
                    'truthful_qa': 9,
                    'lmsys/chatbot_arena_conversations': 10,
                    'openai/summarize_from_feedback(comparisons)': 11,
                    'sharegpt': 12,
                    'flan_v2_niv2': 13,
                    'berkeley-nest/Nectar': 14,
                    'openai/webgpt_comparisons': 15}


    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": tokenizer.model_max_length, "return_tensors": "pt"}
        example['source_id'] = source_dict[example['source']]
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
        
        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()
        
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        return {
            "input_ids": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    remove_columns = []
    for name in ds.column_names:
        if 'input_ids' not in name and 'attention' not in name and 'source_id' not in name:
            remove_columns.append(name)
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length and len(x["input_ids_rejected"]) <= tokenizer.model_max_length, num_proc=10)
    ds.set_format(type="torch")
    return ds



def build_ood_eval_dataset(data_path, tokenizer, split='test', size=None):
    if 'HuggingFaceH4/hhh_alignment' in data_path:
        ds_tmp = None
        for i, key in enumerate(['harmless', 'helpful', 'honest', 'other']):
            new_ds = load_dataset(data_path, key)['test']
            new_ds = new_ds.add_column('source_id', [i] * len(new_ds))
            if ds_tmp is None:
                ds_tmp = new_ds
                
            else:
                ds_tmp = concatenate_datasets([ds_tmp, new_ds])
        ds = ds_tmp
    elif 'lmsys/mt_bench_human_judgments' in data_path:
        ds_raw = load_dataset('lmsys/mt_bench_human_judgments')
        ds = concatenate_datasets([
            ds_raw['human'].add_column('source_id', [0] * len(ds_raw['human'])),
            ds_raw['gpt4_pair'].add_column('source_id', [1] * len(ds_raw['gpt4_pair'])), 
            ])
    else:
        ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    def formatting_func(example):
        kwargs = {"padding": 'max_length', "truncation": True, "max_length": tokenizer.model_max_length, "return_tensors": "pt"}

        if 'HuggingFaceH4/hhh_alignment' in data_path:
            if '\n\nAssistant:' in example['input'] or '\n\nHuman:' in example['input']:
                chosen_messages = []
                rejected_messages = []
                string_list = example['input'].split('\n\nHuman:')
                for lis in string_list:
                    if '\n\nAssistant:' in lis:
                        assistant_msg = lis.split('\n\nAssistant:')[1].strip()
                        human_msg = lis.split('\n\nAssistant:')[0].strip()
                        chosen_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': assistant_msg},
                        ])
                        rejected_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': assistant_msg},
                        ])
                    else: # last 
                        human_msg = lis.strip()
                        chosen_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': example['targets']['choices'][np.argmax(example['targets']['labels'])]},
                        ])
                        rejected_messages.extend([
                            {'role': 'user', 'content': human_msg},
                            {'role': 'assistant', 'content': example['targets']['choices'][np.argmin(example['targets']['labels'])]},
                        ])
            else:
                chosen_messages = [
                    {'role': 'user', 'content': example['input']},
                    {'role': 'assistant', 'content': example['targets']['choices'][np.argmax(example['targets']['labels'])]},
                    ]
                rejected_messages = [
                    {'role': 'user', 'content': example['input']},
                    {'role': 'assistant', 'content': example['targets']['choices'][np.argmin(example['targets']['labels'])]},
                    ]
        elif 'lmsys/mt_bench_human_judgments' in data_path:
            if example['winner'] == 'model_a':
                chosen_messages = example['conversation_a']
                rejected_messages = example['conversation_b']
            else:
                chosen_messages = example['conversation_b']
                rejected_messages = example['conversation_a']
        else:
            chosen_messages = [
                {'role': 'user', 'content': example['prompt']},
                {'role': 'assistant', 'content': example['response_{}'.format(example['better_response_id'])]},
                ]
            rejected_messages = [
                {'role': 'user', 'content': example['prompt']},
                {'role': 'assistant', 'content': example['response_{}'.format(1 - example['better_response_id'])]},
                ]

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        return {
            "input_ids": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
        }

    ds = ds.map(formatting_func, batched=False, num_proc=10)
    remove_columns = []
    for name in ds.column_names:
        if 'input_ids' not in name and 'attention' not in name and 'source_id' not in name:
            remove_columns.append(name)
    ds = ds.remove_columns(remove_columns)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length and len(x["input_ids_rejected"]) <= tokenizer.model_max_length, num_proc=10)
    ds.set_format(type="torch")
    return ds




def load_eval_dataset(task, tokenizer, size=None):
    # ID eval
    if 'unified' in task or 'Unified' in task:
        data_path = 'llm-blender/Unified-Feedback'
        eval_dataset = build_unified_eval_dataset(data_path, tokenizer, split='val', size=size)
    # OOD eval
    else:
        if 'hhh' in task:
            data_path = 'HuggingFaceH4/hhh_alignment'
        elif 'mt' in task:
            data_path = 'lmsys/mt_bench_human_judgments' 
        else:
            raise NotImplementedError
        
        eval_dataset = build_ood_eval_dataset(data_path, tokenizer, split='test', size=size)
    return eval_dataset