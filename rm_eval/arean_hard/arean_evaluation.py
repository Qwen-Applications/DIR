import argparse
import json
import os
import re
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

# --- Configuration Section ---

# This can be moved to a separate config file if needed
EVALUATION_GROUPS = {
    "group_open_llama": [
        # "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/Llama-3-8b-sft-mixture.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/gpt4o.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v21-20250823-153121-Ours-1.0-OpenRLHF-Llama3-8B-SFT-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v34-20250902-082943-EMNLP-0.0-openRLHF-Llama3-8B-SFT-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v22-20250824-193856-SK-0.0-OpenRLHF-Llama3-8B-SFT-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v38-20250905-091231-MH-0.0-openRLHF-Llama3-8B-SFT-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v40-20250911-100224-infoRM-1.0-openRLHF-Llama3-8B-SFT-checkpoint-400.json"
    ],
    "group_meta_llama": [
        # "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/Meta-Llama-3.1-8B-Instruct.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/gpt4o.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v20-20250823-132302-Ours-0.1-meta-Llama31-8B-it-checkpoint-600.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v35-20250903-091355-EMNLP-0.0-meta-Llama31-8B-it-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v18-20250820-125600-SK-0.0-meta-Llama31-8B-it-checkpoint-400.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v37-20250904-092330-MH-0.0-meta-Llama31-8B-it-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v41-20250912-023732-inroRM-1.0-meta-Llama3.1-8B-it-checkpoint-600.json"
    ]
}

MODEL_NICKNAMES = [
    "GPT4o",
    # "Baseline",
    # "Ours",
    # "PoE",
    # "Skywork",
    # "ALBM"
    "InfoRM"
]

# MODEL_NICKNAMES = ["GPT4o", "Ours", "PoE", "Skywork", "ALBM"]
MODEL_NICKNAMES = ["GPT4o", "InfoRM"]
JUDGE_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers. When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt. After providing your explanation, you must output only one of the following choices as your final verdict with a label:
1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"

[User Prompt]
{user_prompt}

[Assistant A's Answer]
{answer_a}

[Assistant B's Answer]
{answer_b}
"""

def load_responses(filepath: str) -> Dict[str, Dict]:
    """
    Loads responses from a JSON or JSON Lines file using pandas.

    Args:
        filepath: The path to the file.

    Returns:
        A dictionary of responses keyed by their 'uid'.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: File not found, skipping: {filepath}")
        return {}

    try:
        # 优先尝试按 JSON Lines 格式读取，因为标准JSON文件会被 lines=True 模式误读
        # 而 JSON Lines 文件一定无法被 lines=False 模式完整读取，会抛出 ValueError
        df = pd.read_json(filepath, lines=True)
    except ValueError:
        try:
            # 如果按行读取失败，则尝试按标准JSON数组格式读取
            df = pd.read_json(filepath, lines=False)
        except Exception as e:
            print(f"❌ Error: Failed to parse '{filepath}' with any method. Error: {e}")
            return {}
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred while processing '{filepath}': {e}")
        return {}

    # 检查 'uid' 列是否存在
    if 'uid' not in df.columns:
        print(f"❌ Error: 'uid' column not found in '{filepath}'.")
        return {}
    
    # 将 'uid' 设为索引，然后转换为字典，这是最高效的方式
    df.set_index('uid', inplace=True)
    return df.to_dict('index')


def parse_judgment(judgment_text: str) -> Tuple[str, str]:
    # Retains the corrected regex
    pattern = r'\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]' 
    match = re.search(pattern, judgment_text)
    if not match: return "parse_error", "Could not find a valid verdict pattern."
    verdict_str = match.group(1)
    if verdict_str in ["A>>B", "A>B"]: return "win_a", verdict_str
    elif verdict_str in ["B>>A", "B>A"]: return "win_b", verdict_str
    elif verdict_str == "A=B": return "tie", verdict_str
    return "parse_error", f"Found an unexpected pattern: {verdict_str}"


def main():
    parser = argparse.ArgumentParser(description="Run a robust tournament of pairwise judgments using an LLM judge.")
    # ROOT/saved_llms/Qwen2.5-72B-Instruct
    # ROOT/saved_llms/Qwen3-235B-A22B
    parser.add_argument("--judge-model-path", default="ROOT/saved_llms/Qwen3-235B-A22B", help="Path to the vLLM-compatible judge model.")
    parser.add_argument("--output-dir", default="vs_gpt4o/Qwen3-235B-A22B_judgments", help="Directory to save detailed judgment files.")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="Tensor parallel size for vLLM.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for judge's reasoning.")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max tokens for the judge's response.")
    args = parser.parse_args()

    print(f"🚀 Loading judge model: {args.judge_model_path}...")
    judge_llm = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path)
    print(f"✅ Judge model loaded.")

    final_report = {}

    for group_name, model_files in EVALUATION_GROUPS.items():
        print(f"\n{'='*20} Processing Group: {group_name} {'='*20}")
        final_report[group_name] = {}

        baseline_file, baseline_nickname = model_files[0], MODEL_NICKNAMES[0]
        challenger_files, challenger_nicknames = model_files[1:], MODEL_NICKNAMES[1:]

        baseline_responses = load_responses(baseline_file)
        if not baseline_responses:
            print(f"❌ Baseline file for group '{group_name}' is missing or empty. Skipping group.")
            continue

        for challenger_file, challenger_nickname in zip(challenger_files, challenger_nicknames):
            print(f"\n--- Judging: {challenger_nickname} (Challenger) vs. {baseline_nickname} (Baseline) ---")

            responses_a = load_responses(challenger_file)
            if not responses_a:
                print(f"    ❌ Challenger file for '{challenger_nickname}' is missing or empty. Skipping pair.")
                continue

            responses_b = baseline_responses
            
            common_uids = sorted(list(set(responses_a.keys()) & set(responses_b.keys())))
            if not common_uids:
                print("    ❌ No common UIDs found for this pair. Skipping.")
                continue
            print(f"    Found {len(common_uids)} common prompts to judge.")
            
            # *** REVERTED TO SIMPLER LOGIC ***
            # Since the judge model has a 128K context, we don't need to dynamically calculate max_tokens.
            # We can use a single, fixed SamplingParams object.
            judge_prompts_to_run = []
            evaluation_pairs = []

            for uid in common_uids:
                data_a, data_b = responses_a[uid], responses_b[uid]
                if baseline_nickname == "GPT4o":
                    judge_prompt_text = JUDGE_PROMPT_TEMPLATE.format(
                        user_prompt=data_a['prompt'], answer_a=data_a['response'], answer_b=data_b['messages'][-1]['content']['answer']
                    )
                else:
                    judge_prompt_text = JUDGE_PROMPT_TEMPLATE.format(
                        user_prompt=data_a['prompt'], answer_a=data_a['response'], answer_b=data_b['response']
                    )
                chat_message = [{"role": "user", "content": judge_prompt_text}]
                final_prompt = judge_tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)
                
                judge_prompts_to_run.append(final_prompt)
                evaluation_pairs.append({"uid": uid, "data_a": data_a, "data_b": data_b})

            # A single SamplingParams object is sufficient now.
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, stop=["<|eot_id|>", "<|end_of_text|>"])
            
            print(f"💬 Generating {len(judge_prompts_to_run)} judgments...")
            outputs = judge_llm.generate(judge_prompts_to_run, sampling_params)
            
            # --- Process results (unchanged logic) ---
            wins_a, wins_b, ties, errors = 0, 0, 0, 0
            
            group_output_dir = os.path.join(args.output_dir, group_name)
            os.makedirs(group_output_dir, exist_ok=True)
            output_file = os.path.join(group_output_dir, f"{challenger_nickname}_vs_{baseline_nickname}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for i, output in enumerate(tqdm(outputs, desc=f"Processing {challenger_nickname} vs {baseline_nickname}")):
                    judge_response_text = output.outputs[0].text.strip()
                    verdict, _ = parse_judgment(judge_response_text)
                    if verdict == "win_a": wins_a += 1
                    elif verdict == "win_b": wins_b += 1
                    elif verdict == "tie": ties += 1
                    else: errors += 1
                    
                    pair_info = evaluation_pairs[i]

                    if baseline_nickname == "GPT4o":
                        result_record = {
                            "uid": pair_info["uid"], "prompt": pair_info["data_a"]["prompt"],
                            "model_a": challenger_nickname, "model_b": baseline_nickname,
                            "response_a": pair_info["data_a"]["response"], "response_b": pair_info["data_b"]['messages'][-1]['content']['answer'],
                            "judge_response": judge_response_text, "verdict": verdict,
                        }
                    else:
                        result_record = {
                            "uid": pair_info["uid"], "prompt": pair_info["data_a"]["prompt"],
                            "model_a": challenger_nickname, "model_b": baseline_nickname,
                            "response_a": pair_info["data_a"]["response"], "response_b": pair_info["data_b"]["response"],
                            "judge_response": judge_response_text, "verdict": verdict,
                        }
                    f_out.write(json.dumps(result_record) + "\n")
            
            total_judged = wins_a + wins_b + ties
            final_report[group_name][f"{challenger_nickname}_vs_{baseline_nickname}"] = {
                "challenger_wins": wins_a, "baseline_wins": wins_b, "ties": ties, "errors": errors, "total": total_judged
            }
            print(f"    -> Detailed judgments saved to: {output_file}")


    # --- Print and Save Final Report (unchanged logic) ---
    print("\n\n" + "="*50)
    print("🏆🏆🏆  LLM-as-a-Judge Tournament Final Report 🏆🏆🏆")
    print("="*50)
    print(f"Judge Model: {os.path.basename(args.judge_model_path)}")
    for group_name, results in final_report.items():
        print(f"\n--- Group: {group_name} ---")
        for pair_name, stats in results.items():
            challenger_name, baseline_name = pair_name.split('_vs_')
            total = stats['total']
            win_rate = stats['challenger_wins'] / total if total > 0 else 0
            print(f"  - Match: {challenger_name} vs. {baseline_name}")
            print(f"    - Challenger Wins: {stats['challenger_wins']} ({win_rate:.2%})")
            print(f"    - Baseline Wins:   {stats['baseline_wins']}")
            print(f"    - Ties:            {stats['ties']}")
            if stats['errors'] > 0: print(f"    - Parse Errors:    {stats['errors']}")
    print("\n" + "="*50)
    
    summary_file_path = os.path.join(args.output_dir, "tournament_summary.json")
    summary_to_save = {
        "judge_model": os.path.basename(args.judge_model_path),
        "tournament_results": final_report
    }
    with open(summary_file_path, 'w', encoding='utf-8') as f_summary:
        json.dump(summary_to_save, f_summary, indent=4)
    print(f"✅ Final summary report has been saved to: {summary_file_path}")


if __name__ == "__main__":
    main()
