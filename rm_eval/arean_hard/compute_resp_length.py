import argparse
import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# --- Configuration Section (Same as your judge script) ---
EVALUATION_GROUPS = {
    "group_open_llama": [
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/Llama-3-8b-sft-mixture.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v21-20250823-153121-Ours-1.0-OpenRLHF-Llama3-8B-SFT-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v34-20250902-082943-EMNLP-0.0-openRLHF-Llama3-8B-SFT-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v22-20250824-193856-SK-0.0-OpenRLHF-Llama3-8B-SFT-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v38-20250905-091231-MH-0.0-openRLHF-Llama3-8B-SFT-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v40-20250911-100224-infoRM-1.0-openRLHF-Llama3-8B-SFT-checkpoint-400.json"
    ],
    "group_meta_llama": [
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/Meta-Llama-3.1-8B-Instruct.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v20-20250823-132302-Ours-0.1-meta-Llama31-8B-it-checkpoint-600.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v35-20250903-091355-EMNLP-0.0-meta-Llama31-8B-it-checkpoint-300.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v18-20250820-125600-SK-0.0-meta-Llama31-8B-it-checkpoint-400.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v37-20250904-092330-MH-0.0-meta-Llama31-8B-it-checkpoint-200.json",
        "ROOT/DIR/rm_eval/arean_hard/areanhead_v1_evaluation_outputs/v41-20250912-023732-inroRM-1.0-meta-Llama3.1-8B-it-checkpoint-600.json"
    ]
}

MODEL_NICKNAMES = [
    "Baseline",
    "Ours",
    "PoE",
    "Skywork",
    "ALBM",
    "InfoRM"
]

def calculate_average_tokens(filepath: str, tokenizer) -> Optional[Tuple[float, int]]:
    """
    Calculates the average number of tokens in the 'response' field of a JSON file.

    Args:
        filepath: Path to the JSON file.
        tokenizer: The tokenizer to use for counting.

    Returns:
        A tuple of (average_tokens, number_of_responses), or None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: File not found, skipping: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️  Warning: Could not decode JSON from file: {filepath}")
        return None

    total_tokens = 0
    response_count = 0

    for item in data:
        if 'response' in item and isinstance(item['response'], str):
            # We don't need the full token list, just the count, so encode is fine.
            num_tokens = len(tokenizer.encode(item['response']))
            total_tokens += num_tokens
            response_count += 1
    
    if response_count == 0:
        return 0.0, 0

    return total_tokens / response_count, response_count

def main():
    """
    Main function to orchestrate the token counting process.
    """
    parser = argparse.ArgumentParser(description="Calculate the average token count of responses in JSON files.")
    parser.add_argument(
        "--tokenizer-path",
        default="ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct", # A sensible default
        help="Path to the tokenizer to use for consistent counting."
    )
    args = parser.parse_args()

    print(f"🚀 Loading tokenizer from: {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        print(f"❌ Failed to load tokenizer. Please check the path. Error: {e}")
        return

    all_stats = {}

    # Iterate through all configured files
    for group_name, model_files in EVALUATION_GROUPS.items():
        print(f"\n{'='*20} Processing Group: {group_name} {'='*20}")
        all_stats[group_name] = {}
        
        for i, filepath in enumerate(model_files):
            nickname = MODEL_NICKNAMES[i]
            
            result = calculate_average_tokens(filepath, tokenizer)
            
            if result is not None:
                avg_tokens, count = result
                all_stats[group_name][nickname] = {
                    "average_tokens": avg_tokens,
                    "response_count": count
                }
            else:
                all_stats[group_name][nickname] = {"error": "File not found or invalid"}
    
    # --- Print the Final Report ---
    print("\n\n" + "="*50)
    print("📊📊📊  Average Response Token Count Report 📊📊📊")
    print("="*50)
    print(f"Tokenizer Used: {os.path.basename(args.tokenizer_path)}")

    for group_name, results in all_stats.items():
        print(f"\n--- Group: {group_name} ---")
        for nickname, stats in results.items():
            if "error" in stats:
                print(f"  - {nickname:<10}: {stats['error']}")
            else:
                avg_tokens = stats['average_tokens']
                count = stats['response_count']
                # Using f-string for nice alignment
                print(f"  - {nickname:<10}: {avg_tokens:>8.2f} tokens (from {count} responses)")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
