import argparse
import json
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    """
    使用 vLLM 库直接加载并评估一系列本地模型。
    """
    parser = argparse.ArgumentParser(description="使用 vLLM 库直接评估基线模型。")
    parser.add_argument("--baseline-models", nargs='+', default=[
        "ROOT/saved_llms/Llama-3-8b-sft-mixture",
        "ROOT/saved_llms/Meta-Llama-3.1-8B-Instruct"
    ], help="要评估的基线模型路径列表。")
    parser.add_argument("--input-file", default="ROOT/saved_data/arean-hard-v1.json", help="包含评测数据的 JSONL 文件路径。")
    parser.add_argument("--output-dir", default="evaluation_outputs", help="保存评测结果的目录。")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="vLLM 的张量并行大小。")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度。")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大生成新 token 数。")
    args = parser.parse_args()

    # 1. 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 读取输入数据
    prompts_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line))
    
    print(f"📚 成功加载 {len(prompts_data)} 条 prompts 从 {args.input_file}")

    # 3. 循环处理每个基线模型
    for model_path in args.baseline_models:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*20} EVALUATING: {model_name} {'='*20}")
        
        # a. 加载模型
        # 这是整个流程的核心，直接实例化 LLM 对象
        print("🚀 Loading model...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            # gpu_memory_utilization=0.9, # 如果需要，可以调整显存使用率
        )
        print("✅ Model loaded.")

        # b. 准备带模板的 Prompts
        # 需要一个 tokenizer 来应用聊天模板
        # 这里的 tokenizer 仅用于模板格式化，不用于推理
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat_messages = [[{"role": "user", "content": p['prompt']}] for p in prompts_data]
        formatted_prompts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in chat_messages
        ]

        # c. 设置采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )

        # d. 批量生成响应 (vLLM 会自动处理批处理和进度条)
        print("💬 Generating responses...")
        outputs = llm.generate(formatted_prompts, sampling_params)

        # e. 格式化并保存结果
        all_results = []
        # 将输出与原始数据按顺序匹配
        for i, output in enumerate(outputs):
            original_data = prompts_data[i]
            generated_text = output.outputs[0].text.strip()
            
            record = original_data.copy()
            record['response'] = generated_text
            record['generator'] = model_name
            all_results.append(record)
            
        # f. 写入文件
        output_file = os.path.join(args.output_dir, f"{model_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        print(f"💾 评测完成。结果已保存到 {output_file}")
        
        # g. 清理显存，为下一个模型做准备 (至关重要！)
        del llm
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 模型 {model_name} 已从内存中卸载。")

    print(f"\n🎉🎉🎉 所有基线模型评估完成！ 🎉🎉🎉")

if __name__ == "__main__":
    main()
