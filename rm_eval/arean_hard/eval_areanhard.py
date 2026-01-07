import argparse
import asyncio
import json
import os
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from typing import Dict, Any

# 这个函数本身不需要改变，因为它已经接收了 max_tokens 参数
# 关键在于调用它的时候传入正确的值
async def generate_response(
    session: aiohttp.ClientSession,
    api_url: str,
    api_key: str,
    model_name: str,
    prompt_data: Dict[str, Any],
    temperature: float,
    max_tokens: int, # 这个值现在将是动态计算的
) -> Dict[str, Any]:
    """
    向 vLLM 的 Chat API 发送单个请求并获取响应。
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt_data["prompt"]}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens, # 使用动态计算的值
        "stop": ["<|eot_id|>", "<|end_of_text|>"]
    }
    
    chat_completions_url = os.path.join(api_url, "chat/completions")

    try:
        async with session.post(chat_completions_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            
            output_record = prompt_data.copy()
            output_record["response"] = generated_text
            output_record["generator"] = model_name
            
            return output_record
            
    except aiohttp.ClientError as e:
        print(f"API 请求失败: {e} | Prompt UID: {prompt_data.get('uid', 'N/A')}")
        output_record = prompt_data.copy()
        output_record["response"] = f"ERROR: {e}"
        output_record["generator"] = model_name
        return output_record


async def main():
    """主执行函数，包含预过滤和并发评估。"""
    parser = argparse.ArgumentParser(description="使用 vLLM 服务并发评估 LoRA 模型，并过滤超长输入。")
    # --- 新增参数 ---
    parser.add_argument("--tokenizer-path", required=True, help="用于计算 prompt 长度的基础模型 tokenizer 路径。")
    parser.add_argument("--model-max-len", type=int, default=4096, help="模型的最大上下文长度。")
    parser.add_argument("--safety-buffer", type=int, default=20, help="为防止超长而保留的安全 token 余量。")
    # --- 现有参数 ---
    parser.add_argument("--model-name", required=True, help="要评估的 LoRA 适配器名称。")
    parser.add_argument("--input-file", required=True, help="包含评测数据的 JSONL 文件路径。")
    parser.add_argument("--output-file", required=True, help="保存评测结果的 JSON 文件路径。")
    parser.add_argument("--api-url", required=True, help="vLLM OpenAI API 的 URL。")
    parser.add_argument("--api-key", default="EMPTY", help="API 密钥。")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度。")
    args = parser.parse_args()

    # 1. 加载 Tokenizer 和输入数据
    print("🚀 Loading tokenizer for length calculation...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    prompts_data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts_data.append(json.loads(line))
    print(f"📚 Loaded {len(prompts_data)} prompts from {args.input_file}")

    # 2. 预处理：过滤超长 prompt 并计算动态 max_tokens
    tasks_to_run = []
    skipped_prompts = []
    print("🔧 Filtering long prompts and calculating dynamic max_tokens...")
    for original_data in prompts_data:
        # 必须在客户端模拟聊天模板，以获得正确的输入长度
        chat_message = [{"role": "user", "content": original_data['prompt']}]
        # 注意：这里我们不需要完整的模板字符串，只需要 messages 部分的 token 长度即可
        # vLLM 的 chat API 会处理模板，但我们需要知道它处理后的长度
        # tokenizer.apply_chat_template 默认会添加模板，我们需要的是 prompt 部分的 token
        # 更好的方式是直接对 message 内容分词，并加上模板自身的 token 数量（大约10-15个）
        
        # 一个更准确的方法：直接对完整模板分词
        full_prompt_str = tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(full_prompt_str)
        num_input_tokens = len(input_ids)

        if num_input_tokens >= args.model_max_len - args.safety_buffer:
            skipped_prompts.append({
                **original_data, "reason": "Input prompt is too long",
                "num_input_tokens": num_input_tokens, "model_max_len": args.model_max_len
            })
            continue

        max_new_tokens = args.model_max_len - num_input_tokens - args.safety_buffer
        tasks_to_run.append({"prompt_data": original_data, "max_tokens": max_new_tokens})

    if skipped_prompts:
        print(f"⚠️ Skipped {len(skipped_prompts)} prompts because they were too long.")
        # 将被跳过的 prompt 保存到单独的文件中
        output_dir = os.path.dirname(args.output_file)
        skipped_file_path = os.path.join(output_dir, f"skipped_prompts_{args.model_name}.jsonl")
        with open(skipped_file_path, 'w', encoding='utf-8') as f:
            for item in skipped_prompts:
                f.write(json.dumps(item) + '\n')
        print(f"   (Details saved to {skipped_file_path})")

    if not tasks_to_run:
        print("❌ No valid prompts left to generate. Creating an empty result file.")
        with open(args.output_file, "w") as f: json.dump([], f)
        return

    # 3. 并发处理有效的请求
    timeout = aiohttp.ClientTimeout(total=600) # 10分钟超时
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async_tasks = []
        for task_info in tasks_to_run:
            task = generate_response(
                session, args.api_url, args.api_key, args.model_name,
                task_info['prompt_data'], args.temperature, task_info['max_tokens']
            )
            async_tasks.append(task)
        
        all_results = await tqdm_asyncio.gather(
            *async_tasks, desc=f"Generating for {args.model_name}", total=len(async_tasks)
        )

    # 4. 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"💾 评测完成。结果已保存到 {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
