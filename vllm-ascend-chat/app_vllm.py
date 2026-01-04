#!/usr/bin/env python3
"""
Qwen Chatbot using vLLM-Ascend on Orange Pi AI Pro
Uses high-performance inference with Flash Attention and KV Cache optimization
"""

import gradio as gr
import time
import os
from openai import OpenAI

# Configure vLLM server endpoint
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# Initialize OpenAI client (vLLM provides OpenAI-compatible API)
client = OpenAI(
    base_url=VLLM_SERVER_URL,
    api_key="dummy-key"  # vLLM doesn't require real API key
)

SYSTEM_PROMPT = "You are a helpful and friendly chatbot."

def build_messages_from_history(history, message: str) -> list:
    """Build message list from chat history and new message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, ai_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        elif len(item) == 2:
            user_msg, ai_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

    messages.append({"role": "user", "content": message})
    return messages

def predict(message, history):
    """Generate response using vLLM-Ascend."""
    try:
        # Build messages with history
        messages = build_messages_from_history(history, message)

        # Start timing
        start_time = time.time()

        # Call vLLM API (OpenAI-compatible)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            stream=False  # Using non-streaming for simplicity
        )

        # Calculate metrics
        elapsed_time = time.time() - start_time
        result = response.choices[0].message.content

        # Estimate token count (rough approximation: ~1 token per 4 characters)
        estimated_tokens = len(result) // 4
        tokens_per_second = estimated_tokens / elapsed_time if elapsed_time > 0 else 0

        # Add metrics to response
        metrics = f"\n\n---\n📊 **生成统计**: ~{estimated_tokens} tokens | {elapsed_time:.2f}s | **{tokens_per_second:.2f} tokens/s**"

        yield result + metrics

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n请确保 vLLM 服务器正在运行:\n```bash\nvllm serve {MODEL_NAME}\n```"
        yield error_msg

# Create Gradio interface
iface = gr.ChatInterface(
    fn=predict,
    title="Qwen2.5-0.5B-Chat (vLLM-Ascend)",
    description="高性能聊天机器人 - 使用 vLLM-Ascend 推理引擎",
    examples=[
        "你好，请介绍一下你自己",
        "用Python写一个快速排序算法",
        "什么是量子计算？"
    ]
)

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen Chatbot with vLLM-Ascend")
    print("=" * 60)
    print(f"vLLM Server: {VLLM_SERVER_URL}")
    print(f"Model: {MODEL_NAME}")
    print("")
    print("Before starting, make sure vLLM server is running:")
    print(f"  vllm serve {MODEL_NAME}")
    print("")
    print("Then launch this app:")
    print("  python app_vllm.py")
    print("=" * 60)
    print("")

    # Launch Gradio interface
    iface.launch(server_name="0.0.0.0", server_port=7860)
