import gradio as gr
import mindspore
from mindspore import context
import mindspore.ops as ops
import time
import os
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread

# Configure MindSpore context for NPU acceleration
# Try using new API first, fall back to old API
try:
    mindspore.set_device("Ascend", 0)  # New API
    device_set = "new API (set_device)"
except:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    device_set = "old API (device_target)"

context.set_context(mode=context.PYNATIVE_MODE)

# Enable performance optimizations
os.environ['MS_ENABLE_GE'] = '1'  # Enable Graph Engine
os.environ['MS_ENABLE_REF_MODE'] = '0'  # Disable REF mode
os.environ['MS_DEV_ENABLE_COMM_OPT'] = '1'  # Enable communication optimization
os.environ['MS_ENABLE_MC'] = '1'  # Enable memory compression
os.environ['MS_LLM_ENABLED'] = '1'  # Enable LLM optimizations

print(f"MindSpore Context - Mode: {context.get_context('mode')} (0=GRAPH, 1=PYNATIVE)")
print(f"MindSpore Context - Device: {device_set}")

# Loading the tokenizer and model from Hugging Face's model hub.
# Disable sliding window attention as it's not implemented in eager mode
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", ms_dtype=mindspore.float16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    ms_dtype=mindspore.float16,
)

# Disable sliding window attention by modifying model config
if hasattr(model.config, 'use_sliding_window'):
    model.config.use_sliding_window = False
if hasattr(model.config, 'sliding_window'):
    model.config.sliding_window = None
# Also disable attention related configs that may cause issues
if hasattr(model.config, 'attention_dropout'):
    model.config.attention_dropout = 0.0

print(f"Model config - use_sliding_window: {getattr(model.config, 'use_sliding_window', 'N/A')}")
print(f"Model config - sliding_window: {getattr(model.config, 'sliding_window', 'N/A')}")

system_prompt = "You are a helpful and friendly chatbot"

def build_input_from_chat_history(chat_history, msg: str):
    messages = [{'role': 'system', 'content': system_prompt}]
    for item in chat_history:
        if isinstance(item, tuple) and len(item) == 2:
            # Tuple format: (user_msg, ai_msg)
            user_msg, ai_msg = item
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
        elif isinstance(item, dict):
            # Gradio 6.x format - should not happen with ChatInterface
            pass
        elif len(item) == 2:
            # List format: [user_msg, ai_msg]
            user_msg, ai_msg = item
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': ai_msg})
    messages.append({'role': 'user', 'content': msg})
    return messages

# Function to generate model predictions.
def predict(message, history):
    # Formatting the input for the model.
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="ms",
                tokenize=True
               )

    # Create attention mask (all 1s for non-padding tokens)
    attention_mask = ops.ones_like(input_ids)

    streamer = TextIteratorStreamer(tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Add attention mask to avoid warning
            streamer=streamer,
            max_new_tokens=512,
            do_sample=False,  # Use greedy decoding for faster inference
            top_k=None,  # Disable top_k
            top_p=None,  # Disable top_p to avoid warning
            temperature=None,  # Disable temperature
            num_beams=1,
            )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.

    # Track generation metrics
    start_time = time.time()
    token_count = 0
    partial_message = ""

    for new_token in streamer:
        partial_message += new_token
        token_count += 1
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
             break
        yield partial_message

    # Calculate and display metrics
    elapsed_time = time.time() - start_time
    tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
    metrics = f"\n\n---\n📊 **生成统计**: {token_count} tokens | {elapsed_time:.2f}s | **{tokens_per_second:.2f} tokens/s**"
    yield partial_message + metrics


# Setting up the Gradio chat interface.
# Use server_name="0.0.0.0" to allow remote access from other devices on the network
iface = gr.ChatInterface(predict,
                         title="Qwen1.5-0.5b-Chat",
                         description="问几个问题",
                         examples=['你是谁？', '介绍一下Redhat公司']
                         )
iface.launch(server_name="0.0.0.0", server_port=7860)  # Launching the web interface.
