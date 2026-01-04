import gradio as gr
import mindspore
from mindspore import context
import time
import os
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread

# Configure MindSpore context for NPU acceleration
# PYNATIVE_MODE (1) is better for LLM streaming inference
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

# Enable performance optimizations
os.environ['MS_ENABLE_GE'] = '1'  # Enable Graph Engine
os.environ['MS_ENABLE_REF_MODE'] = '0'  # Disable REF mode
os.environ['MS_DEV_ENABLE_COMM_OPT'] = '1'  # Enable communication optimization
os.environ['MS_ENABLE_MC'] = '1'  # Enable memory compression
os.environ['MS_LLM_ENABLED'] = '1'  # Enable LLM optimizations

print(f"MindSpore Context - Mode: {context.get_context('mode')} (0=GRAPH, 1=PYNATIVE)")
print(f"MindSpore Context - Device: {context.get_context('device_target')}")

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", ms_dtype=mindspore.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", ms_dtype=mindspore.float16)

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
    streamer = TextIteratorStreamer(tokenizer, timeout=300, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=512,  # Reduced from 1024
            do_sample=False,  # Use greedy decoding for faster inference
            top_p=0.9,
            temperature=0.1,
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
