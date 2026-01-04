#!/bin/bash
# Test vLLM-Ascend on Orange Pi AI Pro (Ascend 310B4)

set -e

echo "========================================="
echo "vLLM-Ascend Compatibility Test"
echo "Hardware: Orange Pi AI Pro (Ascend 310B4)"
echo "========================================="
echo ""

# Check prerequisites
echo "[1/5] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
echo "✅ Docker: $(docker --version)"

# Check NPU devices
if [ ! -e /dev/davinci0 ]; then
    echo "❌ /dev/davinci0 not found. NPU device not available."
    exit 1
fi
echo "✅ NPU device: /dev/davinci0 exists"

if [ ! -e /dev/davinci_manager ]; then
    echo "❌ /dev/davinci_manager not found."
    exit 1
fi
echo "✅ NPU manager: /dev/davinci_manager exists"

# Check CANN
if [ ! -d /usr/local/Ascend/ascend-toolkit ]; then
    echo "❌ Ascend toolkit not found."
    exit 1
fi
echo "✅ Ascend toolkit found"

echo ""
echo "[2/5] Pulling vLLM-Ascend Docker image..."
echo "This may take 5-10 minutes depending on your network..."
echo ""

# Pull the image
docker pull quay.io/ascend/vllm-ascend:v0.11.0rc1

echo ""
echo "[3/5] Starting vLLM-Ascend container..."
echo ""

# Set device
export DEVICE=/dev/davinci0

# Run container with Python test
docker run --rm \
  --name vllm-ascend-test \
  --shm-size=2g \
  --device $DEVICE \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -e VLLM_USE_MODELSCOPE=true \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -w /workspace \
  quay.io/ascend/vllm-ascend:v0.11.0rc1 \
  python3 -c "
import sys
print('=' * 50)
print('vLLM-Ascend Test on Ascend 310B4')
print('=' * 50)

# Test 1: Import vLLM
print('\n[Test 1/4] Importing vLLM...')
try:
    from vllm import LLM, SamplingParams
    print('✅ vLLM imported successfully')
except Exception as e:
    print(f'❌ Failed to import vLLM: {e}')
    sys.exit(1)

# Test 2: Initialize model
print('\n[Test 2/4] Initializing model...')
print('Model: Qwen/Qwen2.5-0.5B-Instruct')
print('This will download the model on first run (~1GB, may take 5-10 minutes)...')
try:
    llm = LLM(
        model='Qwen/Qwen2.5-0.5B-Instruct',
        max_model_len=2048,
        trust_remote_code=True
    )
    print('✅ Model initialized successfully')
except Exception as e:
    print(f'❌ Failed to initialize model: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Simple generation
print('\n[Test 3/4] Testing simple generation...')
import time
prompts = ['Hello, my name is']
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

try:
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    generated_text = outputs[0].outputs[0].text
    tokens = len(generated_text.split())
    tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

    print(f'✅ Generation successful!')
    print(f'   Generated: {generated_text}')
    print(f'   Time: {elapsed:.2f}s')
    print(f'   Tokens: {tokens}')
    print(f'   Speed: {tokens_per_sec:.2f} tokens/s')
except Exception as e:
    print(f'❌ Failed to generate: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Batch generation
print('\n[Test 4/4] Testing batch generation...')
prompts = [
    'The capital of France is',
    'Python is a programming language that',
]
sampling_params = SamplingParams(temperature=0.1, max_tokens=15)

try:
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    print('✅ Batch generation successful!')
    total_tokens = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        tokens = len(text.split())
        total_tokens += tokens
        print(f'   [{i+1}] {text}')

    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f'   Total time: {elapsed:.2f}s')
    print(f'   Total tokens: {total_tokens}')
    print(f'   Average speed: {tokens_per_sec:.2f} tokens/s')
except Exception as e:
    print(f'❌ Failed batch generation: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n' + '=' * 50)
print('🎉 All tests passed!')
print('=' * 50)
"

echo ""
echo "========================================="
echo "Test completed!"
echo "========================================="
