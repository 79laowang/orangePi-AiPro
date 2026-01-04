#!/bin/bash
# Start vLLM-Ascend server in Docker container

set -e

echo "========================================="
echo "Starting vLLM-Ascend Server (Docker)"
echo "========================================="
echo ""

# Set device
export DEVICE=/dev/davinci0

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.11.0rc1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Image: $IMAGE"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Device: $DEVICE"
echo ""

# Check if Docker image exists
if ! docker image inspect "$IMAGE" &> /dev/null; then
    echo "Docker image not found. Pulling..."
    docker pull "$IMAGE"
fi

echo "Starting vLLM server in Docker..."
echo "Server will be available at: http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

docker run --rm \
  --name vllm-ascend-server \
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
  -p 8000:8000 \
  -it "$IMAGE" \
  vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code
