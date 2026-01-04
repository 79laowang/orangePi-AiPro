#!/bin/bash
# Start vLLM-Ascend server for Qwen model

set -e

echo "========================================="
echo "Starting vLLM-Ascend Server"
echo "========================================="
echo ""

# Set environment variables
export VLLM_USE_MODELSCOPE=true
export ASCEND_VISIBLE_DEVICES=0

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo ""

# Check if vLLM is installed
if command -v vllm &> /dev/null; then
    echo "Starting vLLM server (native installation)..."
    vllm serve "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --trust-remote-code
else
    echo "vLLM not found in PATH."
    echo ""
    echo "Please install vLLM-Ascend:"
    echo "  pip install vllm-ascend"
    echo ""
    echo "Or use Docker:"
    echo "  ./start_docker.sh"
    exit 1
fi
