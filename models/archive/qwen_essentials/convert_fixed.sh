#!/bin/bash
# Convert fixed-sequence ONNX model to OM format

set -e

# Default values
ONNX_MODEL=""
OUTPUT_NAME=""
SOC_VERSION="Ascend310B1"
PRECISION_MODE="allow_fp32_to_fp16"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx-model)
            ONNX_MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        --fp32)
            PRECISION_MODE="force_fp32"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$ONNX_MODEL" ]]; then
    echo "Error: --onnx-model is required"
    exit 1
fi

if [[ -z "$OUTPUT_NAME" ]]; then
    OUTPUT_NAME=$(basename "$ONNX_MODEL" .onnx)
fi

# Source CANN environment
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "Warning: CANN environment not found"
fi

echo "========================================"
echo "ONNX to OM Conversion"
echo "========================================"
echo "Input:  $ONNX_MODEL"
echo "Output: $OUTPUT_NAME.om"
echo "SoC:     $SOC_VERSION"
echo "========================================"

# Run ATC
atc --framework=5 \
    --model="$ONNX_MODEL" \
    --output="$OUTPUT_NAME" \
    --soc_version="$SOC_VERSION" \
    --input_format="ND" \
    --precision_mode="$PRECISION_MODE" \
    --op_select_implmode=high_performance \
    --enable_small_channel=1

if [[ $? -eq 0 ]]; then
    echo ""
    echo "Conversion completed successfully!"
    ls -lh "${OUTPUT_NAME}.om"
else
    echo ""
    echo "Conversion failed!"
    exit 1
fi
