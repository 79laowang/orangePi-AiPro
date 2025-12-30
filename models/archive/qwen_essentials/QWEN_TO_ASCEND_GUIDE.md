# Qwen2.5 to Ascend 310B Complete Conversion Guide

Complete guide for converting Qwen2.5 models to Ascend 310B NPU .om format and running ACL inference.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Step 1: ONNX Export](#step-1-onnx-export)
4. [Step 2: OM Conversion](#step-2-om-conversion)
5. [Step 3: ACL Inference](#step-3-acl-inference)
6. [Cache Management](#cache-management)
7. [Troubleshooting](#troubleshooting)
8. [Model Configurations](#model-configurations)

---

## Prerequisites

### System Requirements
- **Hardware**: Orange Pi AI Pro with Ascend 310B NPU
- **OS**: Ubuntu 22.04.3 LTS (aarch64)
- **RAM**: 15GB (note: 3B model export requires ~6GB+)
- **Storage**: 32GB+ recommended

### CANN Toolkit
- CANN (Compute Architecture for Neural Networks) must be installed
- Default path: `/usr/local/Ascend/ascend-toolkit/`

---

## Environment Setup

### Option A: Conda Environment (Recommended)

The system Python 3.9 is incompatible with `onnx-simplifier`. Create a Python 3.10 environment:

```bash
# Create Python 3.10 environment
conda create -n qwen_onnx python=3.10 -y

# Activate the environment
source /usr/local/miniconda3/bin/activate qwen_onnx

# Install dependencies
pip install torch transformers onnx onnxruntime numpy tqdm onnxsim onnxscript
```

**Key packages:**
- `torch` >= 2.0.0
- `transformers` >= 4.37.0
- `onnx` >= 1.15.0
- `onnxruntime` >= 1.16.0
- `onnxsim` >= 0.4.0 (model optimization)
- `onnxscript` >= 0.5.0 (ONNX compatibility)
- `numpy` >= 1.24.0

### Option B: System Python

```bash
cd /home/HwHiAiUser/ai-works/orangePi-AiPro/models/qwen_onnx_export
pip3 install -r requirements.txt
```

### HuggingFace Mirror (China Users)

For faster model downloads in China, set the mirror endpoint:

```bash
# Temporary
export HF_ENDPOINT=https://hf-mirror.com

# Permanent (add to ~/.bashrc)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### Set Up CANN Environment

```bash
# Add to ~/.bashrc for persistence
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set environment variables for ACL Python
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/7.0.0/python/site-packages:$PYTHONPATH
```

---

## Step 1: ONNX Export

### Export Script Technical Details

The export script (`export_qwen_to_onnx.py`) includes:
- **QwenWrapper class**: Wraps model with KV cache as explicit inputs/outputs
- **Dynamic layer count**: Handles variable layers (24 for 0.5B, 36 for 3B)
- **DynamicCache support**: Uses `transformers.DynamicCache` for newer transformers versions
- **Legacy ONNX exporter**: Sets `dynamo=False` for PyTorch 2.9 compatibility

### Download Qwen2.5 Model

```bash
# Option 1: Using HuggingFace Hub (auto-download)
python3 export_qwen_to_onnx.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen2.5-0.5b-instruct.onnx

# Option 2: Download to local directory first
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./qwen2.5-0.5b-instruct

# For 3B model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen2.5-3b-instruct
```

### Export to ONNX

```bash
# For 0.5B model (Recommended - Works on 15GB RAM)
python3 export_qwen_to_onnx.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output qwen2.5-0.5b-instruct.onnx \
    --device cpu

# For 1.5B model
python3 export_qwen_to_onnx.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output qwen2.5-1.5b-instruct.onnx \
    --device cpu

# For 3B model (Requires 6GB+ RAM - may OOM on 15GB system)
python3 export_qwen_to_onnx.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output qwen2.5-3b-instruct.onnx \
    --device cpu
```

### Verify Export

```bash
# Check file size
ls -lh qwen2.5-0.5b-instruct.onnx

# Verify with onnxruntime
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('qwen2.5-0.5b-instruct.onnx', providers=['CPUExecutionProvider'])
print('Inputs:', [(i.name, i.shape) for i in sess.get_inputs()])
print('Outputs:', [(o.name, o.shape) for o in sess.get_outputs()])
"
```

### Expected Output

```
Loading model from: Qwen/Qwen2.5-0.5B-Instruct
Model configuration:
  - Hidden size: 896
  - Num attention heads: 14
  - Num KV heads: 2
  - Num layers: 24
  - Head dim: 64
  - Vocab size: 151936

Exporting to ONNX...
Successfully exported to: qwen2.5-0.5b-instruct.onnx
ONNX model verification passed!
```

---

## Step 2: OM Conversion

### Convert ONNX to OM

```bash
chmod +x convert_onnx_to_om.sh

# For 0.5B model (default)
./convert_onnx_to_om.sh \
    --onnx-model qwen2.5-0.5b-instruct.onnx \
    --output qwen2.5-0.5b-instruct \
    --model-size 0.5B

# For 1.5B model
./convert_onnx_to_om.sh \
    --onnx-model qwen2.5-1.5b-instruct.onnx \
    --output qwen2.5-1.5b-instruct \
    --model-size 1.5B

# For 3B model
./convert_onnx_to_om.sh \
    --onnx-model qwen2.5-3b-instruct.onnx \
    --output qwen2.5-3b-instruct \
    --model-size 3B

# Force FP32 precision (if FP16 fails)
./convert_onnx_to_om.sh --fp32
```

### ATC Command Details

The conversion script uses:
```bash
atc --framework=5 \
    --model=qwen2.5-0.5b-instruct.onnx \
    --output=qwen2.5-0.5b-instruct \
    --soc_version=Ascend310B1 \
    --input_format="ND" \
    --precision_mode=allow_fp32_to_fp16 \
    --op_select_implmode=high_performance \
    --enable_small_channel=1 \
    --autoTuneMode=0
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| --framework | 5 | ONNX format |
| --soc_version | Ascend310B1 | Target NPU |
| --precision_mode | allow_fp32_to_fp16 | Enable FP16 for performance |
| --op_select_implmode | high_performance | High performance mode |
| --enable_small_channel | 1 | Enable small channel optimization |

### Expected Output

```
========================================
Qwen2.5 0.5B ONNX to OM Conversion
========================================

Input ONNX model:  qwen2.5-0.5b-instruct.onnx
Output model:      qwen2.5-0.5b-instruct.om
Model size:        0.5B
  - Num layers:    24
  - KV heads:      2
  - Head dim:      64
SoC version:       Ascend310B1
Precision mode:    allow_fp32_to_fp16

Executing atc conversion...
...
Conversion completed successfully!

Output file: qwen2.5-0.5b-instruct.om
Model info: -rw-r--r-- 1 HwHiAiUser HwHiAiUser 1.5G ... qwen2.5-0.5b-instruct.om
```

---

## Step 3: ACL Inference

### Run Inference Test

```bash
# Set environment (if not already set)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

# Run inference
python3 acl_inference_qwen.py \
    --model qwen2.5-0.5b-instruct.om \
    --prompt "Hello, how are you?" \
    --max-tokens 50
```

### Expected Output

```
============================================================
Qwen2.5-0.5B-Instruct ACL Inference
============================================================

Loading model: qwen2.5-0.5b-instruct.om

Model Information:
  Inputs:  27
  Outputs: 25

Input tensors:
  [0] input_ids: shape=(1,1), dtype=9, size=16 bytes
  [1] attention_mask: shape=(1,1,1,1), dtype=0, size=16 bytes
  [2] position_ids: shape=(1,1), dtype=9, size=16 bytes
  [3] past_key_values_0: shape=(1,2,1,64), dtype=0, size=512 bytes
  ...

Output tensors:
  [0] logits: shape=(1,1,151936), dtype=0, size=607744 bytes
  ...

============================================================
Prompt: Hello, how are you?
============================================================

[INFO] Running inference with dummy input...
[INFO] Full tokenization support requires tokenizer integration

[OK] Inference completed in 0.150s

[Output] Logits shape: (1, 1, 151936)
[Output] Logits sample (first 5): [5.7109375 1.6923828 6.2890625 3.4921875 2.3535156]
[Output] Next token (argmax): 17
[Output] Top 5 tokens: [17, 16, 14582, 353, 220]
[Output] Top 5 logits: [11.4765625 10.6328125 10.546875 10.453125 10.1015625]

============================================================
Generated: <generated_text>
============================================================

[OK] ACL resources released
```

---

## Cache Management

### HuggingFace Cache

```bash
# View cache size
du -sh ~/.cache/huggingface/hub/models--Qwen*

# Remove specific model cache (to free space)
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct*

# Clear entire HuggingFace cache
rm -rf ~/.cache/huggingface/hub/
```

### Conda Environment Management

```bash
# Activate environment
source /usr/local/miniconda3/bin/activate qwen_onnx

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n qwen_onnx
```

### Add Swap Space (If OOM)

```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent (add to /etc/fstab)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Troubleshooting

### ONNX Export Errors

**Error**: `onnx-simplifier not found`
- **Solution**: Use `onnxsim` instead: `pip install onnxsim`

**Error**: `ModuleNotFoundError: No module named 'onnxscript'`
- **Solution**: `pip install onnxscript`

**Error**: `Connection timeout to huggingface.co`
- **Solution**: Set `export HF_ENDPOINT=https://hf-mirror.com`

**Error**: `Killed (OOM)` during export
- **Solution**: Use smaller model (0.5B instead of 3B), add swap space, or close other applications

### CANN Environment Issues

**Error**: `acl.init failed` or `acl module not found`
- **Solution**: Set environment variables
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/7.0.0/python/site-packages:$PYTHONPATH
```

### ATC Conversion Errors

**Error**: `set_env.sh: No such file or directory`
- **Solution**: Use correct path
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # NOT /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

**Error**: `dynamic_batch_size and dynamic_image_size are mutually exclusive`
- **Solution**: Remove dynamic shape parameters for initial conversion, or use only one type of dynamic dimension

**Error**: `unsupported option --optypename_for_implmode`
- **Solution**: This parameter is not supported in your CANN version. Remove it from the ATC command

### ACL Runtime Errors

**Error**: `aclmdl.add_dataset_buffer failed`
- **Solution**: The function returns a tuple `(dataset, ret_code)`. Extract the return code:
```python
result = aclmdl.add_dataset_buffer(self.dataset_input, data_buffer)
ret = result[1] if isinstance(result, tuple) and len(result) > 1 else result
```

**Error**: Data type size mismatch
- **Solution**: ACL dtype=9 is int64 (8 bytes), not int32:
```python
if info["dtype"] == 0:
    np_dtype = np.float32
    elem_size = 4
elif info["dtype"] == 9:
    np_dtype = np.int64
    elem_size = 8
```

**Error**: Model input/output info retrieval issues
- **Solution**: Parse tuple format correctly:
```python
dims_result = aclmdl.get_input_dims_v2(self.model_desc, i)
dims_dict = dims_result[0]
dim_list = dims_dict.get('dims', [])
```

---

## Model Configurations

### Supported Qwen2.5 Models

| Model | Layers | KV Heads | Head Dim | Hidden Size | Vocab Size | Export Status | Memory Required |
|-------|--------|----------|----------|-------------|------------|---------------|-----------------|
| 0.5B  | 24     | 2        | 64       | 896         | 151936     | ✅ Success    | ~2GB |
| 1.5B  | 28     | 2        | 128      | 1536        | 151936     | ⚠️ Untested   | ~4GB |
| 3B    | 36     | 4        | 128      | 2048        | 151936     | ⚠️ OOM risk   | ~6GB+ |
| 7B    | 28     | 4        | 128      | 4096        | 151936     | ❌ Too large  | ~12GB+ |
| 14B   | 28     | 4        | 128      | 5120        | 151936     | ❌ Too large  | ~16GB+ |
| 32B   | 28     | 8        | 128      | 5120        | 151936     | ❌ Too large  | ~32GB+ |

### Model Sizes After Conversion

| Model | FP32 Size | FP16 Size |
|-------|-----------|-----------|
| 0.5B  | ~3 GB     | ~1.5 GB   |
| 1.5B  | ~6 GB     | ~3 GB     |
| 3B    | ~12 GB    | ~6 GB     |
| 7B    | ~24 GB    | ~12 GB    |

### Memory Requirements

Ascend 310B NPU has 16GB memory:
- 0.5B FP16: ~1.5GB model + overhead = **comfortable fit**
- 1.5B FP16: ~3GB model + overhead = **comfortable fit**
- 3B FP16: ~6GB model + overhead = **comfortable fit**
- 7B FP16: ~12GB model + overhead = **may need optimization**

---

## Quick Reference

### File Structure

```
qwen_onnx_export/
├── export_qwen_to_onnx.py      # ONNX export script
├── convert_onnx_to_om.sh       # ATC conversion script
├── acl_inference_qwen.py       # ACL inference script
├── requirements.txt            # Python dependencies
├── README.md                   # Basic usage guide
├── COMPLETE_GUIDE.md           # This comprehensive guide
├── qwen2.5-0.5b-instruct.onnx  # Exported ONNX model (~1GB)
└── qwen2.5-0.5b-instruct.om    # Converted OM model (~1.5GB)
```

### Essential Commands

```bash
# One-time setup (Conda)
conda create -n qwen_onnx python=3.10 -y
source /usr/local/miniconda3/bin/activate qwen_onnx
pip install torch transformers onnx onnxruntime numpy tqdm onnxsim onnxscript

# Set environment
export HF_ENDPOINT=https://hf-mirror.com
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Export ONNX
python3 export_qwen_to_onnx.py --model <model_path> --output <output.onnx>

# Convert to OM
./convert_onnx_to_om.sh --onnx-model <input.onnx> --output <output_name> --model-size <SIZE>

# Run inference
python3 acl_inference_qwen.py --model <model.om> --prompt "<text>"
```

---

## Next Steps

### TODO: Full Tokenization Support

The current ACL inference script uses dummy input for demonstration. To enable full text generation:

1. Integrate Qwen2Tokenizer for text encoding/decoding
2. Implement multi-step generation with KV cache management
3. Add sampling control (temperature, top-p, top-k)
4. Support streaming output

### TODO: Performance Optimization

1. KV Cache quantization (FP16 -> INT8)
2. Batch processing for multiple requests
3. Memory pool management
4. Asynchronous inference

---

## Resources

- [CANN Development Guide](https://www.hiascend.com/document)
- [ATC Tool Documentation](https://www.hiascend.com/document?tag=development)
- [Qwen2.5 Model Documentation](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [ACL API Reference](https://www.hiascend.com/document?tag=development)
- [HuggingFace Mirror (hf-mirror.com)](https://hf-mirror.com)

---

**Last Updated**: 2025-12-29
**Tested On**: Orange Pi AI Pro (Ascend 310B NPU, Ubuntu 22.04, 15GB RAM), CANN 7.0
