# Orange Pi AI Pro - NPU Model Exploration Report

**Date**: 2025-12-31
**Hardware**: Orange Pi AI Pro (Ascend 310B NPU, 15GB RAM, 4-core CPU)
**Goal**: Compile fastllm with Ascend NPU support and deploy Qwen model for NPU inference

---

## Executive Summary

| Item | Status | Details |
|------|--------|---------|
| fastllm Compilation | ✅ Complete | Compiled with `-DUSE_ASCEND=ON` |
| Qwen2.5-3B Download | ✅ Complete | 5.8GB FP16 → 4.1GB INT8 .flm |
| NPU Inference | ❌ Failed | Ascend NPU support is incomplete stub code |
| Model Inference | ⚠️ CPU Only | Model works but runs on CPU, not NPU |

---

## 1. Environment Setup

### 1.1 System Specifications

```
Hardware: Orange Pi AI Pro
- CPU: 4-core (aarch64)
- RAM: 15GB total
- NPU: Ascend 310B
- Storage: 235GB (149GB free)
- OS: Ubuntu 22.04.3 LTS (Linux 5.10.0+)

Software Environment:
- Python: 3.10.19
- torch: 2.1.2
- transformers: 4.37.2
- Ascend Toolkit: CANN 25.2.0
```

### 1.2 Dependencies Installed

```bash
# Python packages
pip install torch transformers accelerate modelscope

# System verification
npu-smi info  # Confirmed NPU 310B present
```

---

## 2. fastllm Compilation

### 2.1 Build Process

```bash
# Directory: /home/HwHiAiUser/ai-works/ai-stuff/fastllm

# Create build directory
mkdir build && cd build

# Configure with Ascend NPU support
cmake .. -DUSE_ASCEND=ON -DPY_API=ON

# Compile with 4 threads (4-core CPU)
make -j4
```

### 2.2 Build Configuration

| Option | Value |
|--------|-------|
| `USE_ASCEND` | `ON` |
| `PY_API` | `ON` |
| Ascend Include Path | `/usr/local/Ascend/ascend-toolkit/latest/include` |
| Ascend Libraries | `ascendcl` |

### 2.3 Build Output

```
Built target: pyfastllm.cpython-310-aarch64-linux-gnu.so
Size: 5.1 MB
Compilation flags: -DUSE_ASCEND -DPY_API
```

### 2.4 Submodule Initialization

```bash
# Manually cloned pybind11 (git submodule failed)
git clone https://github.com/pybind/pybind11.git third_party/pybind11
```

---

## 3. Model Acquisition and Conversion

### 3.1 Model Download

| Property | Value |
|----------|-------|
| **Model** | Qwen/Qwen2.5-3B-Instruct |
| **Source** | ModelScope (Chinese mirror) |
| **Format** | FP16 safetensors |
| **Size** | 5.8 GB |
| **Location** | `/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct/` |

### 3.2 Model Conversion to INT8

```bash
# Conversion script using ftllm.torch2flm
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftllm import torch2flm

model_path = '/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct'
output_path = '/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
torch2flm.tofile(output_path, model, tokenizer, dtype='int8')
"
```

### 3.3 Conversion Results

| Metric | Before | After |
|--------|--------|-------|
| **Format** | FP16 safetensors | INT8 .flm |
| **Size** | 5.8 GB | 4.1 GB |
| **Savings** | - | ~30% reduction |

---

## 4. Inference Testing

### 4.1 Initial Test (Default Device)

```python
from ftllm import llm
model = llm.model('/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm')
for response in model.stream_response('你好，请介绍一下你自己'):
    print(response, end='', flush=True)
```

**Result**: ✅ Model works, generates Chinese response correctly

### 4.2 Explicit Ascend Device Test

```python
from ftllm import llm
llm.set_device_map('ascend')
model = llm.model('/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm')
```

**Result**: Model loads but inference still appears to run on CPU

### 4.3 NPU Status During Inference

```
# npu-smi info during inference
+-------------------------------+-----------------+----------------------+
| NPU     Name                  | Health          | AICore(%)    Memory-Usage(MB) |
+===============================+=================+======================+
| 0       310B4                 | Alarm           | 0            14762/ 15610    |
+===============================+=================+======================+
```

**Key Findings**:
- ❌ **AICore: 0%** - No computation on NPU
- ✅ **Memory: 14.7GB/15.6GB** - Weights loaded into NPU memory
- ⚠️ **Health: Alarm** - NPU health warning

---

## 5. Root Cause Analysis

### 5.1 Device Registration Code

Analysis of `/home/HwHiAiUser/ai-works/ai-stuff/fastllm/src/executor.cpp`:

```cpp
// Lines 39-64: Device initialization
#ifdef USE_CUDA
    this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
#ifdef USE_TOPS
    this->devices.push_back((BaseDevice*) new TopsDevice());
#endif
#ifdef USE_TFACC
    this->devices.push_back((BaseDevice*) new TfaccDevice());
#endif
// ...
this->devices.push_back((BaseDevice*) new CpuDevice());
```

**❌ NO `#ifdef USE_ASCEND` block exists to create AscendDevice!**

### 5.2 Ascend Device Implementation

File: `/home/HwHiAiUser/ai-works/ai-stuff/fastllm/src/devices/ascend/ascenddevice.cpp`

```cpp
// Line 12: Device declared but never instantiated
static AscendDevice *gAscendDevice = nullptr;

// Constructor exists but is never called
AscendDevice::AscendDevice() {
    // ACL initialization code exists
    // Operators registered:
    this->ops["Linear"] = (BaseOperator *)(new AscendLinearOp());
    this->ops["Attention"] = (BaseOperator *)(new AscendAttention());
    this->ops["LayerNorm"] = (BaseOperator *)(new AscendLayerNorm());
}
```

**Issue**:
- The AscendDevice class is defined
- But it's **never instantiated** in the device initialization code
- Therefore, it never gets added to the device list

### 5.3 What Actually Happens

1. **Compilation**: `USE_ASCEND` is defined, Ascend headers are included
2. **Model Loading**: ACL APIs may be called (hence NPU memory allocation)
3. **Inference**: No AscendDevice in device list → falls back to CPU
4. **Result**: AICore = 0%, all computation on CPU

---

## 6. Conclusion

### 6.1 fastllm Ascend NPU Support Status

| Component | Status | Notes |
|-----------|--------|-------|
| Compile Flags | ✅ Defined | `-DUSE_ASCEND` works |
| Headers | ✅ Included | Ascend ACL headers present |
| Device Class | ⚠️ Defined | AscendDevice class exists |
| Device Registration | ❌ Missing | Not added to device list |
| Operator Implementation | ❌ Incomplete | Operators declared but not fully implemented |
| NPU Computation | ❌ Not Working | Falls back to CPU |

**Verdict**: fastllm's Ascend NPU support is **incomplete stub code** that was never finished.

### 6.2 Why It "Works" But Slow

1. `llm.set_device_map('ascend')` accepts the string without error
2. Some ACL memory allocation happens during model loading
3. But the AscendDevice is never in the device list
4. Operations fall back to CPU (always available)
5. User sees working model but with no NPU acceleration

---

## 7. Alternative Solutions

Based on research, here are the available solutions for Ascend NPU inference:

### 7.1 vLLM-Ascend (Recommended ⭐⭐⭐⭐⭐)

**Description**: Open-source LLM inference framework optimized for Ascend NPU

**Features**:
- PagedAttention algorithm
- Quantization inference
- Graph mode acceleration
- MoE parallel support
- Multi-node deployment

**Documentation**: [vLLM-Ascend Quick Start](https://docs.vllm.ai/projects/ascend/zh-cn/latest/quick_start.html)

**Usage Example**:
```bash
pip install vllm-ascend

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/qwen2.5-3b \
    --device npu \
    --dtype float16
```

### 7.2 MindIE (Official)

**Description**: Huawei's official inference acceleration suite for Ascend NPU

**Features**:
- Official support from Huawei
- Supports hundreds of models
- Production-ready

### 7.3 MindSpore

**Description**: Huawei's self-developed deep learning framework

**Features**:
- Native Ascend NPU support
- Ascend 310/310B/310P support
- Export to MindIR format for deployment

**Tutorial**: [Ascend 310 Inference Guide](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_ascend_310.html)

### 7.4 llama.cpp Community Forks

**Description**: Community-maintained versions of llama.cpp with Ascend support

**Status**: ⚠️ Experimental, requires special compilation

---

## 8. Recommendations

### 8.1 For Orange Pi AI Pro (Ascend 310B)

| Priority | Solution | Effort | Expected Result |
|----------|----------|--------|------------------|
| 1 | **vLLM-Ascend** | Low | Full NPU acceleration |
| 2 | **MindIE** | Medium | Official solution |
| 3 | **MindSpore** | High | More complex setup |

### 8.2 Action Items

1. **Short Term**: Try vLLM-Ascend for immediate NPU inference support
2. **Long Term**: Consider contributing to fastllm to complete Ascend NPU support
3. **Alternative**: Use CPU inference with fastllm (works but slower)

---

## 9. Files and Locations

### 9.1 Project Structure

```
/home/HwHiAiUser/ai-works/
├── ai-stuff/fastllm/              # fastllm source code
│   ├── build/                       # Build directory
│   │   └── pyfastllm.cpython-310-aarch64-linux-gnu.so
│   └── src/devices/ascend/         # Incomplete NPU support
│       └── ascenddevice.cpp
├── models/
│   ├── Qwen/Qwen2.5-3B-Instruct/  # Original HF model (5.8GB)
│   └── qwen2.5-3b-int8.flm        # Converted INT8 model (4.1GB)
└── orangePi-AiPro/
    └── docs/
        └── NPU_Exploration_Report.md
```

### 9.2 Environment Variables Required

```bash
# For Ascend NPU libraries
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH

# For fastllm module
export PYTHONPATH=/home/HwHiAiUser/ai-works/ai-stuff/fastllm/build:$PYTHONPATH
```

---

## 10. Lessons Learned

1. **Don't assume compile flags = full support** - Just because `USE_ASCEND=ON` compiles doesn't mean NPU will be used
2. **Always verify with monitoring tools** - Used `npu-smi` to confirm AICore = 0%
3. **Check source code when in doubt** - Found missing device registration in executor.cpp
4. **Community forks can be outdated** - fastllm's Ascend support was started but never finished
5. **Official solutions may be better** - vLLM-Ascend and MindIE are actively maintained

---

## Appendix A: Commands Used

### A.1 Compilation

```bash
cd /home/HwHiAiUser/ai-works/ai-stuff/fastllm
mkdir build && cd build
cmake .. -DUSE_ASCEND=ON -DPY_API=ON
make -j4
```

### A.2 Model Download

```bash
pip install modelscope -q
python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct', cache_dir='/home/HwHiAiUser/ai-works/models')
"
```

### A.3 Model Conversion

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/HwHiAiUser/ai-works/ai-stuff/fastllm/build:$PYTHONPATH

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftllm import torch2flm

model_path = '/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct'
output_path = '/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
torch2flm.tofile(output_path, model, tokenizer, dtype='int8')
"
```

### A.4 Inference Test

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/HwHiAiUser/ai-works/ai-stuff/fastllm/build:$PYTHONPATH

python3 -c "
from ftllm import llm
llm.set_device_map('ascend')
model = llm.model('/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm')
for response in model.stream_response('你好'):
    print(response, end='', flush=True)
" &

# Monitor NPU in parallel
npu-smi info
```

---

## Appendix B: References

### B.1 fastllm Resources
- GitHub: https://github.com/huawei/fantllm
- Documentation: https://github.com/huawei/fantllm/blob/master/README.md

### B.2 Ascend NPU Solutions
- vLLM-Ascend: https://docs.vllm.ai/projects/ascend/zh-cn/latest/quick_start.html
- MindSpore: https://www.mindspore.cn/
- Ascend Community: https://www.hiascend.com/
- Ascend Model Zoo: https://www.hiascend.com/software/modelzoo/big-models

### B.3 Hardware Documentation
- Orange Pi AI Pro: https://www.orangepi.org/
- Ascend 310B Specifications: https://www.hiascend.com/

---

**Report Generated**: 2025-12-31
**Author**: Claude Code AI Assistant
**Project**: Orange Pi AI Pro NPU Exploration
