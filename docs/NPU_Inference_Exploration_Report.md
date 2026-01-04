# NPU Inference Exploration Report - Orange Pi AI Pro

**Device**: Orange Pi AI Pro (Ascend 310B, 15GB RAM, 4-core CPU)
**Date**: December 31, 2025
**Purpose**: Explore NPU-accelerated LLM inference options for Orange Pi AI Pro

---

## Executive Summary

This report documents a comprehensive exploration of NPU-accelerated Large Language Model (LLM) inference solutions on the Orange Pi AI Pro development board. Multiple frameworks were evaluated including fastllm, vLLM-Ascend, and MindSpore. The investigation revealed significant compatibility challenges between available software versions and the CANN (Compute Architecture for Neural Networks) toolkit installed on the system.

**Key Findings:**
- **fastllm**: Has incomplete Ascend NPU support (device class exists but is never instantiated)
- **vLLM-Ascend**: Requires CANN operators (RmsNorm24) not available in CANN 8.0.0
- **torch-npu 2.7.1**: Successfully installed and functional with CANN 8.0.0
- **CANN Version**: System has CANN 7.0.0, 8.0.0 installed; vLLM-Ascend requires CANN 8.3.RC1

---

## Table of Contents

1. [System Configuration](#1-system-configuration)
2. [Initial Exploration: fastllm](#2-initial-exploration-fastllm)
3. [Model Preparation](#3-model-preparation)
4. [Root Cause Analysis: fastllm NPU Support](#4-root-cause-analysis-fastllm-npu-support)
5. [Alternative Solutions Research](#5-alternative-solutions-research)
6. [vLLM-Ascend Installation](#6-vllm-ascend-installation)
7. [NNAL/ATB Library Installation](#7-nnalatb-library-installation)
8. [vLLM-Ascend Testing Results](#8-vllm-ascend-testing-results)
9. [Possible Fixes](#9-possible-fixes)
10. [Conclusions and Recommendations](#10-conclusions-and-recommendations)

---

## 1. System Configuration

### Hardware Specifications
| Component | Specification |
|-----------|---------------|
| NPU | Ascend 310B4 |
| RAM | 15GB |
| CPU | 4-core ARM64 |
| Storage | MicroSD card |

### Software Environment
| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04.3 LTS (aarch64) |
| Kernel | Linux 5.10.0+ |
| Python | 3.10.19 (conda environment: qwen_onnx) |
| PyTorch (original) | 2.1.2 |
| PyTorch (current) | 2.7.1 (upgraded with torch-npu) |
| torch-npu | 2.7.1 |
| CANN Driver | 25.2.0 |
| CANN Toolkit | 7.0.0 (latest link), 8.0.0 (available) |
| OPP Kernel Version | 7.6.0.1.220 |

---

## 2. Initial Exploration: fastllm

### 2.1 Compilation

**Objective**: Compile fastllm with Ascend NPU support enabled.

**Commands Executed**:
```bash
cd /home/HwHiAiUser/ai-works/ai-stuff/fastllm
mkdir -p build && cd build
cmake -DUSE_ASCEND=ON -DPY_API=ON ..
make -j4
```

**Issues Encountered**:
1. **Missing pybind11 submodule**: Fixed by manually cloning:
   ```bash
   git clone https://github.com/pybind/pybind11.git third_party/pybind11
   ```

2. **Missing accelerate package**: Fixed with:
   ```bash
   pip install accelerate
   ```

**Result**: Successfully compiled `pyfastllm.cpython-310-aarch64-linux-gnu.so` (5.1 MB)

### 2.2 Testing NPU Utilization

**Test Code**:
```python
from ftllm import llm

model = llm.model("/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm")
for response in model.stream_response("你好，请介绍一下你自己"):
    print(response, end='', flush=True)
```

**Result**: Model inference worked correctly, but NPU monitoring showed:
```
+===============+=================+======================================================+
| NPU     Name   | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 0       310B4  | Alarm           | 0.0          60                15    / 15            |
| 0       0       | NA              | 0            13591/ 15610                            |
+===============================+=================+======================================================+
```

**Observation**: AICore usage at 0% - NPU was allocated but not utilized for computation.

---

## 3. Model Preparation

### 3.1 Model Selection

**Selected Model**: Qwen2.5-3B-Instruct
- **Reason**: Appropriate size for 15GB RAM system
- **Parameters**: 3B parameters
- **Format**: FP16 (5.8GB)

**Download Source**: ModelScope (Chinese mirror for faster download)
```bash
pip install modelscope
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-3B-Instruct',
                  cache_dir='/home/HwHiAiUser/ai-works/models')
```

### 3.2 Model Conversion to INT8

**Conversion Code**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftllm import torch2flm

model_path = '/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct'
output_path = '/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='cpu',
    trust_remote_code=True
).eval()

torch2flm.tofile(output_path, model, tokenizer, dtype='int8')
```

**Result**: Successfully created 4.1GB INT8 quantized model (30% size reduction)

---

## 4. Root Cause Analysis: fastllm NPU Support

### 4.1 Code Investigation

**File**: `/home/HwHiAiUser/ai-works/ai-stuff/fastllm/src/executor.cpp`

**Device Registration Code** (lines 39-64):
```cpp
Executor::Executor() {
#ifdef USE_CUDA
    this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
#ifdef USE_TOPS
    this->devices.push_back((BaseDevice*) new TopsDevice());
#endif
#ifdef USE_TFACC
    this->devices.push_back((BaseDevice*) new TfaccDevice());
#endif
    // NO #ifdef USE_ASCEND block exists!
    this->devices.push_back((BaseDevice*) new CpuDevice());
}
```

**Critical Finding**: The `AscendDevice` class is defined in `src/devices/ascend/ascenddevice.cpp` but is **never instantiated** in the executor. No `#ifdef USE_ASCEND` block exists to register the Ascend device.

**Conclusion**: fastllm's Ascend NPU support is **incomplete stub code** - the device class exists but is never registered in the device initialization.

---

## 5. Alternative Solutions Research

### 5.1 vLLM-Ascend

**Description**: Open-source LLM inference framework with Ascend NPU support

**GitHub**: https://github.com/vllm-project/vllm-ascend

**Key Features**:
- Built on top of vLLM
- Supports Ascend NPU (310B, 910B, etc.)
- Active development by Huawei and community

**Requirements**:
- CANN 8.3.RC1 (recommended)
- torch-npu 2.7.1
- Python >= 3.10, < 3.12

### 5.2 MindIE

**Description**: Huawei's proprietary inference engine for Ascend NPUs

**Status**: Limited public documentation; typically requires enterprise license

### 5.3 MindSpore

**Description**: Huawei's open-source deep learning framework with native Ascend support

**Status**: Officially supported on Orange Pi AI Pro; well-documented

**Documentation**: [MindSpore Orange Pi Setup](https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html)

---

## 6. vLLM-Ascend Installation

### 6.1 Installation Process

#### Step 1: torch-npu Installation

**Issue**: Initial attempt to install torch-npu 2.8.0 failed due to CANN version mismatch with CANN 7.0.0 (latest link).

**Resolution**: Switched to CANN 8.0.0 environment:
```bash
source /usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/script/set_env.sh
pip install torch-npu
```

**Result**: torch-npu 2.7.1 successfully installed (also upgraded torch to 2.7.1)

**Verification**:
```python
import torch
import torch_npu
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
# Output: torch: 2.7.1+cpu, torch_npu: 2.7.1, NPU available: True
```

#### Step 2: vLLM Installation

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.11.0
```

**Issue**: Initial download from default PyPI was extremely slow (21 kB/s)

**Resolution**: Used Tsinghua mirror for faster downloads (2+ MB/s)

**Result**: vLLM 0.11.0 successfully installed

#### Step 3: vLLM-Ascend Installation

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm-ascend==0.11.0rc1
```

**Result**: vLLM-Ascend 0.11.0rc1 successfully installed

**Version Conflicts**:
- vLLM 0.11.0 requires torch==2.8.0
- vLLM-Ascend 0.11.0rc1 requires torch==2.7.1
- torch-npu 2.7.1 was installed, downgrading torch to 2.7.1

---

## 7. NNAL/ATB Library Installation

### 7.1 The Problem

First vLLM-Ascend test failed with:
```
OSError: libatb.so: cannot open shared object file: No such file or directory
Please check that the nnal package is installed.
```

### 7.2 NNAL Package Download

**Package**: Ascend-cann-nnal_8.0.0_linux-aarch64.run (399MB)

**Download Command**:
```bash
wget --header="Referer: https://www.hiascend.com/" \
  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-nnal_8.0.0_linux-aarch64.run
```

**Result**: Successfully downloaded to /tmp/

### 7.3 Installation Attempts

#### Attempt 1: Standard Installation (Failed)
```bash
/tmp/Ascend-cann-nnal_8.0.0_linux-aarch64.run --quiet --install --install-path=/usr/local/Ascend/nnal
```
**Error**: `create /usr/local/Ascend/nnal/nnal fail !` (permission denied)

#### Attempt 2: Home Directory Installation (Failed)
```bash
/tmp/Ascend-cann-nnal_8.0.0_linux-aarch64.run --quiet --install --install-path=/home/HwHiAiUser/Ascend/nnal
```
**Error**:
```
Check owner failed, can not find cann in /usr/local/Ascend/ascend-toolkit/laster
```
**Note**: The installer has a bug - it looks for "laster" instead of "latest"

### 7.4 Manual Extraction

```bash
/tmp/Ascend-cann-nnal_8.0.0_linux-aarch64.run --noexec --extract=/tmp/nnal_extract
/tmp/nnal_extract/run_package/Ascend-cann-atb_8.0.0_linux-aarch64.run --noexec --extract=/tmp/atb_extract
```

**Result**: Successfully extracted ATB libraries to `/tmp/atb_extract/atb/`

### 7.5 Manual Installation

```bash
mkdir -p /home/HwHiAiUser/Ascend/atb
cp -r /tmp/atb_extract/atb /home/HwHiAiUser/Ascend/
```

**ATB Libraries Located**:
- `/home/HwHiAiUser/Ascend/atb/cxx_abi_0/lib/libatb.so`
- `/home/HwHiAiUser/Ascend/atb/cxx_abi_1/lib/libatb.so`

**Environment Setup**:
```bash
source /home/HwHiAiUser/Ascend/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/script/set_env.sh
export VLLM_TARGET_DEVICE="ascend"
```

---

## 8. vLLM-Ascend Testing Results

### 8.1 Test Configuration

```python
import os
os.environ["VLLM_TARGET_DEVICE"] = "ascend"

from vllm import LLM, SamplingParams

model_path = "/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct"

llm = LLM(
    model=model_path,
    max_model_len=2048,
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="half",
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=50)
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)
```

### 8.2 Initialization Log

**Successful Steps**:
1. Platform plugin ascend activated ✓
2. Qwen2ForCausalLM architecture resolved ✓
3. PIECEWISE compilation enabled on NPU ✓
4. ACL graph batch sizes calculated: 48 ✓
5. Model weights loaded: 5.79 GB ✓

**Error Occurred**:
```
RuntimeError: EZ3003: No supported Ops kernel and engine are found for [RmsNorm24], optype [RmsNorm].

Possible Cause: The operator is not supported by the system.
Solution: 1. Check that the OPP component is installed properly.
           2. Submit an issue to request for the support of this operator type.
```

### 8.3 Root Cause Analysis

**Missing Operator**: `RmsNorm24`
- This is a specific variant of the RmsNorm (Root Mean Square Normalization) operator
- The number "24" likely refers to the hidden_size dimension or similar parameter
- **Current CANN 8.0.0**: Does not include this operator variant
- **Required by vLLM-Ascend**: CANN 8.3.RC1 (which has newer operators)

**Current System OPP Version**:
```
Version=7.6.0.1.220
version_dir=8.0.0
ops_version=7.6.0.1.220
```

---

## 9. Possible Fixes

### 9.1 Upgrade to CANN 8.3.RC1 (Recommended)

**Pros**:
- Most complete solution
- Includes all required operators
- Officially tested with vLLM-Ascend

**Cons**:
- Requires sudo/root access
- May need firmware/driver updates
- Potential compatibility risks

**Resources**:
- [CANN 8.3.RC1 Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0089.html)
- [CANN Community Download Center](https://www.hiascend.com/developer/download/community/result)

### 9.2 Install Additional OPP Kernel Packages

**Approach**: Search for and install additional operator packages for Ascend 310B

**Potential Packages**:
- `Ascend-cann-kernels-310b` - Additional kernel operators for 310B
- Custom operator packages from Huawei

### 9.3 Use enforce_eager Mode

**Workaround** to bypass ACL graph compilation:

```python
llm = LLM(
    model=model_path,
    enforce_eager=True,    # Disable ACL graph compilation
    max_model_len=1024,     # Reduce context length
)
```

**Pros**: No system changes required
**Cons**: Performance degradation; may not fully bypass the issue

### 9.4 Use Older vLLM-Ascend Version

**Approach**: Try vLLM-Ascend 0.10.x or 0.9.x which may be compatible with CANN 8.0.0

**Pros**: May work with current CANN version
**Cons**: May have missing features and bug fixes

### 9.5 Alternative Frameworks

If vLLM-Ascend cannot work with CANN 8.0.0:

| Framework | Status | Notes |
|-----------|--------|-------|
| **MindSpore** | Officially supported | Well-documented for Orange Pi AI Pro |
| **fastllm** | CPU works | NPU support incomplete |
| **Transformers + torch-npu** | Available | Direct PyTorch NPU usage |

---

## 10. Conclusions and Recommendations

### 10.1 Key Findings Summary

| Framework | Status | NPU Support | Notes |
|-----------|--------|-------------|-------|
| fastllm | Installed | ⚠️ Incomplete | AscendDevice never instantiated |
| vLLM-Ascend | Installed | ❌ Incompatible | Missing RmsNorm24 operator |
| torch-npu | Installed | ✅ Working | NPU detected and available |
| MindSpore | Not tested | ✅ Official | Recommended alternative |

### 10.2 Recommendations

#### Immediate Options:

1. **Try enforce_eager workaround** (easiest, no system changes)
   ```python
   llm = LLM(model=model_path, enforce_eager=True, max_model_len=1024)
   ```

2. **Use MindSpore** for NPU-accelerated inference
   - Officially supported on Orange Pi AI Pro
   - Well-documented

3. **Use fastllm with CPU** (already working)
   - Acceptable for small models
   - No NPU acceleration but functional

#### Long-term Solutions:

1. **Upgrade to CANN 8.3.RC1** (most complete fix)
   - Requires sudo access
   - May need system updates
   - Full vLLM-Ascend compatibility

2. **Wait for CANN updates** from Orange Pi / Huawei
   - New releases may include missing operators
   - May be bundled with system updates

### 10.3 Environment Setup Reference

**For future vLLM-Ascend usage** (after CANN upgrade):

```bash
# Source CANN environment (use 8.3.RC1 after upgrade)
source /usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/script/set_env.sh

# Source ATB environment
source /home/HwHiAiUser/Ascend/atb/set_env.sh

# Set target device
export VLLM_TARGET_DEVICE="ascend"

# Optional: Set HCCL expansion for better performance
export HCCL_OP_EXPANSION_MODE=AIV
```

### 10.4 Files and Locations

| Item | Location |
|------|----------|
| fastllm build | `/home/HwHiAiUser/ai-works/ai-stuff/fastllm/build/` |
| Qwen2.5-3B model | `/home/HwHiAiUser/ai-works/models/Qwen/Qwen2.5-3B-Instruct/` |
| INT8 fastllm model | `/home/HwHiAiUser/ai-works/models/qwen2.5-3b-int8.flm` |
| ATB libraries | `/home/HwHiAiUser/Ascend/atb/` |
| Previous report | `/home/HwHiAiUser/ai-works/orangePi-AiPro/docs/NPU_Exploration_Report.md` |

---

## Appendix A: Command Reference

### Environment Setup
```bash
# CANN 8.0.0 environment
source /usr/local/Ascend/ascend-toolkit/8.0.0/aarch64-linux/script/set_env.sh

# ATB environment
source /home/HwHiAiUser/Ascend/atb/set_env.sh

# Python environment
conda activate qwen_onnx
```

### NPU Monitoring
```bash
npu-smi info
```

### Testing torch-npu
```python
import torch
import torch_npu
print(f'torch: {torch.__version__}')
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
```

---

## Appendix B: Sources and References

1. **vLLM-Ascend Documentation**
   - Installation Guide: https://docs.vllm.ai/projects/ascend/en/latest/installation.html
   - GitHub Repository: https://github.com/vllm-project/vllm-ascend

2. **CANN Documentation**
   - CANN 8.3.RC1 Installation: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0089.html
   - CANN Community Downloads: https://www.hiascend.com/developer/download/community/result
   - Orange Pi AI Pro算子开发环境搭建指导: https://public-download.obs.cn-east-2.myhuaweicloud.com/...

3. **Orange Pi AI Pro Resources**
   - MindSpore Setup: https://www.mindspore.cn/tutorials/zh-CN/r2.7.1/orange_pi/environment_setup.html
   - Orange Pi Downloads: http://www.orangepi.cn/html/serviceAndSupport/index.html

4. **fastllm**
   - GitHub Repository: https://github.com/hsjac/fastllm

---

**Report Version**: 1.0
**Last Updated**: December 31, 2025
**Author**: Claude Code (Assistant)
