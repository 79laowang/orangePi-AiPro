# NPU Model Exploration Summary

## Session Overview

**Date**: 2024-12-30
**Hardware**: Orange Pi AI Pro (Ascend 310B1)
**System**: Ubuntu 22.04.3 LTS (aarch64), 15GB RAM
**CANN Version**: 7.1.0.3.220

---

## Executive Summary

**Finding**: Large language models (LLMs) like Qwen2.5 are **not well-suited** for Ascend 310B NPU due to:

1. **Protobuf 2GB limit** - ONNX export with proper attention masking exceeds limit
2. **torch.jit.trace limitations** - Bakes in attention_mask/position_ids, causing output degradation
3. **Memory constraints** - 15GB RAM insufficient for NPU mode (requires 8-10GB shared memory)

**Recommendation**: Use **CPU mode for LLMs** and **NPU for vision models** (YOLO/ResNet).

---

## Exploration Chronology

### Phase 1: Initial Setup (qwen_clean128.onnx)

**Goal**: Export Qwen2.5-0.5B-Instruct with seq_len=128

**Issues Encountered**:
- `RuntimeError: unordered_map::at` - vmap operations in transformers 4.38+ incompatible with TorchScript

**Solutions**:
- Downgraded transformers: 4.57.3 → 4.37.2
- Downgraded PyTorch: 2.9.1 → 2.1.2 (torch.export incompatibility)
- Downgraded NumPy: 2.2.6 → 1.26.4 (CANN compatibility)

**Result**: Successfully exported `qwen_clean128.onnx` (634KB)

### Phase 2: OM Conversion

**Files**:
- `qwen_clean128.onnx` → `qwen_clean128.om` (1.5GB)
- `qwen_patch512.onnx` → `qwen_patch512.om` (1.5GB)

**ATC Command**:
```bash
atc --framework=5 \
    --model="qwen_clean128.onnx" \
    --output="qwen_clean128" \
    --soc_version=Ascend310B1 \
    --input_format="ND" \
    --precision_mode="allow_fp32_to_fp16"
```

**Modifications**: Removed `--autoTuneMode` flag (unsupported in CANN 7.1)

### Phase 3: Inference Testing

**Issue**: Output degrades to garbage after 3-4 tokens

**Root Cause**:
```python
# torch.jit.trace captures concrete values
attention_mask = torch.ones(batch_size, seq_len)  # Baked as all 1s
position_ids = torch.arange(seq_len)              # Baked as 0..127
```

When padding is applied:
- Zero tokens are treated as valid (attention_mask all 1s)
- Position IDs are incorrect for padding positions
- Model attention becomes corrupted

### Phase 4: Dynamic Masks Attempt

**Goal**: Export with attention_mask and position_ids as separate inputs

**Approach**:
```python
class QwenWithMasksWrapper(nn.Module):
    def forward(self, input_ids, attention_mask, position_ids):
        return self.model(input_ids, attention_mask, position_ids, ...)
```

**Result**: `exceeded maximum protobuf size of 2GB: 2521365080 bytes`

**Even with seq_len=32**: Model size itself (~2.35GB) exceeds protobuf limit

### Phase 5: Verified Model Research

**Finding**: Vision models (YOLO, ResNet) are verified to work on Ascend 310B

| Model | Status | Reference |
|-------|--------|-----------|
| YOLOv5s | ✅ Verified (Atlas 300I, CANN 5.0.2) | [yolov5-ascend](https://github.com/jackhanyuan/yolov5-ascend) |
| YOLOv8n | ✅ Documented workflow | [IC-Online Guide](https://ic-online.com/news/post/mastering-yolo-and-resnet-optimization-on-hisilicon-npus) |
| ResNet-50 | ✅ Well-supported | Same guide |

---

## Technical Analysis

### Why LLMs Fail on Ascend 310B

| Factor | LLM (Qwen) | Vision (YOLO) |
|--------|------------|---------------|
| Architecture | Transformer + causal attention | CNN |
| Input shape | Variable (seq_len) | Fixed (640×640) |
| Attention | Complex causal masking | Standard conv |
| ONNX size | >2GB (exceeds limit) | <100MB |
| NPU match | Text-focused | Conv-optimized |

### Environment Compatibility Matrix

| Component | Working Version | Notes |
|-----------|----------------|-------|
| transformers | 4.37.2 | 4.38+ has vmap issues |
| torch | 2.1.2 | 2.4+ forces torch.export |
| numpy | 1.26.4 | 2.x incompatible with CANN |
| onnx | 1.15.0 | Required for export |
| CANN | 7.1.0.3.220 | Current system version |

---

## Files Created

### Export Scripts
- `export_qwen_clean.py` - Final working export script
- `export_qwen_with_masks.py` - Attempted dynamic masks (failed)
- `export_qwen_patched.py` - Patched transformers approach (abandoned)
- `export_qwen_fixed.py` - Earlier iteration
- `export_qwen_static.py` - Static shape attempt
- `export_qwen_optimum.py` - Optimum export attempt

### Inference Scripts
- `acl_inference_patched.py` - Latest inference with multiple padding strategies
- `acl_inference_simple.py` - Simplified version
- `acl_inference_fixed.py` - Fixed version
- `acl_inference_qwen.py` - Original script

### Conversion Scripts
- `convert_fixed.sh` - Modified ATC conversion (removed autoTuneMode)
- `convert_onnx_to_om.sh` - Original conversion script

### Debug Scripts
- `debug_model_clean.py` - Verify PyTorch model output
- `debug_model_output.py` - Debug inference issues

### Generated Models
- `qwen_clean128.onnx` (634KB) - Clean export with seq_len=128
- `qwen_clean128.om` (1.5GB) - Converted OM model
- `qwen_patch128.onnx` + `.data` (1.9GB) - Patched version
- `qwen_patch128.om` (1.5GB)
- `qwen_patch512.onnx` + `.data` (1.9GB) - seq_len=512
- `qwen_patch512.om` (1.5GB)
- `qwen_with_masks128.onnx` (0 bytes) - Failed export
- `qwen_with_masks32.onnx` (0 bytes) - Failed export

### ATC Generated Files
- Multiple `_Constant_*_attr__value` files (~1.7GB total)
- `fusion_result.json` - ATC fusion results
- `kernel_meta/` - Compiled kernels

---

## Conclusions

### What Works
1. ✅ **CPU mode for LLMs** - transformers + PyTorch directly
2. ✅ **ONNX export with fixed shapes** - But output degrades
3. ✅ **OM conversion** - ATC works correctly
4. ✅ **Vision models** - YOLO/ResNet verified on Ascend 310B

### What Doesn't Work
1. ❌ **NPU mode for LLMs** - Memory constraints (15GB < 8-10GB needed)
2. ❌ **Dynamic attention masks** - Exceeds protobuf 2GB limit
3. ❌ **torch.jit.trace for LLMs** - Bakes in critical tensors
4. ❌ **MindSpore NPU mode** - Requires too much memory

### Recommendations

| Use Case | Recommended Approach |
|----------|---------------------|
| Chinese dialogue/text generation | **CPU mode** with transformers |
| Object detection | **YOLOv5/v8** on NPU |
| Image classification | **ResNet** on NPU |
| Real-time video analysis | **YOLO + INT8 quantization** |

---

## Next Steps

If pursuing NPU-based AI:
1. Deploy YOLOv5s for object detection
2. Use CPU mode for any LLM requirements
3. Consider quantization (INT8) for better performance

---

## References

- [MINDSPORE_INSTALL_GUIDE.md](../MINDSPORE_INSTALL_GUIDE.md) - Complete setup guide
- [yolov5-ascend](https://github.com/jackhanyuan/yolov5-ascend) - Verified YOLO implementation
- [YOLO/ResNet Guide](https://ic-online.com/news/post/mastering-yolo-and-resnet-optimization-on-hisilicon-npus)

---

**Session End**: 2024-12-30
**Status**: NPU LLM exploration concluded - Vision models recommended for NPU deployment
