# Qwen NPU Exploration Archive

This directory contains preserved files from the NPU model exploration session.

## Session Summary

See `NPU_EXPLORATION_SUMMARY.md` in the project root for complete findings.

## Preserved Files

### qwen_essentials/
Working scripts and reference model:

- `export_qwen_clean.py` - Final working ONNX export script
- `acl_inference_patched.py` - ACL inference reference implementation
- `convert_fixed.sh` - ATC conversion script (CANN 7.1 compatible)
- `qwen_clean128.onnx` - Exported ONNX model (634KB)
- `qwen_clean128.om` - Converted OM model (1.5GB)
- `README.md` - Original README
- `QWEN_TO_ASCEND_GUIDE.md` - Original guide

## Key Findings

1. **LLMs on Ascend 310B NPU** - Not recommended due to:
   - Protobuf 2GB limit when exporting with dynamic attention masks
   - torch.jit.trace bakes in critical tensors (attention_mask, position_ids)
   - Memory constraints (15GB < 8-10GB shared memory needed)

2. **Alternative Approaches**:
   - Use **CPU mode** for LLMs (10-20 tokens/s, stable)
   - Use **NPU mode** for vision models (YOLO, ResNet)

3. **Verified NPU Models**:
   - YOLOv5s - Object detection (verified)
   - YOLOv8n - Object detection (documented)
   - ResNet-50 - Image classification (well-supported)

## Usage

To run Qwen inference on **CPU** (recommended):
```bash
pip install transformers torch sentencepiece
# Use transformers library directly with CPU
```

To deploy **YOLO** on NPU:
```bash
git clone https://github.com/jackhanyuan/yolov5-ascend.git
cd yolov5-ascend
python detect_yolov5_ascend.py
```

