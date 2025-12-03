# Performance Test Summary

Quick reference guide for CPU vs NPU performance benchmarks on Orange Pi AI Pro.

## Results at a Glance

### CPU Performance (Measured)
```
Inference Time:   1,222 ms per image
Throughput:       0.82 images/second
Memory Usage:     10 MB
Stability:        Excellent (σ = 30ms)
Real-time:        ❌ No
```

### NPU Performance (Projected - Ascend 310B)
```
Inference Time:   ~80 ms per image
Throughput:       ~12.5 images/second
Speedup:          15.3x faster than CPU
Real-time:        ✅ Yes (>10 FPS)
```

## Quick Commands

```bash
# Run CPU benchmark
python3 performance_test.py

# Load CANN environment and run full test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 performance_test.py

# Generate visualization charts
python3 create_comparison_chart.py
```

## Key Findings

### ✅ Working
- PyTorch with NPU support installed
- Ascend driver loaded (`/dev/hisi_bbox0`)
- CANN toolkit installed (v7.0.0)
- CPU benchmarking functional
- Stable performance measurements

### ⚠️ Issues
- NPU operator models not compiled
- Error: "build op model failed, result = 500001"
- Requires administrator access

## Common Fixes

### NPU Not Detected
```bash
ls -la /dev/hisi_*
echo $ASCEND_HOME_PATH
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
```

### Import Errors
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

## Performance Impact

### CPU-Only (Current)
- **Suitable for:**
  - Batch processing (2,952 images/hour)
  - Model development
  - Non-real-time applications

### With NPU Enabled
- **Expected:**
  - 15x faster inference
  - Real-time object detection (12+ FPS)
  - Live video analytics

## Documentation

- **README.md** - Comprehensive usage guide
- **ai_performance_report.md** - Detailed technical analysis
- **cpu_vs_npu_comparison.png** - Performance visualization

---

**Last Updated:** 2025-12-03
**Platform:** Orange Pi AI Pro (Ascend 310B)
