# CPU vs NPU Performance Benchmark Suite

## Overview

This directory contains comprehensive performance testing tools and results for comparing CPU and NPU (Neural Processing Unit) inference performance on the Orange Pi AI Pro with Ascend 310B NPU.

## Directory Structure

```
cpu-npu-benchmark/
├── README.md                              # This file
├── performance_test.py                    # Main benchmark script
├── create_comparison_chart.py             # Visualization generator
├── ai_performance_report.md               # Detailed analysis report (15 pages)
├── PERFORMANCE_TEST_SUMMARY.md            # Quick reference guide
├── cpu_vs_npu_comparison.png              # Performance visualization charts
└── performance_results_*.txt              # Raw benchmark data files
```

## Quick Start

### Prerequisites

```bash
# Ensure CANN environment is loaded (for NPU support)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Check PyTorch and NPU availability
python3 -c "import torch; import torch_npu; print(f'PyTorch: {torch.__version__}')"
```

### Running Tests

#### 1. Run CPU vs NPU Performance Benchmark

```bash
python3 performance_test.py
```

This will:
- Run 100 iterations of CNN inference on CPU
- Run 100 iterations of CNN inference on NPU (when available)
- Generate detailed performance metrics
- Save results to `performance_results_YYYYMMDD_HHMMSS.txt`

#### 2. Generate Visualization Charts

```bash
python3 create_comparison_chart.py
```

This creates:
- `cpu_vs_npu_comparison.png` with 4 performance comparison charts

## Test Configuration

### Model Architecture
- **Type:** Custom CNN (ResNet-like)
- **Input Size:** 224×224×3 (ImageNet standard)
- **Output:** 1000 classes
- **Batch Size:** 1

### Benchmark Parameters
- **Warm-up Iterations:** 10
- **Test Iterations:** 100
- **Metrics Tracked:**
  - Mean inference time
  - Standard deviation
  - Min/Max times
  - Throughput (inferences/sec)
  - Memory usage

## Results Summary

### CPU Performance (Measured)
- **Inference Time:** 1,221.93 ms per image
- **Throughput:** 0.82 images/second
- **Memory Usage:** 10 MB
- **Stability:** Excellent (σ = 30ms)
- **Real-time Capable:** No

### NPU Performance (Theoretical)
- **Inference Time:** ~80 ms per image (projected)
- **Throughput:** ~12.5 images/second (projected)
- **Speedup:** 15.3x faster than CPU
- **Real-time Capable:** Yes (>10 FPS)

## Troubleshooting

### NPU Not Detected

```bash
# Check NPU device files
ls -la /dev/hisi_*

# Verify CANN environment
echo $ASCEND_HOME_PATH

# Test NPU import
python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
```

### Operator Compilation Errors

If you encounter:
```
build op model failed, result = 500001
```

This indicates the NPU operator models haven't been compiled. Solutions:
1. Ensure CANN environment is sourced: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
2. Contact system administrator to compile operator models
3. Check that Ascend drivers match CANN toolkit version

### Import Errors

```bash
# Set library path
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# Ensure Python path includes torch_npu
export PYTHONPATH=/usr/local/miniconda3/lib/python3.9/site-packages:$PYTHONPATH
```

## Documentation

### Full Reports
- **ai_performance_report.md** - Comprehensive 15-page analysis
- **PERFORMANCE_TEST_SUMMARY.md** - Quick reference guide
- **Performance charts** - Visual comparison of CPU vs NPU

### Raw Data
- **performance_results_*.txt** - Detailed metrics from each test run

## System Requirements

### Hardware
- Orange Pi AI Pro with Ascend 310B NPU
- 4GB+ RAM (15GB available on test platform)
- ARM64 (aarch64) architecture

### Software
- Ubuntu 22.04.3 LTS
- Python 3.9+
- PyTorch 2.1.0
- Torch-NPU 2.1.0+
- CANN Toolkit 7.0.0+

## Performance Optimization

### For CPU (Current)
1. Use smaller models
2. Apply quantization (INT8)
3. Reduce input image size
4. Use batch processing
5. Enable compiler optimizations

### For NPU (When Enabled)
1. Use Ascend-optimized models
2. Apply INT8 quantization
3. Leverage hardware acceleration
4. Optimize tensor shapes
5. Use recommended operator set

## Expected Use Cases

### CPU-Only (Current Performance)
- ✓ Model development and prototyping
- ✓ Batch image processing (2,952 images/hour)
- ✓ Non-real-time applications
- ✓ Training on small datasets

### NPU-Enabled (Projected Performance)
- ✓ Real-time object detection (12+ FPS)
- ✓ Live video analytics
- ✓ Interactive AI applications
- ✓ Video surveillance systems
- ✓ Autonomous robots
- ✓ Smart cameras
- ✓ IoT edge devices

## Contributing

To extend this benchmark:

1. **Add New Models**
   - Edit `performance_test.py`
   - Define new model architecture in `SimpleCNN` class or add new class
   - Update test parameters

2. **Add New Metrics**
   - Modify `benchmark_inference()` function
   - Add tracking for additional measurements
   - Update result analysis

3. **Custom Visualizations**
   - Edit `create_comparison_chart.py`
   - Add new plots using matplotlib
   - Export different formats (PDF, SVG, etc.)

## References

- [Ascend 310B Documentation](https://www.huaweicloud.com/product/ascend.html)
- [PyTorch-NPU](https://github.com/Ascend/pytorch)
- [CANN Toolkit](https://www.huaweicloud.com/product/cann.html)
- [Orange Pi AI Pro](http://www.orangepi.org/)

## License

This benchmark suite is provided as-is for educational and research purposes.

## Support

For issues:
- **NPU Configuration:** Check CANN documentation
- **Operator Compilation:** Requires administrator access
- **Performance Issues:** Review ai_performance_report.md

---

**Last Updated:** 2025-12-03
**Platform:** Orange Pi AI Pro (Ascend 310B)
**PyTorch Version:** 2.1.0
