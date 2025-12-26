# Orange Pi AI Pro

A comprehensive guide for system configuration, user experience, learning, and AI development on the Orange Pi AI Pro single-board computer.

## Table of Contents

- [Overview](#overview)
- [System Configuration](#system-configuration)
- [User Experience](#user-experience)
- [Learning Resources](#learning-resources)
- [AI Development](#ai-development)
- [Applications](#applications)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Orange Pi AI Pro is a powerful single-board computer designed for AI applications, featuring:
- High-performance AI processing capabilities
- Rich I/O interfaces
- Support for various AI frameworks
- Linux-based operating system

This repository contains documentation, configuration guides, and examples to help you get started with AI development on the Orange Pi AI Pro.

## System Configuration

### Initial Setup

1. **Hardware Requirements**
   - Orange Pi AI Pro board with Huawei HiSilicon Ascend 310B AI processor
   - MicroSD card (32GB+ recommended)
   - Power supply (5V/3A recommended)
   - USB-C cable
   - Optional: Cooling fan, case

2. **Operating System Installation**
   ```bash
   # Download the latest OS image
   # Flash to microSD card using tools like:
   # - https://github.com/79laowang/orangePi-AiPro/blob/main/scripts/img-etcher-Pi.sh
   # - dd command
   # - opiaipro_ubuntu22.04_desktop_aarch64_20250925.img.xz
   ```

3. **System Configuration**
   - Update system packages
   - Configure development environment

### Development Environment Setup

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential git python3-pip python3-venv

# Install AI/ML libraries
pip3 install numpy opencv-python tensorflow-lite
```

### CANN (Compute Architecture for Neural Networks) Setup

The Orange Pi AI Pro features a Huawei HiSilicon Ascend 310B AI processor. Understanding the key components:

- **NPU (Neural Processing Unit)**: 华为的AI加速器，相当于NVIDIA的GPU
- **CANN (Compute Architecture for Neural Networks)**: 相当于NVIDIA的CUDA，提供驱动层和算子库
- **ATC (Ascend Tensor Compiler)**: 核心工具，将开源模型转换为昇腾能跑的.om格式
- **ACL (Ascend Computing Language)**: 底层API，用于调用NPU进行推理

#### Check CANN Installation
```bash
# Check NPU status
npu-smi info

# Expected output should show:
# - NPU name: 310B4 (Ascend 310B)
# - CANN version: 25.2.0
# - Health status
# - Memory usage
```

#### Environment Configuration
CANN environment variables should be automatically configured through `/etc/ascend_install.info` and `set_env.sh`. Verify with:

```bash
# Check if environment variables are set
echo $ASCEND_TOOLKIT_HOME
echo $LD_LIBRARY_PATH

# ATC (Ascend Tensor Compiler) should be available
which atc
atc --help
```

#### Model Conversion with ATC
ATC (Ascend Tensor Compiler) 是将开源模型(ONNX/Caffe/TensorFlow)转换成昇腾.om格式的核心工具:

```bash
# Convert TensorFlow model to OM format for Ascend NPU
atc --model=model.pb --framework=3 --output=model_output --soc_version=Ascend310B3

# Convert ONNX model to OM format
atc --model=model.onnx --framework=5 --output=model_output --soc_version=Ascend310B3

# Convert with input shape and optimization
atc --model=model.onnx --framework=5 --input_shape="input:1,3,224,224" --output=model_optimized --soc_version=Ascend310B3 --input_format=NCHW
```

### Qwen-Code Installation

To install and configure qwen-code on your Orange Pi AI Pro:

```bash
# 1. Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# Or use domestic mirror:
curl -o- https://gitee.com/mirrors/nvm/raw/master/install.sh | bash

# 2. Reload bash configuration
source ~/.bashrc

# 3. Verify nvm installation
nvm --version

# 4. Upgrade Node.js to version 20+
nvm install 20
nvm use 20

# 5. Configure npm to use user directory
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 6. Configure npm to use domestic mirror
npm config set registry https://registry.npmmirror.com

# 7. Install qwen-code
npm install -g @qwen-code/qwen-code

# 8. Verify installation
qwen
```

## User Experience

### Performance Monitoring

```bash
# Monitor CPU and memory usage
htop

# Check NPU usage
npu-smi info

# Monitor system resources
iostat -x 1
```

### Online Resources

- [Orange Pi Official Documentation](https://www.orangepi.org/)
- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python for Embedded Systems](https://docs.python.org/3/)

## AI Development

### Setting Up AI Frameworks

#### TensorFlow Lite
```bash
# Install TensorFlow Lite
pip3 install tensorflow-lite-runtime

# For development
pip3 install tensorflow
```

#### MindSpore (华为AI框架)
```bash
# Install MindSpore for Ascend
pip3 install mindspore-ascend

# Verify installation
python -c "import mindspore; print(mindspore.__version__)"
```

#### PyTorch (if supported)
```bash
# Install PyTorch
pip3 install torch torchvision
```

#### OpenCV for Computer Vision
```bash
# Install OpenCV
pip3 install opencv-python opencv-contrib-python
```

### Model Conversion

```bash
# Convert TensorFlow model to TensorFlow Lite
python3 -m tensorflow.lite.python.convert \
  --input_file=model.pb \
  --output_file=model.tflite
```

### AI Development Examples

#### ACL (Ascend Computing Language) API Usage
ACL是调用NPU进行推理的底层API，提供C++和Python接口:

```python
import acl
import numpy as np

# ACL初始化
acl.init()
context, ret = acl.rt.create_context(device_id=0)

# 加载OM模型
model_id, ret = acl.mdl.load_from_file("model.om")

# 创建数据集
dataset = acl.mdl.create_dataset()
# 准备输入数据并传输到设备
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
# ... ACL数据处理流程

# 执行推理
ret = acl.mdl.execute(model_id, dataset, dataset)

# 清理资源
acl.mdl.unload(model_id)
acl.rt.destroy_context(context)
acl.finalize()
```

#### Computer Vision with Ascend NPU
```python
import cv2
import numpy as np

# 使用ACL API进行NPU推理的完整流程
# 结合OpenCV进行图像预处理和结果可视化
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # 预处理图像数据
        # 通过ACL调用NPU进行推理
        # 后处理并显示结果
        cv2.imshow('Ascend NPU Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Debugging Tips

```bash
# Check system logs
sudo journalctl -xe

# Monitor system resources
htop
iostat -x 1

# Check hardware
dmesg | tail
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Orange Pi community for hardware support
- Open source AI/ML framework developers
- Contributors and testers

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Last Updated**: 2025

**Maintained by**: [Ke Wang]

