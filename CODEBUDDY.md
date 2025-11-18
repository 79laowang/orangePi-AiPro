# CODEBUDDY.md This file provides guidance to CodeBuddy Code when working with code in this repository.

## Project Overview

This is a comprehensive documentation and configuration repository for the Orange Pi AI Pro single-board computer, designed for AI/ML development on embedded hardware. The project centers around the Orange Pi AI Pro which features Huawei HiSilicon Ascend 310/310B AI processors running Ubuntu 22.04.3 LTS (aarch64).

## Key Components

- **Hardware**: Orange Pi AI Pro with Ascend 310/310B NPU
- **OS**: Ubuntu 22.04.3 LTS (Linux 5.10.0+) aarch64 architecture  
- **Primary Languages**: Python, Node.js (via qwen-code)
- **AI Frameworks**: TensorFlow Lite, OpenCV, PyTorch (conditional)

## Architecture

The project follows a four-layer architecture pattern:

1. **System Configuration Layer**: Hardware setup, OS optimization, network configuration
2. **Development Environment Layer**: Python/Node.js setup, AI framework installation
3. **Application Layer**: Computer vision, NLP, edge AI implementations
4. **Performance Optimization Layer**: Model quantization, hardware acceleration, real-time inference

## Essential Commands

### System Setup and Maintenance

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential git python3-pip python3-venv

# Expand filesystem to use full SD card
sudo resize2fs /dev/mmcblk0p2

# Check system information
ip addr show
htop
watch -n 1 cat /sys/class/thermal/thermal_zone*/temp
```

### AI Framework Installation

```bash
# TensorFlow Lite
pip3 install tensorflow-lite-runtime tensorflow

# OpenCV for Computer Vision
pip3 install opencv-python opencv-contrib-python

# PyTorch (if hardware supported)
pip3 install torch torchvision

# Core ML libraries
pip3 install numpy
```

### Qwen-Code Installation (Node.js Environment)

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
# or use domestic mirror: curl -o- https://gitee.com/mirrors/nvm/raw/master/install.sh | bash

source ~/.bashrc
nvm install 20 && nvm use 20

# Configure npm for user directory and Chinese mirror
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
npm config set registry https://registry.npmmirror.com

# Install qwen-code
npm install -g @qwen-code/qwen-code
```

### Model Conversion and Deployment

```bash
# Convert TensorFlow model to TensorFlow Lite
python3 -m tensorflow.lite.python.convert \
  --input_file=model.pb \
  --output_file=model.tflite

# Test camera functionality
fswebcam test.jpg
lsusb
```

### Image Flashing

```bash
# List removable devices
sudo ./scripts/img-etcher-Pi.sh --list

# Write image with verification
sudo ./scripts/img-etcher-Pi.sh <image-file.img.xz> --verify

# Write with specific mode
sudo ./scripts/img-etcher-Pi.sh <image-file.img.gz>
# Select write mode when prompted (1=Balanced, 2=Fast, 3=Robust, 4=Legacy)
```

## File Structure

```
/home/HwHiAiUser/ai-works/orangePi-AiPro/
├── README.md                    # Comprehensive development guide
├── LICENSE                      # Apache 2.0 license
├── CLAUDE.md                    # Documentation for AI assistants
├── QWEN.md                      # Context for Qwen Code AI assistant
├── scripts/                     # System configuration scripts
│   ├── 01-orangepi-startup-text # MOTD (Message of the Day) configuration
│   ├── img-etcher-Pi.sh        # Enhanced image flashing tool
│   └── README.md               # Documentation for MOTD scripts
└── .git/                       # Git repository metadata
```

## Development Workflow

1. **Hardware Setup**: Configure Orange Pi AI Pro with proper power supply and cooling
2. **System Configuration**: Install Ubuntu, update packages, enable hardware acceleration
3. **Environment Setup**: Install Python development tools and AI frameworks
4. **Application Development**: Use TensorFlow Lite for edge AI inference
5. **Optimization**: Implement model quantization and hardware acceleration

## Hardware-Specific Considerations

### Orange Pi AI Pro Specifications
- **Processor**: Huawei HiSilicon Ascend 310/310B AI processor
- **Architecture**: aarch64 (ARM64)
- **Memory**: Configurable RAM (typically 15GB as shown in MOTD)
- **Storage**: MicroSD card (32GB+ recommended)
- **Power**: 5V/3A power supply required

### Performance Limitations
- CPU temperature monitoring not fully supported by Ubuntu
- Requires adequate cooling for sustained AI workloads
- Model size limitations due to embedded constraints

## Common Development Tasks

### Computer Vision Applications
Typical pattern for camera-based AI applications:
```python
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model and setup camera
interpreter = tf.lite.Interpreter(model_path="model.tflite")
cap = cv2.VideoCapture(0)

# Process frames in real-time
while True:
    ret, frame = cap.read()
    # Preprocess, infer, postprocess
```

### Model Optimization Workflow
1. Train models on full-scale systems
2. Convert to TensorFlow Lite format
3. Apply quantization for size reduction
4. Deploy and optimize for edge inference

## Testing and Troubleshooting

### Hardware Validation
```bash
# Check camera connectivity
lsusb
fswebcam test.jpg

# Monitor system resources
htop
iostat -x 1
sudo journalctl -xe
```

### Network Configuration
```bash
# Static IP configuration
sudo nano /etc/netplan/01-netcfg.yaml

# Restart networking
sudo systemctl restart networking
```

### Common Issues
- **Boot Problems**: Check SD card integrity and power supply
- **Performance**: Monitor temperature, optimize model size
- **Camera Issues**: Verify USB connection and test with fswebcam

## Target Applications

The codebase enables development of:
- **Real-time Object Detection**: Using TensorFlow Lite with camera input
- **Face Recognition Systems**: Access control and security applications
- **Smart Home Automation**: Voice control and image-based security
- **Industrial IoT**: Quality inspection and predictive maintenance
- **Robotics**: Autonomous navigation and human-robot interaction

## Development Environment Configuration

### Python Environment
```bash
# Create virtual environment for AI projects
python3 -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt
```

### Remote Development
```bash
# Enable SSH access
sudo systemctl enable ssh && sudo systemctl start ssh

# VNC for desktop access
sudo apt install tightvncserver
vncserver :1
```

## Performance Optimization Strategies

1. **Model Quantization**: Reduce model size while maintaining accuracy
2. **Hardware Acceleration**: Utilize NPU capabilities for AI inference
3. **Multi-threading**: Optimize for ARM64 architecture
4. **Memory Management**: Monitor and optimize memory usage for embedded constraints