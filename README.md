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
   - Orange Pi AI Pro board
   - MicroSD card (32GB+ recommended)
   - Power supply (5V/3A recommended)
   - USB-C cable
   - Optional: Cooling fan, case

2. **Operating System Installation**
   ```bash
   # Download the latest OS image
   # Flash to microSD card using tools like:
   # - balenaEtcher
   # - dd command
   # - Raspberry Pi Imager
   ```

3. **First Boot Configuration**
   - Connect via serial console or SSH
   - Set up user account and password
   - Configure network settings
   - Update system packages

### System Optimization

#### Enable GPU Acceleration
```bash
# Configure GPU settings in device tree
# Enable hardware acceleration for AI workloads
```

#### Memory and Storage
```bash
# Expand filesystem to use full SD card
sudo resize2fs /dev/mmcblk0p2

# Configure swap space if needed
sudo swapon --show
```

#### Network Configuration
```bash
# Static IP configuration (if needed)
sudo nano /etc/netplan/01-netcfg.yaml
```

### Development Environment Setup

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential git python3-pip python3-venv

# Install AI/ML libraries
pip3 install numpy opencv-python tensorflow-lite
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

### Desktop Environment

The Orange Pi AI Pro supports various desktop environments:
- **Xfce**: Lightweight and efficient
- **GNOME**: Full-featured desktop
- **KDE**: Modern and customizable

### Remote Access

#### SSH Access
```bash
# Enable SSH (if not already enabled)
sudo systemctl enable ssh
sudo systemctl start ssh

# Connect from another machine
ssh username@orange-pi-ip
```

#### VNC Remote Desktop
```bash
# Install VNC server
sudo apt install -y tightvncserver

# Start VNC server
vncserver :1
```

### Performance Monitoring

```bash
# Monitor CPU and memory usage
htop

# Check GPU usage
nvidia-smi  # If applicable

# Monitor temperature
watch -n 1 cat /sys/class/thermal/thermal_zone*/temp
```

## Learning Resources

### Getting Started Tutorials

1. **Basic Linux Commands**
   - File system navigation
   - Package management
   - Process management
   - Network configuration

2. **Python Programming**
   - Python basics on embedded systems
   - Working with GPIO pins
   - Interfacing with sensors

3. **AI/ML Fundamentals**
   - Introduction to machine learning
   - Neural networks basics
   - Computer vision concepts
   - Natural language processing

### Recommended Learning Path

1. **Week 1-2: System Basics**
   - OS installation and configuration
   - Basic Linux administration
   - Hardware interfacing

2. **Week 3-4: Python Development**
   - Python programming fundamentals
   - Libraries and package management
   - GPIO and hardware control

3. **Week 5-6: AI Introduction**
   - Machine learning concepts
   - TensorFlow Lite basics
   - Simple inference examples

4. **Week 7-8: Advanced Projects**
   - Computer vision projects
   - Real-time inference
   - Model optimization

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

### Example: Image Classification

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
image = Image.open("test_image.jpg")
image = image.resize((224, 224))
input_data = np.expand_dims(image, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get results
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = output_data[0]
```

### Example: Object Detection

```python
import cv2
import numpy as np

# Load model and labels
# Initialize camera or load image
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Preprocess frame
    # Run inference
    # Draw bounding boxes
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Performance Optimization

1. **Model Quantization**
   - Reduce model size
   - Improve inference speed
   - Maintain acceptable accuracy

2. **Hardware Acceleration**
   - Utilize NPU (Neural Processing Unit) if available
   - GPU acceleration
   - Multi-threading

3. **Model Pruning**
   - Remove unnecessary connections
   - Reduce computational requirements

## Applications

### Computer Vision Applications

1. **Face Recognition System**
   - Real-time face detection
   - Face recognition and identification
   - Access control system

2. **Object Tracking**
   - Real-time object tracking
   - Motion detection
   - Surveillance applications

3. **Image Classification**
   - Custom image classifiers
   - Product recognition
   - Quality control

### Natural Language Processing

1. **Voice Assistant**
   - Speech recognition
   - Command processing
   - Text-to-speech

2. **Language Translation**
   - Real-time translation
   - Multi-language support

### Edge AI Applications

1. **Smart Home Automation**
   - Voice control
   - Image-based security
   - Predictive maintenance

2. **Industrial IoT**
   - Quality inspection
   - Predictive analytics
   - Anomaly detection

3. **Robotics**
   - Autonomous navigation
   - Object manipulation
   - Human-robot interaction

### Project Examples

#### Example 1: Real-time Object Detection
```bash
# Clone example repository
git clone https://github.com/example/object-detection-opi
cd object-detection-opi

# Install dependencies
pip3 install -r requirements.txt

# Run application
python3 detect_objects.py --camera 0
```

#### Example 2: Face Recognition Door Lock
```bash
# Setup face recognition system
# Configure camera
# Train model with authorized faces
# Implement access control logic
```

## Troubleshooting

### Common Issues

1. **Boot Problems**
   - Check SD card integrity
   - Verify power supply (adequate amperage)
   - Re-flash OS image

2. **Performance Issues**
   - Monitor temperature (add cooling if needed)
   - Check for background processes
   - Optimize model size

3. **Camera Not Working**
   ```bash
   # Check camera connection
   lsusb
   # Test camera
   fswebcam test.jpg
   ```

4. **Network Connectivity**
   ```bash
   # Check network interface
   ip addr show
   # Restart network service
   sudo systemctl restart networking
   ```

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

