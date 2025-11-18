替换系统中/etc/update-motd.d/01-orangepi-startup-text

显示内容如下：
```
  ___                                    ____   _      _     ___   ____              
 / _ \  _ __  __ _  _ __    __ _   ___  |  _ \ (_)    / \   |_ _| |  _ \  _ __  ___  
| | | || '__|/ _` || '_ \  / _` | / _ \ | |_) || |   / _ \   | |  | |_) || '__|/ _ \ 
| |_| || |  | (_| || | | || (_| ||  __/ |  __/ | |  / ___ \  | |  |  __/ | |  | (_) |
 \___/ |_|   \__,_||_| |_| \__, | \___| |_|    |_| /_/   \_\|___| |_|    |_|   \___/ 
                           |___/                                                     
Welcome to Orange Pi Ai Pro
This system is based on Ubuntu 22.04.3 LTS (GNU/Linux 5.10.0+ aarch64)

Welcome to Ubuntu 22.04.3 LTS with Linux 5.10.0+

System load:   17%             Up time:       7 hours, 46 minutes Local users:   3              
Memory usage:  7% of 15Gi      IP:            192.168.9.7    
CPU usage:     4%              Usage of /:    22% of 235G    
```
Orange Pi Ai Pro的主板使用的是华为海思 HiSilicon Ascend 310 或 Ascend 310B AI处理器,
Ubuntu对系统的适配不是太好，获得不了CPU的温度。

---

## Image Etcher Script (`img-etcher-Pi.sh`)

A safe CLI tool for writing Raspberry Pi/Orange Pi images to SD cards and USB drives. Enhanced with comprehensive safety features, multiple write modes, and automatic dependency management.

### Features

- **Multiple Format Support**: .img, .img.xz, .img.gz, .img.zst
- **Enhanced Safety**: System disk detection, removable device verification
- **Multiple Write Modes**: Balanced, Fast, Robust, and Legacy options
- **Progress Display**: Real-time write progress with `pv`
- **Automatic Dependencies**: Installs required packages automatically
- **Device Health Checks**: SMART monitoring and disk size validation
- **Verification Option**: Optional write verification with `cmp`

### Usage

```bash
# Basic usage
sudo ./scripts/img-etcher-Pi.sh <image-file.img>

# With verification
sudo ./scripts/img-etcher-Pi.sh <image-file.img.xz> --verify

# List available devices
sudo ./scripts/img-etcher-Pi.sh --list

# Force write to non-removable device (not recommended)
sudo ./scripts/img-etcher-Pi.sh <image-file> --force

# Show help
sudo ./scripts/img-etcher-Pi.sh --help
```

### Write Modes

1. **Balanced (Recommended)**: Good speed with progress display
2. **Fast Mode**: Maximum speed, no progress display
3. **Robust Mode**: Slower but more resilient to device errors
4. **Legacy Mode**: Traditional dd without optimizations

### Safety Features

- **System Disk Protection**: Automatically detects and prevents writing to system disks
- **Removable Device Detection**: Only shows USB/SD devices by default
- **Size Validation**: Checks if image fits on target device
- **Mount Point Analysis**: Prevents writing to mounted system partitions
- **SMART Health Checks**: Monitors device health when available

### Dependencies

The script automatically installs required packages:
- **Required**: `util-linux`, `grep`, `gawk`, `coreutils`, `diffutils`
- **Optional**: `pv`, `xz`, `gzip`, `zstd`, `bmap-tools`

### Examples

```bash
# Write compressed Ubuntu image
sudo ./scripts/img-etcher-Pi.sh ubuntu-22.04.img.xz --verify

# List all removable devices
sudo ./scripts/img-etcher-Pi.sh --list

# Write with robust mode for problematic SD cards
sudo ./scripts/img-etcher-Pi.sh image.img.gz
# Select option 3 when prompted for write mode
```

