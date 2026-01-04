# System Info Display Script (sys-info.sh)

A colorful system monitoring dashboard for Orange Pi AI Pro that displays real-time system metrics with color-coded thresholds.

## Features

- **Rainbow ASCII banner** using figlet + lolcat
- **Color-coded metrics** - Green (healthy) / Red (exceeds threshold)
- **User-space load** - Excludes kernel threads for accurate measurement
- **Auto-detection** - CPU cores, NPU temperature, disk/memory usage

## Metrics Displayed

| Metric | Description | Threshold | Status Colors |
|--------|-------------|-----------|---------------|
| **System load** | User processes in R state (running) | CPU core count | Green: < 3, Red: >= 3 |
| **Up time** | System running time | - | No threshold |
| **Local users** | Number of logged-in users | - | No threshold |
| **Memory usage** | RAM used percentage | 80% | Green: < 80%, Red: >= 80% |
| **IP address** | Primary network IP | - | No threshold |
| **CPU usage** | CPU utilization percentage | 80% | Green: < 80%, Red: >= 80% |
| **NPU temp** | NPU temperature (310B4 chip) | 65°C | Green: < 65°C, Red: >= 65°C |
| **Usage of /** | Root disk usage | 80% | Green: < 80%, Red: >= 80% |

## Usage

```bash
./sys-info.sh
```

### Display Options

#### Option 1: Add to ~/.bashrc (per-user)
Add to your `~/.bashrc` for automatic display on login:

```bash
echo '/home/HwHiAiUser/bin/sys-info.sh' >> ~/.bashrc
```

#### Option 2: System-wide MOTD (recommended for all users)
Install as a dynamic MOTD script to display on every SSH/login session:

```bash
# 1. Copy script to bin directory (if not already there)
mkdir -p ~/bin
cp sys-info.sh ~/bin/sys-info.sh
chmod +x ~/bin/sys-info.sh

# 2. Create symlink in /etc/update-motd.d/
sudo ln -s ~/bin/sys-info.sh /etc/update-motd.d/01-orangepi-startup-text

# 3. Update MOTD (runs all scripts in /etc/update-motd.d/)
sudo update-motd

# 4. Test immediately (optional)
run-parts /etc/update-motd.d
```

**Note:** The `01-` prefix ensures this script runs first in the MOTD sequence, displaying at the top before other system messages.

## Thresholds

All thresholds are configured at the top of the script (lines 59-65):

```bash
LOAD_THRESHOLD=$(nproc)      # = 3 for 3-core system
MEMORY_THRESHOLD=80
CPU_THRESHOLD=80
NPU_TEMP_THRESHOLD=65
DISK_THRESHOLD=80
```

## Why User Load Instead of /proc/loadavg?

This Orange Pi ARM system includes kernel threads and uninterruptible (D state) processes in `/proc/loadavg`, causing inflated load values (~18 when idle).

**Example:**
```
/proc/loadavg:    17.42  ← Total (including 18 kernel threads in D state)
User load:         2.00  ← Actual user processes running
```

The `get_user_load()` function counts only user-space processes in **R state** (running/runnable), giving a more accurate picture of system load.

## Functions

| Function | Purpose |
|----------|---------|
| `get_cpu_usage()` | CPU utilization from `top` |
| `get_npu_temp()` | NPU temperature from `npu-smi` |
| `get_user_load()` | User process count (R state only) |
| `get_ip_addresses()` | Network IPs (hides docker, veth, br, tun, lo) |
| `colorize_value()` | Apply green/red color based on threshold |

## Requirements

- `figlet` - ASCII art banner
- `lolcat` - Rainbow colors
- `npu-smi` - NPU temperature
- `bc` - Floating point calculation

## Installation

```bash
sudo apt install figlet lolcat bc
```

## Customization

### Change NPU temperature threshold:
```bash
# Line 64
NPU_TEMP_THRESHOLD=60  # Default: 65
```

### Change colors:
```bash
# Lines 55-56
RED='\033[0;31m'      # Red for exceeded
GREEN='\033[0;32m'    # Green for healthy
YELLOW='\033[1;33m'   # Add warning color
```

### Add additional metrics:
```bash
# Add function
function get_gpu_temp() {
    local gpu_temp=$(your_command_here)
    echo "${gpu_temp}'C"
}

# Set threshold
GPU_TEMP_THRESHOLD=80

# Get and colorize value
GPU_TEMP=$(get_gpu_temp)
GPU_TEMP_COLOR=$(colorize_value "$GPU_TEMP" "$GPU_TEMP_THRESHOLD" "no")

# Display in printf
printf "GPU temp:      %-22s\n" "$GPU_TEMP_COLOR"
```

## Output Example

```
 ___                                  ____   _      _     ___   ____
/ _ \ _ __  __ _ _ __    __ _  ___   |  _ \ (_)    / \   |_ _| |  _ \  _ __  ___
| | | | '__|/ _` | '_ \  / _` | / _ \  | |_) || |   / _ \   | |  | |_) || '__|/ _ \
| |_| | |  | (_| | | | || (_| ||  __/  |  __/ | |  / ___ \  | |  |  __/ | |  | (_) |
 \___/|_|   \__,_|_| |_| \__, | \___|  |_|    |_| /_/   \_\|___| |_|    |_|   \___/
                          |___/
Welcome to Orange Pi Ai Pro
This system is based on Ubuntu 22.04.3 LTS (GNU/Linux 5.10.0+ aarch64)

Welcome to Ubuntu 22.04.3 LTS with Linux 5.10.0+

System load:   2.00         Up time:       1 hour, 56 minutes     Local users:   3
Memory usage:  12% of 16GB  IP:            192.168.9.7
CPU usage:     35%          NPU temp:      50'C                   Usage of /:   40% of 235G
```

## License

Free to use and modify.
