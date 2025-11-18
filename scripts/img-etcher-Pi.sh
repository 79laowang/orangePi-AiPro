#!/usr/bin/env bash
# etcher-rpi.sh — Safe CLI tool for writing Raspberry Pi images (Fedora 40+)
# Author: ke wang + GPT-5 assistant
# Enhanced with safety features and better device detection

set -euo pipefail

BLOCK_SIZE=4M
VERSION="1.3.7"

# ====== Color output ======
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}ℹ ${NC}$1"; }
warn() { echo -e "${YELLOW}⚠ ${NC}$1"; }
error() { echo -e "${RED}❌ ${NC}$1"; }
success() { echo -e "${GREEN}✅ ${NC}$1"; }

# ====== Initialize variables ======
XZ_OK="no"
GZ_OK="no"
ZSTD_OK="no"
BMAP_OK="no"
PV_OK="no"
NUMFMT_OK="no"

# ====== Dependency management ======
check_and_install_dependencies() {
    local missing_packages=()
    local optional_packages=()
    
    # Required packages
    local required_cmds=("lsblk" "grep" "awk" "dd" "cmp")
    local required_packages=("util-linux" "grep" "gawk" "coreutils" "diffutils")
    
    # Optional but recommended packages
    local optional_cmds=("pv" "xz" "gzip" "zstd" "bmaptool" "numfmt")
    
    # Check required commands
    for i in "${!required_cmds[@]}"; do
        if ! command -v "${required_cmds[$i]}" &>/dev/null; then
            missing_packages+=("${required_packages[$i]}")
        fi
    done
    
    # Check optional commands and build package list
    for cmd in "${optional_cmds[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            case "$cmd" in
                "pv") optional_packages+=("pv") ;;
                "xz") optional_packages+=("xz") ;;
                "gzip") optional_packages+=("gzip") ;;
                "zstd") optional_packages+=("zstd") ;;
                "bmaptool") optional_packages+=("bmap-tools") ;;
                "numfmt") optional_packages+=("coreutils") ;;
            esac
        fi
    done
    
    # Install missing required packages if any
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_packages[*]}"
        info "Attempting to install missing packages..."
        
        if command -v dnf &>/dev/null; then
            if ! sudo dnf install -y "${missing_packages[@]}"; then
                error "Failed to install packages. Please check your network connection and try again."
                exit 1
            fi
        elif command -v yum &>/dev/null; then
            if ! sudo yum install -y "${missing_packages[@]}"; then
                error "Failed to install packages. Please check your network connection and try again."
                exit 1
            fi
        else
            error "Cannot install packages: no package manager found (dnf/yum)"
            info "Please manually install: sudo dnf install ${missing_packages[*]}"
            exit 1
        fi
        
        # Verify installation
        for i in "${!required_cmds[@]}"; do
            if ! command -v "${required_cmds[$i]}" &>/dev/null; then
                error "Failed to install ${required_cmds[$i]}. Please install manually: sudo dnf install ${required_packages[$i]}"
                exit 1
            fi
        done
        
        success "Required dependencies installed successfully"
    fi
    
    # Suggest optional packages
    if [[ ${#optional_packages[@]} -gt 0 ]]; then
        info "Optional packages for enhanced features: ${optional_packages[*]}"
        read -rp "👉 Install optional packages? [y/N] " install_optional
        if [[ "$install_optional" =~ ^[Yy]$ ]]; then
            if command -v dnf &>/dev/null; then
                if sudo dnf install -y "${optional_packages[@]}"; then
                    success "Optional packages installed"
                else
                    warn "Failed to install some optional packages, but continuing anyway..."
                fi
            fi
        fi
    fi
    
    # Set availability flags
    XZ_OK=$(command -v xz >/dev/null && echo "yes" || echo "no")
    GZ_OK=$(command -v gzip >/dev/null && echo "yes" || echo "no")
    ZSTD_OK=$(command -v zstd >/dev/null && echo "yes" || echo "no")
    BMAP_OK=$(command -v bmaptool >/dev/null && echo "yes" || echo "no")
    PV_OK=$(command -v pv >/dev/null && echo "yes" || echo "no")
    NUMFMT_OK=$(command -v numfmt >/dev/null && echo "yes" || echo "no")
    
    # Debug output
    if [[ "${DEBUG:-false}" == "true" ]]; then
        info "Dependency status: XZ=$XZ_OK, GZ=$GZ_OK, ZSTD=$ZSTD_OK, BMAP=$BMAP_OK, PV=$PV_OK, NUMFMT=$NUMFMT_OK"
    fi
}

# ====== Helper: list removable devices ======
list_devices() {
    info "Scanning removable USB/SD devices..."
    echo "------------------------------------"
    
    # Improved device detection
    if command -v lsblk &>/dev/null; then
        # Method 1: Use lsblk with better filtering
        echo "Device     Size    Type    Removable  Mountpoint"
        echo "------------------------------------"
        
        # List all block devices and filter for removable or common SD/USB types
        lsblk -d -o NAME,SIZE,TYPE,TRAN,RM,MOUNTPOINT,LABEL,MODEL | \
        while IFS= read -r line; do
            # Skip header
            if [[ "$line" == NAME* ]]; then
                continue
            fi
            
            local name size type trans rm_val mount label model
            read -r name size type trans rm_val mount label model <<< "$line"
            
            # Check if device is removable or likely to be USB/SD
            if [[ "$rm_val" == "1" ]] || \
               [[ "$trans" == "usb" ]] || \
               [[ "$trans" == "mmc" ]] || \
               [[ "$name" =~ ^sd[b-z]$ ]] || \
               [[ "$name" =~ ^mmcblk[0-9]+$ ]]; then
                printf "%-10s %-8s %-7s %-10s %s\n" \
                    "/dev/$name" "$size" "$type" "$rm_val" "${mount:-none}"
            fi
        done
        
        # If no removable devices found, show all non-system devices
        local found_devices
        found_devices=$(lsblk -d -o NAME,RM | awk '$2=="1" {print $1}' | wc -l)
        if [[ "$found_devices" -eq 0 ]]; then
            warn "No removable devices detected. Showing all non-system block devices:"
            echo ""
            echo "All block devices:"
            lsblk -d -o NAME,SIZE,TYPE,MOUNTPOINT,LABEL | grep -v -E "(sda|nvme0n1|vda)"
        fi
        
    else
        # Fallback method
        warn "lsblk not available. Using basic device detection..."
        find /dev -maxdepth 1 -name 'sd[a-z]' -o -name 'mmcblk[0-9]' 2>/dev/null | \
        while read -r dev; do
            if [[ -b "$dev" ]]; then
                echo "$dev"
            fi
        done
    fi
    
    echo "------------------------------------"
    info "Note: Only use devices marked as removable or SD/USB devices"
}

# ====== Check if device is system disk ======
is_system_disk() {
    local device=$1
    local device_name=$(basename "$device")
    
    # Debug: show what we're checking
    if [[ "${DEBUG:-false}" == "true" ]]; then
        info "Checking if $device is a system disk..."
    fi
    
    # Method 1: Check if device contains critical system mount points
    local system_mounts
    system_mounts=$(lsblk -n -o MOUNTPOINT "$device" 2>/dev/null | grep -E '^/$|/boot|/home|/var|/usr|/etc' | head -1)
    if [[ -n "$system_mounts" ]]; then
        if [[ "${DEBUG:-false}" == "true" ]]; then
            info "Found system mount: $system_mounts on $device"
        fi
        return 0
    fi
    
    # Method 2: Check if any partition is mounted as system using findmnt
    for mount in / /boot /home /var /usr /etc; do
        local source_device
        source_device=$(findmnt -n -o SOURCE "$mount" 2>/dev/null || true)
        if [[ -n "$source_device" ]]; then
            # Extract the base device name (remove partition number)
            local base_device
            base_device=$(echo "$source_device" | sed 's/[0-9]*$//')
            base_device=$(basename "$base_device")
            
            if [[ "$base_device" == "$device_name" ]]; then
                if [[ "${DEBUG:-false}" == "true" ]]; then
                    info "Found system partition: $source_device for $mount on $device"
                fi
                return 0
            fi
        fi
    done
    
    # Method 3: Check if this is the boot device by examining /proc/mounts and /etc/fstab
    local root_device
    root_device=$(findmnt -n -o SOURCE / 2>/dev/null | sed 's/[0-9]*$//' | xargs basename 2>/dev/null || true)
    if [[ "$root_device" == "$device_name" ]]; then
        if [[ "${DEBUG:-false}" == "true" ]]; then
            info "Device $device contains root filesystem"
        fi
        return 0
    fi
    
    # Method 4: Check if device name matches known system disk patterns AND is not removable
    local is_removable
    is_removable=$(lsblk -dn -o RM "$device" 2>/dev/null || echo "1")
    
    # Only consider it a system disk if it matches common system disk names AND is not removable
    if [[ "$is_removable" != "1" ]] && [[ "$device_name" =~ ^(sda|nvme0n1|vda)$ ]]; then
        if [[ "${DEBUG:-false}" == "true" ]]; then
            info "Non-removable system disk pattern matched: $device_name"
        fi
        return 0
    fi
    
    # Method 5: Check if device contains the currently running OS
    if [[ -d "/sys/block/$device_name" ]]; then
        local device_id
        device_id=$(readlink -f "/sys/block/$device_name" 2>/dev/null || true)
        local root_id
        root_id=$(findmnt -n -o SOURCE / 2>/dev/null | xargs -I {} readlink -f "/sys/block/{}" 2>/dev/null | head -1 || true)
        
        if [[ -n "$device_id" && -n "$root_id" && "$device_id" == "$root_id" ]]; then
            if [[ "${DEBUG:-false}" == "true" ]]; then
                info "Device $device is the root device"
            fi
            return 0
        fi
    fi
    
    if [[ "${DEBUG:-false}" == "true" ]]; then
        info "Device $device is NOT a system disk"
    fi
    return 1
}

# ====== Get disk size ======
get_disk_size() {
    local device=$1
    if command -v blockdev &>/dev/null; then
        blockdev --getsize64 "$device" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ====== Get file size ======
get_file_size() {
    local file=$1
    if [[ -f "$file" ]]; then
        stat -c%s "$file" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ====== Get uncompressed image size ======
get_uncompressed_size() {
    local image=$1
    case "$image" in
        *.xz)
            if [[ "$XZ_OK" == "yes" ]]; then
                xz --robot --list "$image" | awk '/totals/{print $5}' 2>/dev/null || echo "0"
            else
                echo "0"
            fi
            ;;
        *.gz)
            if [[ "$GZ_OK" == "yes" ]]; then
                gzip -l "$image" | awk 'NR==2{print $2}' 2>/dev/null || echo "0"
            else
                echo "0"
            fi
            ;;
        *.zst)
            # zstd doesn't have a built-in way to get uncompressed size easily
            # We'll return 0 and handle it differently
            echo "0"
            ;;
        *)
            get_file_size "$image"
            ;;
    esac
}

# ====== Format size for display ======
format_size() {
    local size=$1
    if [[ "$NUMFMT_OK" == "yes" ]] && [[ -n "$size" ]] && [[ "$size" -gt 0 ]]; then
        numfmt --to=si "$size" 2>/dev/null || echo "${size} bytes"
    else
        # Basic formatting without numfmt
        if [[ -n "$size" ]] && [[ "$size" -gt 1073741824 ]]; then
            echo "$((size / 1073741824)) GB"
        elif [[ -n "$size" ]] && [[ "$size" -gt 1048576 ]]; then
            echo "$((size / 1048576)) MB"
        elif [[ -n "$size" ]] && [[ "$size" -gt 1024 ]]; then
            echo "$((size / 1024)) KB"
        else
            echo "${size} bytes"
        fi
    fi
}

# ====== Show usage ======
show_usage() {
    echo "Usage:"
    echo "  sudo $0 <path-to-image.img|.xz|.gz|.zst> [--verify] [--force]"
    echo "  sudo $0 --list   # List removable devices"
    echo "  sudo $0 --help   # Show this help"
    echo "  sudo $0 --version # Show version"
    echo ""
    echo "Supported formats: .img, .img.xz, .img.gz, .img.zst"
    echo "Dependencies will be automatically checked and installed if needed"
}

# ====== Parse args ======
parse_arguments() {
    VERIFY=false
    FORCE=false
    
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi

    case "$1" in
        --list|-l)
            # For --list mode, we don't need to check all dependencies or require root
            list_devices
            exit 0
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        --version|-v)
            echo "etcher-rpi.sh v$VERSION"
            exit 0
            ;;
        *)
            IMAGE="$1"
            ;;
    esac

    for arg in "$@"; do
        case "$arg" in
            --verify)
                VERIFY=true
                ;;
            --force)
                FORCE=true
                ;;
        esac
    done

    if [[ -n "${IMAGE:-}" && ! -f "$IMAGE" ]]; then
        error "File not found: $IMAGE"
        exit 1
    fi
}

# ====== Select device ======
select_device() {
    list_devices
    echo ""
    
    # Get list of suggested devices
    local suggested_devices=()
    if command -v lsblk &>/dev/null; then
        while IFS= read -r line; do
            local name rm_val
            read -r name rm_val <<< "$line"
            if [[ "$rm_val" == "1" ]] && [[ -b "/dev/$name" ]]; then
                suggested_devices+=("$name")
            fi
        done < <(lsblk -d -n -o NAME,RM 2>/dev/null)
    fi

    if [[ ${#suggested_devices[@]} -eq 1 ]]; then
        info "Auto-detected removable device: /dev/${suggested_devices[0]}"
        read -rp "👉 Press Enter to use /dev/${suggested_devices[0]} or enter another device: " input
        DEVICE="${input:-/dev/${suggested_devices[0]}}"
    elif [[ ${#suggested_devices[@]} -gt 1 ]]; then
        info "Multiple removable devices detected:"
        for dev in "${suggested_devices[@]}"; do
            echo "  /dev/$dev"
        done
        read -rp "👉 Enter target device: " DEVICE
    else
        read -rp "👉 Enter target device (e.g. /dev/sdb or /dev/mmcblk0): " DEVICE
    fi

    # Clean device input
    DEVICE="/dev/$(basename "$DEVICE")"

    if [[ ! -b "$DEVICE" ]]; then
        error "Invalid block device: $DEVICE"
        exit 1
    fi
}

# ====== Enhanced safety checks ======
safety_checks() {
    local device=$1
    local image=$2

    # Check if device is removable
    local is_removable
    is_removable=$(lsblk -dn -o RM "$device" 2>/dev/null || echo "0")
    if [[ "$is_removable" != "1" ]]; then
        warn "Device $device does not appear to be removable! (RM=$is_removable)"
        if [[ "$FORCE" != "true" ]]; then
            read -rp "Type 'YES' to continue anyway: " confirm
            [[ "$confirm" != "YES" ]] && exit 1
        fi
    fi

    # Critical: check if device is system disk
    if is_system_disk "$device"; then
        error "CRITICAL: $device appears to be a system disk! Aborting for safety."
        exit 1
    fi

    # Enhanced disk size vs image size check
    local disk_size
    local compressed_size
    local uncompressed_size
    disk_size=$(get_disk_size "$device")
    compressed_size=$(get_file_size "$image")
    uncompressed_size=$(get_uncompressed_size "$image")
    
    info "Disk size: $(format_size "$disk_size")"
    info "Compressed image size: $(format_size "$compressed_size")"
    if [[ "$uncompressed_size" -gt 0 ]]; then
        info "Uncompressed image size: $(format_size "$uncompressed_size")"
    fi
    
    # Check if uncompressed image will fit
    if [[ "$uncompressed_size" -gt 0 && "$disk_size" -gt 0 && "$disk_size" -lt "$uncompressed_size" ]]; then
        error "Disk size ($(format_size "$disk_size")) is smaller than uncompressed image size ($(format_size "$uncompressed_size"))"
        read -rp "Type 'YES' to continue anyway: " confirm
        [[ "$confirm" != "YES" ]] && exit 1
    fi
    
    # Check if compressed image will fit (as fallback)
    if [[ "$uncompressed_size" -eq 0 && "$disk_size" -gt 0 && "$disk_size" -lt "$compressed_size" ]]; then
        error "Disk size ($(format_size "$disk_size")) is smaller than compressed image size ($(format_size "$compressed_size"))"
        read -rp "Type 'YES' to continue anyway: " confirm
        [[ "$confirm" != "YES" ]] && exit 1
    fi

    # Check for potential device issues
    info "Checking device health..."
    if command -v smartctl &>/dev/null && [[ "$device" =~ /dev/sd ]]; then
        if sudo smartctl -H "$device" 2>/dev/null | grep -q "FAILED"; then
            warn "SMART health check failed for $device"
            read -rp "Type 'YES' to continue anyway: " confirm
            [[ "$confirm" != "YES" ]] && exit 1
        fi
    fi

    # Final confirmation
    echo ""
    warn "THIS WILL COMPLETELY ERASE ALL DATA ON: $device"
    info "All partitions and data will be permanently destroyed!"
    echo ""
    
    read -rp "Type 'ERASE' to confirm and continue: " confirm
    if [[ "$confirm" != "ERASE" ]]; then
        info "Operation cancelled."
        exit 0
    fi
}

# ====== Unmount and clean device ======
unmount_device() {
    local device=$1
    info "Unmounting all partitions on $device..."
    
    # Unmount all partitions
    for partition in "${device}"*; do
        if [[ -b "$partition" ]] && findmnt -n "$partition" &>/dev/null; then
            umount "$partition" 2>/dev/null && success "Unmounted $partition" || warn "Failed to unmount $partition"
        fi
    done
    
    # Clear any existing partition table to avoid "no space" errors
    info "Clearing existing partition table..."
    dd if=/dev/zero of="$device" bs=1M count=10 status=none 2>/dev/null || true
    
    # Force sync
    sync
}

# ====== Optimized write methods ======
write_with_progress_optimized() {
    local input_cmd="$1"
    local output_device="$2"
    local image_size="$3"
    local image_name="$4"
    
    if [[ "$PV_OK" == "yes" && -n "$image_size" && "$image_size" -gt 0 ]]; then
        # Use pv for progress display with optimized settings
        info "Writing $image_name to $output_device (optimized mode)..."
        info "Press Ctrl+C to cancel (data may be incomplete)"
        echo ""
        
        # Use direct I/O to bypass cache and avoid "no space" errors
        if eval "$input_cmd" | pv -s "$image_size" -N "Writing" | dd of="$output_device" bs=$BLOCK_SIZE oflag=direct conv=fsync status=none; then
            return 0
        else
            return 1
        fi
    else
        # Fallback without progress - use dd with status and optimized settings
        warn "Progress display not available. Writing with basic status (optimized)..."
        if eval "$input_cmd" | dd of="$output_device" bs=$BLOCK_SIZE status=progress oflag=direct conv=fsync; then
            return 0
        else
            return 1
        fi
    fi
}

# Alternative fast write method without progress
write_fast_no_progress() {
    local input_cmd="$1"
    local output_device="$2"
    local image_name="$3"
    
    info "Fast writing $image_name to $output_device (no progress display)..."
    warn "No progress will be shown during write operation"
    
    # Maximum performance: no progress, large blocks, direct I/O
    if eval "$input_cmd" | dd of="$output_device" bs=8M oflag=direct conv=fsync status=none; then
        return 0
    else
        return 1
    fi
}

# Robust write method with error handling
write_robust() {
    local input_cmd="$1"
    local output_device="$2"
    local image_name="$3"
    
    info "Robust writing $image_name to $output_device..."
    info "This method is more resilient to device errors"
    
    # Use smaller blocks and direct I/O for better error handling
    if eval "$input_cmd" | dd of="$output_device" bs=1M oflag=direct conv=fsync status=progress; then
        return 0
    else
        return 1
    fi
}

# ====== Write image ======
write_image() {
    local image=$1
    local device=$2
    local image_name=$(basename "$image")
    
    info "Starting write process for $image_name to $device..."
    
    # Get image size for progress display
    local image_size=""
    if [[ "$image" == *.xz ]]; then
        if [[ "$XZ_OK" == "yes" ]]; then
            image_size=$(xz --robot --list "$image" | awk '/totals/{print $5}' 2>/dev/null || echo "")
        fi
    elif [[ "$image" == *.gz ]]; then
        if [[ "$GZ_OK" == "yes" ]]; then
            image_size=$(gzip -l "$image" | awk 'NR==2{print $2}' 2>/dev/null || echo "")
        fi
    else
        image_size=$(get_file_size "$image")
    fi
    
    # Try bmaptool first if available and image has .bmap file
    if [[ "$BMAP_OK" == "yes" && -f "${image%.*}.bmap" ]]; then
        info "Using bmaptool (faster method with .bmap file)"
        if bmaptool copy "$image" "$device"; then
            sync
            return 0
        else
            warn "bmaptool failed, falling back to dd..."
        fi
    fi
    
    # Ask user for write mode preference with improved options
    echo ""
    info "Select write mode:"
    echo "  1) Balanced (recommended) - Good speed with progress"
    echo "  2) Fast mode - Maximum speed, no progress display"
    echo "  3) Robust mode - Slower but more resilient to device errors"
    echo "  4) Legacy mode - Traditional dd without optimizations"
    read -rp "👉 Enter choice [1-4] (default: 1): " write_choice
    
    local write_success=false
    
    # Determine compression and write accordingly
    case "$image" in
        *.xz)
            if [[ "$XZ_OK" == "yes" ]]; then
                case "${write_choice:-1}" in
                    2)
                        if write_fast_no_progress "xz -dc -T0 '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    3)
                        if write_robust "xz -dc -T0 '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    4)
                        # Legacy mode
                        info "Using legacy write mode..."
                        if xz -dc -T0 "$image" | dd of="$device" bs=4M status=progress; then
                            write_success=true
                        fi
                        ;;
                    *)
                        # Balanced mode (default)
                        if write_with_progress_optimized "xz -dc -T0 '$image'" "$device" "$image_size" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                esac
            else
                error "Missing xz. Attempting to install..."
                if command -v dnf &>/dev/null; then
                    sudo dnf install -y xz
                    XZ_OK=$(command -v xz >/dev/null && echo "yes" || echo "no")
                    if [[ "$XZ_OK" == "yes" ]]; then
                        if write_with_progress_optimized "xz -dc -T0 '$image'" "$device" "$image_size" "$image_name"; then
                            write_success=true
                        fi
                    else
                        error "Failed to install xz. Please install manually: sudo dnf install xz"
                        exit 1
                    fi
                else
                    error "Please install xz manually: sudo dnf install xz"
                    exit 1
                fi
            fi
            ;;
        *.gz)
            if [[ "$GZ_OK" == "yes" ]]; then
                case "${write_choice:-1}" in
                    2)
                        if write_fast_no_progress "gzip -dc '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    3)
                        if write_robust "gzip -dc '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    4)
                        info "Using legacy write mode..."
                        if gzip -dc "$image" | dd of="$device" bs=4M status=progress; then
                            write_success=true
                        fi
                        ;;
                    *)
                        if write_with_progress_optimized "gzip -dc '$image'" "$device" "$image_size" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                esac
            else
                error "Missing gzip. Attempting to install..."
                if command -v dnf &>/dev/null; then
                    sudo dnf install -y gzip
                    GZ_OK=$(command -v gzip >/dev/null && echo "yes" || echo "no")
                    if [[ "$GZ_OK" == "yes" ]]; then
                        if write_with_progress_optimized "gzip -dc '$image'" "$device" "$image_size" "$image_name"; then
                            write_success=true
                        fi
                    else
                        error "Failed to install gzip. Please install manually: sudo dnf install gzip"
                        exit 1
                    fi
                else
                    error "Please install gzip manually: sudo dnf install gzip"
                    exit 1
                fi
            fi
            ;;
        *.zst)
            if [[ "$ZSTD_OK" == "yes" ]]; then
                case "${write_choice:-1}" in
                    2)
                        if write_fast_no_progress "zstd -dc '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    3)
                        if write_robust "zstd -dc '$image'" "$device" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                    4)
                        info "Using legacy write mode..."
                        if zstd -dc "$image" | dd of="$device" bs=4M status=progress; then
                            write_success=true
                        fi
                        ;;
                    *)
                        if write_with_progress_optimized "zstd -dc '$image'" "$device" "$image_size" "$image_name"; then
                            write_success=true
                        fi
                        ;;
                esac
            else
                error "Missing zstd. Install via: sudo dnf install zstd"
                exit 1
            fi
            ;;
        *)
            # Regular image file
            if [[ "$BMAP_OK" == "yes" ]]; then
                info "Trying bmaptool (faster) for regular image..."
                if bmaptool copy --nobmap "$image" "$device"; then
                    sync
                    return 0
                else
                    warn "bmaptool failed, falling back to dd..."
                fi
            fi
            
            case "${write_choice:-1}" in
                2)
                    if write_fast_no_progress "cat '$image'" "$device" "$image_name"; then
                        write_success=true
                    fi
                    ;;
                3)
                    if write_robust "cat '$image'" "$device" "$image_name"; then
                        write_success=true
                    fi
                    ;;
                4)
                    info "Using legacy write mode..."
                    if dd if="$image" of="$device" bs=4M status=progress; then
                        write_success=true
                    fi
                    ;;
                *)
                    if write_with_progress_optimized "cat '$image'" "$device" "$image_size" "$image_name"; then
                        write_success=true
                    fi
                    ;;
            esac
            ;;
    esac
    
    if [[ "$write_success" == "true" ]]; then
        # Final sync to ensure all data is written
        info "Final data synchronization..."
        sync
        return 0
    else
        error "Write process failed!"
        
        # Provide troubleshooting advice
        echo ""
        warn "Troubleshooting tips:"
        info "1. Check if the SD card is physically damaged or counterfeit"
        info "2. Try using a different SD card reader or USB port"
        info "3. Verify the image file is not corrupted"
        info "4. Try the 'Robust mode' for problematic devices"
        info "5. Check dmesg for device errors: sudo dmesg | tail -20"
        
        return 1
    fi
}

# ====== Verify write ======
verify_write() {
    local image=$1
    local device=$2
    
    info "Starting verification process..."
    warn "This may take a while depending on SD card speed..."
    
    local image_size
    image_size=$(get_file_size "$image")
    local verify_size=$(( image_size < 1073741824 ? image_size : 1073741824 )) # Max 1GB for verification
    
    case "$image" in
        *.xz)
            if [[ "$XZ_OK" == "yes" ]]; then
                xz -dc "$image" | head -c "$verify_size" | cmp -s - <(head -c "$verify_size" "$device")
            else
                warn "Cannot verify: xz not available"
                return 1
            fi
            ;;
        *.gz)
            if [[ "$GZ_OK" == "yes" ]]; then
                gzip -dc "$image" | head -c "$verify_size" | cmp -s - <(head -c "$verify_size" "$device")
            else
                warn "Cannot verify: gzip not available"
                return 1
            fi
            ;;
        *.zst)
            if [[ "$ZSTD_OK" == "yes" ]]; then
                zstd -dc "$image" | head -c "$verify_size" | cmp -s - <(head -c "$verify_size" "$device")
            else
                warn "Cannot verify: zstd not available"
                return 1
            fi
            ;;
        *)
            cmp -n "$verify_size" "$image" "$device"
            ;;
    esac
    
    local result=$?
    if [[ $result -eq 0 ]]; then
        success "Verification passed: Data written correctly!"
    else
        error "Verification failed: Data mismatch detected!"
        return $result
    fi
}

# ====== Main script ======
main() {
    echo "etcher-rpi.sh v$VERSION - Safe Raspberry Pi Image Writer"
    echo "=================================================="
    
    # Parse arguments first to handle --list without requiring dependencies
    parse_arguments "$@"
    
    # Check and install dependencies (skip for --list mode)
    if [[ -n "${IMAGE:-}" ]]; then
        check_and_install_dependencies
        
        # Require root for write operations
        if [[ $EUID -ne 0 ]]; then
            error "Please run as root: sudo $0 <image> [--verify]"
            exit 1
        fi
        
        # Select target device
        select_device
        
        # Run safety checks
        safety_checks "$DEVICE" "$IMAGE"
        
        # Unmount device
        unmount_device "$DEVICE"
        
        # Write image
        if write_image "$IMAGE" "$DEVICE"; then
            success "Write completed! Image successfully written to $DEVICE"
            
            # Verify if requested
            if [[ "$VERIFY" == true ]]; then
                verify_write "$IMAGE" "$DEVICE" || exit 1
            fi
            
            # Final sync and status
            sync
            echo ""
            success "Operation completed successfully!"
            info "Final device status:"
            lsblk -o NAME,SIZE,MOUNTPOINT,FSTYPE,LABEL "$DEVICE"
            echo ""
            info "You can now safely eject the SD card:"
            info "  sudo eject $DEVICE"
            info "Or physically remove it once the activity light stops blinking."
        else
            error "Write process failed. Please check the device and try again."
            exit 1
        fi
    fi
}

# Run main function with all arguments
main "$@"
