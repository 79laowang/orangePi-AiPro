#!/bin/sh
figlet -tk -w 120 "Orange Pi AI Pro" | lolcat
printf "Welcome to Orange Pi Ai Pro\n"
[ -r /etc/lsb-release ] && . /etc/lsb-release

if [ -z "$DISTRIB_DESCRIPTION" ] && [ -x /usr/bin/lsb_release ]; then
        # Fall back to using the very slow lsb_release utility
        DISTRIB_DESCRIPTION=$(lsb_release -s -d)
fi
printf "This system is based on %s (%s %s %s)\n" "$DISTRIB_DESCRIPTION" "$(uname -o)" "$(uname -r)" "$(uname -m)"
printf "\n"

# 隐藏的网络接口模式
HIDE_IP_PATTERN="^lo$|^docker.*|^veth.*|^br-.*|^tun.*"

function get_cpu_usage() {
    # 获取准确的CPU使用率
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk -F',' '{print 100 - $4}' | awk '{printf "%.0f%%", $1}' 2>/dev/null || echo "N/A")
    echo "$cpu_usage"
}

function get_npu_temp() {
    # Get NPU temperature from npu-smi
    local npu_temp=$(npu-smi info 2>/dev/null | awk '/310B4/ {print $8}')
    if [ -n "$npu_temp" ]; then
        echo "${npu_temp}'C"
    else
        echo "N/A"
    fi
}

function get_ip_addresses() {
    local ips=()
    for f in /sys/class/net/*; do
        local intf=$(basename $f)
        # 匹配需要隐藏的网络接口
        if [[ $intf =~ $HIDE_IP_PATTERN ]]; then
            continue
        else
            local tmp=$(ip -4 addr show dev $intf 2>/dev/null | grep -v "avahi" | awk '/inet/ {print $2}' | cut -d'/' -f1)
            # 只添加IP地址
            [[ -n $tmp ]] && ips+=("$tmp")
        fi
    done
    echo "${ips[@]}"
}

function get_user_load() {
    # 计算仅用户空间的负载（排除内核线程和D状态进程）
    local user_load=$(ps -eo stat,comm | grep -c "^R" | awk '{printf "%.2f", $1}')
    echo "$user_load"
}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Thresholds
# User load: only counts user processes in R state (running), excludes kernel threads
LOAD_THRESHOLD=$(nproc)
MEMORY_THRESHOLD=80
CPU_THRESHOLD=80
NPU_TEMP_THRESHOLD=65
DISK_THRESHOLD=80

function colorize_value() {
    local value=$1
    local threshold=$2
    local is_percentage=$3

    if [ "$is_percentage" = "yes" ]; then
        # Extract numeric value from percentage (e.g., "85%" -> 85)
        local num_value=$(echo $value | sed 's/%//')
        if [ "$num_value" -ge "$threshold" ]; then
            echo -e "${RED}${value}${NC}"
        else
            echo -e "${GREEN}${value}${NC}"
        fi
    else
        # For non-percentage values (load, temperature)
        local num_value=$(echo $value | sed "s/'C//")
        if [ "$num_value" != "N/A" ] && [ "$num_value" != "${num_value%.*}" ]; then
            # Floating point comparison
            if (( $(echo "$num_value >= $threshold" | bc -l) )); then
                echo -e "${RED}${value}${NC}"
            else
                echo -e "${GREEN}${value}${NC}"
            fi
        elif [ "$num_value" != "N/A" ]; then
            if [ "$num_value" -ge "$threshold" ]; then
                echo -e "${RED}${value}${NC}"
            else
                echo -e "${GREEN}${value}${NC}"
            fi
        else
            echo "$value"
        fi
    fi
}

# 获取系统信息
OS_INFO=$(lsb_release -ds 2>/dev/null || echo "Orange Pi 3.0.8 Bullseye")
KERNEL_INFO=$(uname -r)
LOAD_RAW=$(get_user_load)
LOAD_COLOR=$(colorize_value "$LOAD_RAW" "$LOAD_THRESHOLD" "no")
UPTIME=$(uptime -p | sed 's/up //')
USERS=$(who | wc -l)

# 内存信息 - 使用GB而不是GiB
MEMORY_BYTES=$(free -b | grep Mem: | awk '{print $2}')
MEMORY_TOTAL=$(awk "BEGIN {printf \"%.0fGB\", $MEMORY_BYTES/1000/1000/1000}")
MEMORY_USED=$(free | grep Mem: | awk '{printf("%.0f", $3/$2 * 100.0)}')
MEMORY_USED_COLOR=$(colorize_value "${MEMORY_USED}%" "$MEMORY_THRESHOLD" "yes")
MEMORY_USAGE="$MEMORY_USED_COLOR of $MEMORY_TOTAL"

# 磁盘信息
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
DISK_TOTAL=$(df -h / | awk 'NR==2 {print $2}')
DISK_USAGE_COLOR=$(colorize_value "$DISK_USAGE" "$DISK_THRESHOLD" "yes")
DISK_INFO="$DISK_USAGE_COLOR of ${DISK_TOTAL}"

# 获取CPU usage and NPU temp
CPU_USAGE=$(get_cpu_usage)
CPU_USAGE_COLOR=$(colorize_value "$CPU_USAGE" "$CPU_THRESHOLD" "yes")
NPU_TEMP=$(get_npu_temp)
NPU_TEMP_COLOR=$(colorize_value "$NPU_TEMP" "$NPU_TEMP_THRESHOLD" "no")

# 获取并格式化IP地址
IP_ADDRESSES=$(get_ip_addresses)
# 只取前3个IP地址，用空格分隔
FORMATTED_IPS=$(echo $IP_ADDRESSES | awk '{for(i=1;i<=3 && i<=NF;i++) printf "%s ", $i}' | sed 's/ $//')

# 显示信息
echo "Welcome to $OS_INFO with Linux $KERNEL_INFO"
echo ""
printf "System load:   %-22s Up time:       %-22s Local users:   %-22s\n" "$LOAD_COLOR" "$UPTIME" "$USERS"
printf "Memory usage:  %-22s IP:            %-22s\n" "$MEMORY_USAGE" "$(echo $IP_ADDRESSES | awk '{print $1}')"
printf "CPU usage:     %-22s NPU temp:      %-22s Usage of /:   %-s\n" "$CPU_USAGE_COLOR" "$NPU_TEMP_COLOR" "$DISK_INFO"

# 显示额外的IP地址（如果有多个）
IP_COUNT=$(echo $IP_ADDRESSES | wc -w)
if [ $IP_COUNT -gt 1 ]; then
    echo "               Additional IPs: $(echo $IP_ADDRESSES | cut -d' ' -f2-)"
fi
