#!/bin/bash
# Real-time NPU Monitor for Qwen Chatbot
# Run this while testing the chatbot in browser

echo "=========================================="
echo "NPU Real-Time Monitor"
echo "=========================================="
echo ""
echo "Open http://0.0.0.0:7860 in your browser"
echo "Send a message and watch the metrics below"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="

# Initialize counters
count=0
sum_aicore=0
max_aicore=0
min_aicore=100

# Monitor loop
while true; do
    # Get current metrics
    output=$(npu-smi info 2>/dev/null)

    # Parse AICore (提取第二行数据中的百分比)
    aicore=$(echo "$output" | grep "0       0                     |" | awk '{print $4}' | sed 's/%//')

    # Parse memory
    mem=$(echo "$output" | grep "0       0                     |" | awk '{print $5}')

    # Parse temperature
    temp=$(echo "$output" | grep "310B4" | awk '{print $4}')

    # Update stats
    if [ ! -z "$aicore" ] && [ "$aicore" != "NA" ]; then
        sum_aicore=$((sum_aicore + $(echo $aicore | awk '{print int($1)}')))
        count=$((count + 1))

        # Track max/min
        if (( $(echo "$aicore > $max_aicore" | bc -l) )); then
            max_aicore=$aicore
        fi
        if (( $(echo "$aicore < $min_aicore" | bc -l) )); then
            min_aicore=$aicore
        fi
    fi

    # Print current status
    timestamp=$(date +"%H:%M:%S")
    printf "[$timestamp] AICore: %5s%% | Memory: %10s | Temp: %s°C | Max: %5s%% | Avg: %5s%%\n" \
        "$aicore" "$mem" "$temp" "$max_aicore" "$(echo "scale=1; $sum_aicore / $count" | bc 2>/dev/null || echo "0.0")"

    sleep 1
done
