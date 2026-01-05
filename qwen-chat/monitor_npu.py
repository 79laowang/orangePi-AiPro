#!/usr/bin/env python3
"""
Simple Real-time NPU Monitor - Fixed Version 2
"""

import subprocess
import time
import re

def get_npu_metrics():
    """Get current NPU metrics."""
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=2)
        lines = result.stdout.strip().split('\n')

        aicore = 0.0
        memory = "N/A"
        temp = 0.0

        for line in lines:
            # Parse device line: | 0       0                     | NA              | 0            14892/ 15610
            if '0       0' in line and 'NA' in line:
                # The format is: | Chip Device | Bus-Id | AICore% Memory-Usage | ...
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    data = parts[3]  # "0            14892/ 15610"
                    # Extract first number as AICore
                    match = re.search(r'(\d+)', data)
                    if match:
                        aicore = float(match.group(1))
                    # Extract memory like 14892/ 15610
                    mem_match = re.search(r'(\d+)\s*/\s*(\d+)', data)
                    if mem_match:
                        memory = f"{mem_match.group(1)}/{mem_match.group(2)}"

            # Parse temperature line: | 0       310B4                 | Alarm           | 0.0          52
            elif '310B4' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    data = parts[3]  # "0.0          52                1059  / 1059"
                    # Extract temperature (second number, usually 2 digits)
                    numbers = re.findall(r'(\d+\.?\d*)', data)
                    if len(numbers) >= 2:
                        try:
                            temp = float(numbers[1])
                        except: pass

        return aicore, memory, temp

    except Exception as e:
        return 0.0, "N/A", 0.0


def main():
    print("=" * 60)
    print("NPU Real-Time Monitor")
    print("=" * 60)
    print("\nOpen http://0.0.0.0:7860 in your browser")
    print("Send a message and watch the metrics below")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    count = 0
    sum_aicore = 0.0
    max_aicore = 0.0

    try:
        while True:
            aicore, memory, temp = get_npu_metrics()

            # Calculate running average (only count when AICore > 0)
            if aicore > 0:
                count += 1
                sum_aicore += aicore
                if aicore > max_aicore:
                    max_aicore = aicore
                avg = sum_aicore / count
            else:
                avg = sum_aicore / count if count > 0 else 0.0

            # Format output
            timestamp = time.strftime("%H:%M:%S")
            mem_short = memory.split('/')[0] if '/' in memory else memory

            print(f"[{timestamp}] AICore: {aicore:5.1f}% | Memory: {mem_short:>7} MB | Temp: {temp:4.0f}°C | Max: {max_aicore:5.1f}% | Avg: {avg:5.1f}%")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Samples: {count}")
        print(f"  Max AICore: {max_aicore:.1f}%")
        if count > 0:
            print(f"  Avg AICore: {sum_aicore/count:.1f}%")

        # Analysis
        print("\n" + "=" * 60)
        print("Analysis:")
        if count == 0:
            print("  ❌ No inference detected!")
            print("  Make sure to send a message in the chatbot.")
        elif max_aicore >= 60:
            print("  ✅ NPU well utilized (hardware bound)")
            print("  Recommendation: Performance limited by hardware")
        elif max_aicore >= 20:
            print("  ⚠️  Moderate NPU utilization")
            print("  Recommendation: Some CPU overhead, may optimize")
        else:
            print("  ❌ NPU underutilized!")
            print("  Recommendation: Bottleneck is CPU/memory, not NPU")
        print("=" * 60)


if __name__ == "__main__":
    main()
