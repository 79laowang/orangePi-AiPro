#!/usr/bin/env python3
"""
NPU Utilization Monitor for Qwen Chatbot
Monitors NPU metrics during inference to identify bottlenecks
"""

import subprocess
import time
import threading
import json
from datetime import datetime

# Test prompts
TEST_PROMPTS = [
    "你好",
    "What is the capital of France?",
    "写一首关于春天的诗"
]

def parse_npu_smi_output(text):
    """Parse npu-smi info output and extract metrics."""
    lines = text.strip().split('\n')
    metrics = {}

    for line in lines:
        if '310B4' in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    metrics['temp_c'] = float(parts[3])
                except:
                    pass
        elif '|               NA              |' in line or '0                     |' in line:
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    aicore_str = parts[3].strip()
                    if '%' in aicore_str:
                        metrics['aicore_percent'] = float(aicore_str.replace('%', ''))
                except:
                    pass
                try:
                    memory_str = parts[4].strip()
                    if '/' in memory_str:
                        mem_parts = memory_str.split('/')
                        metrics['memory_used_mb'] = int(mem_parts[0])
                        metrics['memory_total_mb'] = int(mem_parts[1])
                except:
                    pass

    return metrics


class NPUMonitor:
    """Monitor NPU utilization during inference."""

    def __init__(self, duration=60, interval=0.5):
        self.duration = duration
        self.interval = interval
        self.metrics_history = []
        self.running = False
        self.thread = None

    def _monitor_loop(self):
        """Internal monitoring loop."""
        start_time = time.time()
        while self.running and (time.time() - start_time) < self.duration:
            try:
                result = subprocess.run(
                    ['npu-smi', 'info'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                metrics = parse_npu_smi_output(result.stdout)
                metrics['timestamp'] = time.time() - start_time
                self.metrics_history.append(metrics)

                # Print real-time stats
                aicore = metrics.get('aicore_percent', 0)
                memory = metrics.get('memory_used_mb', 0)
                temp = metrics.get('temp_c', 0)
                print(f"[{metrics['timestamp']:5.1f}s] AICore: {aicore:5.1f}% | Memory: {memory:5d} MB | Temp: {temp:2.0f}°C")

            except Exception as e:
                print(f"Error monitoring: {e}")

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.metrics_history = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"[*] NPU Monitor started (duration: {self}s, interval: {self}s)")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"[*] NPU Monitor stopped. Collected {len(self.metrics_history)} samples")

    def analyze(self):
        """Analyze collected metrics."""
        if not self.metrics_history:
            return None

        aicore_values = [m.get('aicore_percent', 0) for m in self.metrics_history if m.get('aicore_percent') is not None]
        memory_values = [m.get('memory_used_mb', 0) for m in self.metrics_history if m.get('memory_used_mb') is not None]
        temp_values = [m.get('temp_c', 0) for m in self.metrics_history if m.get('temp_c') is not None]

        if not aicore_values:
            return {"error": "No valid AICore data collected"}

        analysis = {
            'aicore': {
                'min': min(aicore_values),
                'max': max(aicore_values),
                'avg': sum(aicore_values) / len(aicore_values),
                'samples': len(aicore_values)
            },
            'memory': {
                'min': min(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'avg': sum(memory_values) / len(memory_values) if memory_values else 0,
            },
            'temp': {
                'min': min(temp_values) if temp_values else 0,
                'max': max(temp_values) if temp_values else 0,
                'avg': sum(temp_values) / len(temp_values) if temp_values else 0,
            }
        }

        # Determine bottleneck
        avg_aicore = analysis['aicore']['avg']
        if avg_aicore >= 60:
            analysis['bottleneck'] = 'NPU Compute Bound (Normal)'
            analysis['recommendation'] = 'NPU is well utilized. Consider model optimization or hardware upgrade for better performance.'
        elif avg_aicore >= 20:
            analysis['bottleneck'] = 'Mixed CPU/NPU Bound'
            analysis['recommendation'] = 'Some overhead from CPU/NPU synchronization. Try optimizing data pipeline.'
        else:
            analysis['bottleneck'] = 'CPU/Memory Bound'
            analysis['recommendation'] = 'NPU is underutilized. Bottleneck is likely in CPU preprocessing or memory transfer.'

        return analysis


def print_analysis(analysis):
    """Print analysis results in a formatted way."""
    print("\n" + "=" * 60)
    print("NPU UTILIZATION ANALYSIS")
    print("=" * 60)

    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return

    print("\n📊 AICore Utilization:")
    print(f"   Average: {analysis['aicore']['avg']:5.1f}%")
    print(f"   Peak:    {analysis['aicore']['max']:5.1f}%")
    print(f"   Samples: {analysis['aicore']['samples']}")

    print("\n💾 Memory Usage:")
    print(f"   Average: {analysis['memory']['avg']:5.0f} MB")
    print(f"   Peak:    {analysis['memory']['max']:5.0f} MB")

    print("\n🌡️  Temperature:")
    print(f"   Average: {analysis['temp']['avg']:5.1f}°C")
    print(f"   Peak:    {analysis['temp']['max']:5.1f}°C")

    print("\n🔍 Bottleneck Analysis:")
    print(f"   {analysis['bottleneck']}")

    print("\n💡 Recommendation:")
    print(f"   {analysis['recommendation']}")

    print("\n" + "=" * 60)


def run_benchmark(prompt_index=0, monitor_duration=30):
    """
    Run NPU benchmark during inference.

    Args:
        prompt_index: Which test prompt to use (0-2)
        monitor_duration: How long to monitor (seconds)
    """
    if prompt_index >= len(TEST_PROMPTS):
        prompt_index = 0

    prompt = TEST_PROMPTS[prompt_index]

    print("\n" + "=" * 60)
    print("NPU BENCHMARK FOR QWEN CHATBOT")
    print("=" * 60)
    print(f"\nTest Prompt: \"{prompt}\"")
    print(f"Monitor Duration: {monitor_duration}s")
    print("\n" + "=" * 60)

    # Start monitor
    monitor = NPUMonitor(duration=monitor_duration, interval=0.5)
    monitor.start()

    # Wait a bit for monitor to stabilize
    time.sleep(2)

    # Send test request using curl
    print(f"\n[*] Sending test request to chatbot...")
    print("[*] Please wait for the response to complete...\n")

    try:
        # We'll use the OpenAI API format that Gradio provides
        # or simply let the user manually trigger the request
        print(">>> MANUAL TEST REQUIRED <<<")
        print("Please open http://0.0.0.0:7860 in your browser and send a message.")
        print("Press Enter when you start the generation...")
        input()

        # Continue monitoring
        print(f"\n[*] Monitoring for {monitor_duration} seconds...")
        print("[*] AICore utilization will be displayed in real-time above.\n")

        time.sleep(monitor_duration)

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")

    # Stop monitor
    monitor.stop()

    # Analyze and print results
    analysis = monitor.analyze()
    print_analysis(analysis)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/npu_benchmark_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_prompt': prompt,
            'metrics_history': monitor.metrics_history,
            'analysis': analysis
        }, f, indent=2)
    print(f"\n[*] Detailed results saved to: {filename}")

    return analysis


if __name__ == "__main__":
    import sys

    # Parse command line args
    prompt_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("\n" + "=" * 60)
    print("NPU BENCHMARK TOOL")
    print("=" * 60)
    print("\nUsage:")
    print("  python benchmark_npu.py [prompt_index] [duration]")
    print("\nTest Prompts:")
    for i, p in enumerate(TEST_PROMPTS):
        print(f"  {i}: {p}")

    # Run benchmark
    try:
        analysis = run_benchmark(prompt_idx, duration)
    except KeyboardInterrupt:
        print("\n[*] Benchmark interrupted")
