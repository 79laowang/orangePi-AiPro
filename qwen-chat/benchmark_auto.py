#!/usr/bin/env python3
"""
Automated NPU Benchmark for Qwen Chatbot
Fully automated - sends request and monitors NPU during inference
"""

import subprocess
import time
import threading
import json
import requests
from datetime import datetime


def parse_npu_smi(text):
    """Parse npu-smi output."""
    lines = text.split('\n')
    metrics = {}
    for line in lines:
        if '310B4' in line:
            parts = line.split()
            try:
                metrics['temp'] = float(parts[3])
            except: pass
        elif '|               NA              |' in line or (line.strip().startswith('0') and 'NA' in line):
            parts = [p for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                try:
                    aicore = parts[1].strip()
                    if '%' in aicore:
                        metrics['aicore'] = float(aicore.replace('%', ''))
                except: pass
    return metrics


class NPUMonitor:
    """Monitor NPU in background thread."""

    def __init__(self):
        self.running = False
        self.metrics = []
        self.thread = None

    def _loop(self):
        while self.running:
            try:
                result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=2)
                m = parse_npu_smi(result.stdout)
                m['time'] = time.time()
                self.metrics.append(m)
                aicore = m.get('aicore', 0)
                print(f"  AICore: {aicore:5.1f}% | Temp: {m.get('temp', 0):.0f}°C")
            except: pass
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.metrics = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


def send_gradio_request(prompt="你好"):
    """Send request to Gradio ChatInterface."""
    # Gradio 6.x API endpoint
    url = "http://127.0.0.1:7860/api/predict"

    payload = {
        "data": [
            prompt,  # message
            []       # history (empty)
        ],
        "fn_index": 0,  # Usually 0 for ChatInterface predict function
        "session_hash": "benchmark_test"
    }

    print(f"\n[*] Sending request: '{prompt}'")
    print("[*] NPU metrics during inference:\n")

    try:
        response = requests.post(url, json=payload, timeout=120, stream=True)
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8').lstrip('data: '))
                    if 'msg' in data:
                        # Response token
                        token = data.get('data', [''])[0] if isinstance(data.get('data'), list) else ''
                        if token and len(token) > 0:
                            print(f"    Token: {token[:50]}..." if len(token) > 50 else f"    Token: {token}")
                except: pass
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def analyze_metrics(metrics):
    """Analyze collected metrics."""
    if not metrics:
        return None

    aicore = [m.get('aicore', 0) for m in metrics if m.get('aicore') is not None]
    temp = [m.get('temp', 0) for m in metrics if m.get('temp') is not None]

    if not aicore:
        return {'error': 'No AICore data'}

    avg = sum(aicore) / len(aicore)
    peak = max(aicore)

    # Determine bottleneck
    if avg >= 60:
        bottleneck = "NPU Compute Bound (Normal)"
        rec = "NPU well utilized. Performance limited by hardware."
    elif avg >= 20:
        bottleneck = "Mixed CPU/NPU"
        rec = "Some CPU overhead. Consider data pipeline optimization."
    else:
        bottleneck = "CPU/Memory Bound"
        rec = "NPU underutilized! Check preprocessing and data transfer."

    return {
        'aicore_avg': avg,
        'aicore_peak': peak,
        'aicore_samples': len(aicore),
        'temp_avg': sum(temp) / len(temp) if temp else 0,
        'bottleneck': bottleneck,
        'recommendation': rec
    }


def main():
    print("=" * 60)
    print("AUTOMATED NPU BENCHMARK")
    print("=" * 60)

    # Test prompts
    prompts = ["你好", "What is AI?"]

    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i+1}/{len(prompts)} ---")
        print(f"Prompt: '{prompt}'")

        # Start monitor
        monitor = NPUMonitor()
        monitor.start()

        time.sleep(1)  # Let monitor stabilize

        # Send request
        start = time.time()
        success = send_gradio_request(prompt)
        duration = time.time() - start

        time.sleep(2)  # Catch final metrics
        monitor.stop()

        # Analyze
        analysis = analyze_metrics(monitor.metrics)

        print(f"\n[*] Request completed in {duration:.1f}s")
        print(f"\n{'=' * 60}")
        print("RESULTS:")
        print(f"{'=' * 60}")
        if analysis and 'error' not in analysis:
            print(f"  AICore Average: {analysis['aicore_avg']:.1f}%")
            print(f"  AICore Peak:    {analysis['aicore_peak']:.1f}%")
            print(f"  Temperature:    {analysis['temp_avg']:.1f}°C")
            print(f"\n  📊 Bottleneck: {analysis['bottleneck']}")
            print(f"  💡 Recommendation: {analysis['recommendation']}")
        else:
            print(f"  ❌ {analysis.get('error', 'No data collected')}")
        print(f"{'=' * 60}\n")

        time.sleep(2)  # Cooldown between tests


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Interrupted")
