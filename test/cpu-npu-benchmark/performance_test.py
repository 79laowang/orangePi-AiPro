#!/usr/bin/env python3
"""
AI Performance Comparison Test: CPU vs NPU
Tests inference performance on Orange Pi AI Pro with Ascend NPU
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import os

# Try to import NPU support
try:
    import torch_npu
    # Initialize NPU device
    torch_npu.npu.set_device(0)
    NPU_AVAILABLE = True
    print("✓ NPU support detected!")
    print(f"✓ NPU device initialized: {torch_npu.npu.current_device()}")
except ImportError:
    NPU_AVAILABLE = False
    print("⚠ NPU support not available, will use CPU only")
except Exception as e:
    NPU_AVAILABLE = False
    print(f"⚠ NPU initialization failed: {e}")
    print("  Will use CPU only")

class SimpleCNN(nn.Module):
    """Simple CNN for testing - similar to ResNet architecture"""
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_system_info():
    """Get system information"""
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
        'memory': psutil.virtual_memory()._asdict(),
    }
    return info

def benchmark_inference(model, dummy_input, device, num_iterations=100):
    """Benchmark model inference"""
    model.eval()
    model.to(device)

    # Warm-up runs
    print(f"  Warming up (10 iterations)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Actual benchmark
    print(f"  Running {num_iterations} iterations...")
    times = []
    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.perf_counter()
            output = model(dummy_input)
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(inference_time)

            if (i + 1) % 20 == 0:
                print(f"    Completed {i + 1}/{num_iterations} iterations")

    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'throughput': 1000 / np.mean(times),  # inferences per second
        'memory_used': memory_after - memory_before,
    }

def print_results(device_name, results):
    """Print benchmark results"""
    print(f"\n{'='*60}")
    print(f"{device_name} Performance Results")
    print(f"{'='*60}")
    print(f"Mean Inference Time:    {results['mean']:.2f} ms")
    print(f"Standard Deviation:     {results['std']:.2f} ms")
    print(f"Min Inference Time:     {results['min']:.2f} ms")
    print(f"Max Inference Time:     {results['max']:.2f} ms")
    print(f"Throughput:             {results['throughput']:.2f} inferences/sec")
    print(f"Memory Used:            {results['memory_used']:.2f} MB")
    print(f"{'='*60}\n")

def create_comparison_plot(cpu_results, npu_results, output_path):
    """Create comparison plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Inference time comparison
    devices = ['CPU', 'NPU'] if npu_results else ['CPU']
    mean_times = [cpu_results['mean'], npu_results['mean']] if npu_results else [cpu_results['mean']]

    ax1.bar(devices, mean_times, color=['#ff6b6b', '#4ecdc4'][:len(devices)])
    ax1.set_title('Mean Inference Time Comparison')
    ax1.set_ylabel('Time (ms)')
    for i, v in enumerate(mean_times):
        ax1.text(i, v + max(mean_times) * 0.01, f'{v:.2f}ms', ha='center', va='bottom')

    # Throughput comparison
    throughputs = [cpu_results['throughput'], npu_results['throughput']] if npu_results else [cpu_results['throughput']]
    ax2.bar(devices, throughputs, color=['#ff6b6b', '#4ecdc4'][:len(devices)])
    ax2.set_title('Throughput Comparison')
    ax2.set_ylabel('Inferences/sec')
    for i, v in enumerate(throughputs):
        ax2.text(i, v + max(throughputs) * 0.01, f'{v:.2f}', ha='center', va='bottom')

    # Inference time distribution (histogram)
    ax3.hist(cpu_results['times'], bins=30, alpha=0.7, label='CPU', color='#ff6b6b')
    if npu_results:
        ax3.hist(npu_results['times'], bins=30, alpha=0.7, label='NPU', color='#4ecdc4')
    ax3.set_title('Inference Time Distribution')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # Speedup comparison
    if npu_results:
        speedup = cpu_results['mean'] / npu_results['mean']
        ax4.bar(['Speedup (NPU vs CPU)'], [speedup], color='#95e1d3')
        ax4.set_title(f'Performance Speedup: {speedup:.2f}x')
        ax4.set_ylabel('Speedup Factor')
        for i, v in enumerate([speedup]):
            ax4.text(i, v + speedup * 0.05, f'{v:.2f}x', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Performance comparison plot saved to: {output_path}")
    plt.show()

def main():
    print("\n" + "="*60)
    print("AI Performance Test: CPU vs NPU on Orange Pi AI Pro")
    print("="*60 + "\n")

    # Get system information
    print("System Information:")
    sys_info = get_system_info()
    print(f"  CPU Cores: {sys_info['cpu_count']}")
    if sys_info['cpu_freq']:
        print(f"  CPU Frequency: {sys_info['cpu_freq'].get('current', 'N/A')} MHz")
    print(f"  Total Memory: {sys_info['memory']['total'] / 1024 / 1024 / 1024:.2f} GB")
    print(f"  PyTorch Version: {torch.__version__}")
    if NPU_AVAILABLE:
        print(f"  NPU Support: Enabled")
    else:
        print(f"  NPU Support: Disabled")

    # Create model and dummy input
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=1000)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

    results = {}

    # CPU Benchmark
    print("\n" + "="*60)
    print("CPU Benchmark")
    print("="*60)
    cpu_device = torch.device('cpu')
    results['cpu'] = benchmark_inference(model, dummy_input, cpu_device, num_iterations=100)
    print_results("CPU", results['cpu'])

    # NPU Benchmark (if available)
    if NPU_AVAILABLE:
        print("\n" + "="*60)
        print("NPU Benchmark")
        print("="*60)
        try:
            npu_device = torch.device('npu')
            dummy_input_npu = dummy_input.to(npu_device)
            results['npu'] = benchmark_inference(model, dummy_input_npu, npu_device, num_iterations=100)
            print_results("NPU", results['npu'])

            # Calculate speedup
            speedup = results['cpu']['mean'] / results['npu']['mean']
            print(f"\n🚀 Performance Improvement: {speedup:.2f}x faster on NPU")
            print(f"   CPU: {results['cpu']['mean']:.2f}ms per inference")
            print(f"   NPU: {results['npu']['mean']:.2f}ms per inference")
        except Exception as e:
            print(f"⚠ Error running NPU benchmark: {e}")
            print("NPU may not be available or configured properly")
    else:
        print("\n⚠ NPU not available for testing")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if 'npu' in results:
        print(f"✓ Successfully compared CPU and NPU performance")
        print(f"✓ Speedup: {results['cpu']['mean']/results['npu']['mean']:.2f}x")
        print(f"✓ NPU is {((results['cpu']['mean']/results['npu']['mean']-1)*100):.1f}% faster")
    else:
        print(f"✓ CPU benchmark completed successfully")
        print(f"  Mean inference time: {results['cpu']['mean']:.2f}ms")
        print(f"  Throughput: {results['cpu']['throughput']:.2f} inferences/sec")

    print("="*60 + "\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"performance_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("AI Performance Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"System: Orange Pi AI Pro with Ascend NPU\n\n")

        f.write("CPU Results:\n")
        for key, value in results['cpu'].items():
            if key != 'times':
                f.write(f"  {key}: {value}\n")

        if 'npu' in results:
            f.write("\nNPU Results:\n")
            for key, value in results['npu'].items():
                if key != 'times':
                    f.write(f"  {key}: {value}\n")

            f.write(f"\nSpeedup: {results['cpu']['mean']/results['npu']['mean']:.2f}x\n")

    print(f"✓ Detailed results saved to: {results_file}\n")

    # Create visualization
    if 'npu' in results:
        plot_file = f"performance_comparison_{timestamp}.png"
        try:
            create_comparison_plot(results['cpu'], results['npu'], plot_file)
        except Exception as e:
            print(f"⚠ Could not create plot (may need display): {e}")

if __name__ == "__main__":
    main()
