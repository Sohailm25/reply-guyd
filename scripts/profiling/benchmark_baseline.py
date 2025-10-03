#!/usr/bin/env python3
"""
Benchmark baseline training performance for optimization comparison.

Usage:
    python scripts/profiling/benchmark_baseline.py --config config/experiments/polychromic_0.3.yaml
"""

import argparse
import time
import torch
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.train_model import load_config, setup_model_and_tokenizer


def profile_model_generation(model, tokenizer, num_samples=10):
    """Profile generation speed."""
    print("\n" + "="*60)
    print("Profiling Generation Speed")
    print("="*60)
    
    # Create sample prompts
    sample_prompt = "Generate a reply to: This is an interesting discussion about AI."
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
    
    # Time generations
    print(f"\nTiming {num_samples} generations...")
    times = []
    
    for i in range(num_samples):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Generation {i+1}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average generation time: {avg_time:.3f}s")
    print(f"   Throughput: {1/avg_time:.2f} generations/second")
    
    return avg_time


def profile_memory():
    """Profile GPU memory usage."""
    print("\n" + "="*60)
    print("GPU Memory Profile")
    print("="*60)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"📊 Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Peak:      {max_allocated:.2f} GB")
        
        # Get total GPU memory
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        print(f"   Total:     {total_memory:.2f} GB")
        print(f"   Usage:     {(allocated/total_memory)*100:.1f}%")
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_gb': max_allocated,
            'total_gb': total_memory,
            'usage_pct': (allocated/total_memory)*100
        }
    else:
        print("⚠️  CUDA not available")
        return None


def profile_attention_implementation(model):
    """Check what attention implementation is being used."""
    print("\n" + "="*60)
    print("Attention Implementation")
    print("="*60)
    
    # Try to detect attention implementation
    config = model.config
    
    if hasattr(config, '_attn_implementation'):
        impl = config._attn_implementation
        print(f"📊 Attention: {impl}")
        
        if impl == "flash_attention_2":
            print("   ✅ Using Flash Attention 2 (optimized)")
        elif impl == "sdpa":
            print("   ⚠️  Using SDPA (partially optimized)")
        else:
            print("   ⚠️  Using standard attention (not optimized)")
    else:
        print("   ⚠️  Could not detect attention implementation")
        print("   Likely using standard PyTorch attention")
    
    return getattr(config, '_attn_implementation', 'unknown')


def profile_compute_capability():
    """Check GPU compute capability."""
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"📊 Device: {props.name}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Multi-Processors: {props.multi_processor_count}")
        print(f"   Memory: {props.total_memory / 1e9:.2f} GB")
        
        # Check if Flash Attention is supported
        if props.major >= 8:  # Ampere or newer
            print("   ✅ Flash Attention 2 supported (SM 8.0+)")
        elif props.major >= 7:  # Volta
            print("   ⚠️  Flash Attention 1 supported (SM 7.0+)")
        else:
            print("   ❌ Flash Attention not supported (requires SM 7.0+)")
        
        return {
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processors': props.multi_processor_count,
            'memory_gb': props.total_memory / 1e9,
            'flash_attention_supported': props.major >= 7
        }
    else:
        print("⚠️  CUDA not available")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline performance")
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiments/polychromic_0.3.yaml',
        help='Path to experiment configuration'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of generations to profile'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/profiling/baseline_benchmark.json',
        help='Output file for benchmark results'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Baseline Performance Benchmark")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Samples: {args.num_samples}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load config
    print("\nLoading configuration...")
    config = load_config(args.config)
    
    # Profile GPU
    gpu_info = profile_compute_capability()
    
    # Load model
    print("\nLoading model (this may take a minute)...")
    start_load = time.time()
    model, tokenizer = setup_model_and_tokenizer(config)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f}s")
    
    # Profile attention
    attn_impl = profile_attention_implementation(model)
    
    # Profile memory
    memory_info = profile_memory()
    
    # Profile generation
    avg_gen_time = profile_model_generation(model, tokenizer, args.num_samples)
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': args.config,
        'gpu_info': gpu_info,
        'attention_implementation': attn_impl,
        'memory_info': memory_info,
        'model_load_time': load_time,
        'generation': {
            'avg_time_seconds': avg_gen_time,
            'throughput_per_second': 1 / avg_gen_time,
            'num_samples': args.num_samples
        }
    }
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Benchmark Complete")
    print("="*60)
    print(f"Results saved to: {args.output}")
    
    # Summary
    print("\n📊 SUMMARY")
    print("="*60)
    if gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute: SM {gpu_info['compute_capability']}")
    print(f"Attention: {attn_impl}")
    if memory_info:
        print(f"Memory Usage: {memory_info['allocated_gb']:.2f} GB ({memory_info['usage_pct']:.1f}%)")
    print(f"Avg Generation: {avg_gen_time:.3f}s")
    print(f"Throughput: {1/avg_gen_time:.2f} gen/sec")
    
    # Optimization suggestions
    print("\n💡 OPTIMIZATION SUGGESTIONS")
    print("="*60)
    
    if attn_impl != "flash_attention_2":
        print("⚠️  Flash Attention 2 not detected")
        print("   → Install: pip install flash-attn>=2.3.0")
        print("   → Add to model loading: attn_implementation='flash_attention_2'")
        print("   → Expected speedup: 1.5-2x")
    else:
        print("✅ Flash Attention 2 active")
    
    if gpu_info and gpu_info['flash_attention_supported']:
        print("✅ GPU supports Flash Attention")
    elif gpu_info:
        print("⚠️  GPU does not support Flash Attention (requires SM 7.0+)")
    
    print("\n📝 Next Steps:")
    print("   1. Review benchmark results in", args.output)
    print("   2. Apply Tier 1 optimizations (see docs/implementation/CUDA_OPTIMIZATION_PLAN.md)")
    print("   3. Re-run this benchmark to measure improvements")
    print("   4. For full training profile: use nsys or PyTorch profiler")


if __name__ == "__main__":
    main()

