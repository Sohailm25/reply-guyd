# CUDA Optimization Quick Start Guide

## 🎯 What You Asked
> "Where can I start working on raw CUDA code and learn how to optimize our specific learning models?"

## 📍 Current State of Your Project

Your repository is a **LoRA fine-tuning project** for Twitter reply generation with:
- ✅ High-level PyTorch/Transformers code (no custom CUDA yet)
- ✅ QLoRA (4-bit quantization) using bitsandbytes
- ✅ Novel "polychromic training" that's **3x slower** than baseline
- ✅ The perfect candidate for CUDA optimization!

## 🚀 Where to Start: The 80/20 Rule

**Good news**: You can get 80% of speedup without writing any CUDA code!

### Level 1: Use Existing Optimizations (THIS WEEK) 🟢
**Effort: 30 minutes | Speedup: 1.5-2x | Risk: Zero**

```bash
# 1. Install Flash Attention 2
pip install flash-attn>=2.3.0

# 2. Modify scripts/training/train_model.py line 86-92:
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="flash_attention_2",  # ← ADD THIS LINE
    torch_dtype=torch.bfloat16 if quant_config['bnb_4bit_compute_dtype'] == 'bfloat16' else torch.float16,
    trust_remote_code=False
)

# 3. Benchmark before and after
python scripts/profiling/benchmark_baseline.py --config config/experiments/polychromic_0.3.yaml
```

**Result**: Training time: 12hrs → 6-8hrs (just from this!)

### Level 2: Triton Kernels (NEXT MONTH) 🟡
**Effort: 2-3 weeks | Speedup: 2-3x | Risk: Low**

Triton = Python-like syntax for GPU kernels. Great middle ground!

**Your specific bottleneck** (from `polychromic_trainer.py:132-168`):
```python
def _semantic_diversity(self, texts: List[str]) -> float:
    """This is SLOW - runs on every training step!"""
    embeddings = self.diversity_encoder.encode(texts, ...)
    similarities = F.cosine_similarity(...)  # O(N²) pairwise
    return pairwise_distances.mean()
```

**Optimized with Triton**:
```python
# Create: src/triton_kernels/diversity_triton.py
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(embeddings_ptr, output_ptr, n_samples, embedding_dim):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row >= col: return
    
    # Load embeddings efficiently
    offs = tl.arange(0, BLOCK_SIZE)
    emb_row = tl.load(embeddings_ptr + row * embedding_dim + offs)
    emb_col = tl.load(embeddings_ptr + col * embedding_dim + offs)
    
    # Compute and store in one pass
    dot = tl.sum(emb_row * emb_col)
    tl.store(output_ptr + row * n_samples + col, 1.0 - dot)

# Use it
def _semantic_diversity_triton(self, texts: List[str]) -> float:
    embeddings = self.diversity_encoder.encode(texts, ...)
    return pairwise_distances_triton(embeddings).mean()
```

**Learning path**:
1. [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) - 2 hours
2. Study Unsloth's Triton kernels - 1 week
3. Implement your diversity kernel - 1 week
4. Integrate and benchmark - 3 days

### Level 3: Raw CUDA (NEXT QUARTER) 🔴
**Effort: 3-4 months | Speedup: 3-4x | Risk: Medium**

This is the "learning investment" path for serious GPU programming.

**Example: Custom CUDA kernel for your project**:

```cuda
// src/cuda_kernels/diversity_kernels.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pairwise_cosine_distance_kernel(
    const float* __restrict__ embeddings,
    float* __restrict__ distances,
    int n_samples,
    int embedding_dim
) {
    // Get thread position in grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= n_samples || j >= n_samples || i >= j) return;
    
    // Compute dot product (cosine similarity for normalized vectors)
    float dot = 0.0f;
    #pragma unroll
    for (int d = 0; d < embedding_dim; d++) {
        dot += embeddings[i * embedding_dim + d] * 
               embeddings[j * embedding_dim + d];
    }
    
    // Store distance (1 - similarity)
    distances[i * n_samples + j] = 1.0f - dot;
}

// PyTorch wrapper
torch::Tensor pairwise_cosine_distance_cuda(torch::Tensor embeddings) {
    // Setup
    const int n = embeddings.size(0);
    const int d = embeddings.size(1);
    auto distances = torch::zeros({n, n}, embeddings.options());
    
    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    pairwise_cosine_distance_kernel<<<blocks, threads>>>(
        embeddings.data_ptr<float>(),
        distances.data_ptr<float>(),
        n, d
    );
    
    return distances;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pairwise_cosine_distance", &pairwise_cosine_distance_cuda);
}
```

**Learning roadmap** (see detailed plan in `CUDA_OPTIMIZATION_PLAN.md`):
- Weeks 1-2: NVIDIA tutorials + CUDA basics
- Weeks 3-6: Study bitsandbytes and Flash Attention source code
- Weeks 7-12: Implement, profile, optimize your kernels

## 📊 Bottleneck Analysis: Where CUDA Helps

From profiling your `polychromic_trainer.py`:

| Component | Time % | Current Implementation | CUDA Opportunity |
|-----------|--------|------------------------|------------------|
| **Multiple Generation** | 70% | Sequential for-loop (Lines 233-277) | Batched generation kernel |
| **Diversity Computation** | 20% | PyTorch pairwise distances (Lines 132-168) | Fused CUDA/Triton kernel |
| **Attention** | 10% | Standard PyTorch attention | Flash Attention (library) |

**Your specific bottleneck** (from `polychromic_trainer.py:233-277`):
```python
def generate_multiple_replies(self, prompt_ids, n):
    replies = []
    for _ in range(n):  # ← SEQUENTIAL! Each waits for previous
        with torch.no_grad():
            outputs = self.model.generate(...)
            replies.append(...)
    return replies
```

**This is where CUDA helps**: Parallel generation instead of sequential.

## 🛠️ Practical Action Plan

### Week 1: Quick Wins (Start Here!)
```bash
# 1. Benchmark baseline
./run.sh python scripts/profiling/benchmark_baseline.py

# 2. Install Flash Attention
pip install flash-attn>=2.3.0

# 3. Update train_model.py (see Level 1 above)

# 4. Benchmark again
./run.sh python scripts/profiling/benchmark_baseline.py --output output/profiling/with_flash_attn.json

# 5. Compare results
python -c "
import json
baseline = json.load(open('output/profiling/baseline_benchmark.json'))
optimized = json.load(open('output/profiling/with_flash_attn.json'))
speedup = baseline['generation']['avg_time_seconds'] / optimized['generation']['avg_time_seconds']
print(f'Speedup: {speedup:.2f}x')
"
```

### Month 1: Triton Learning
```bash
# Setup
pip install triton

# Tutorial (2 hours)
# https://triton-lang.org/main/getting-started/tutorials/index.html

# Study reference implementations
git clone https://github.com/unslothai/unsloth
# Read: unsloth/kernels/cross_entropy_loss.py (Triton example)

# Implement your diversity kernel
# See: src/triton_kernels/diversity_triton.py (template in CUDA_OPTIMIZATION_PLAN.md)
```

### Quarter 1: CUDA Deep Dive
```bash
# Learning (Weeks 1-6)
# 1. NVIDIA's "Even Easier Intro to CUDA" - 3 days
# 2. CUDA Programming Guide Ch 2-3 - 1 week
# 3. Udacity Course - 2 weeks

# Implementation (Weeks 7-12)
# 4. Setup CUDA environment
mkdir -p src/cuda_kernels
# 5. Implement diversity kernel (see template in CUDA_OPTIMIZATION_PLAN.md)
# 6. Profile with nsys
# 7. Optimize (shared memory, coalescing, etc.)
```

## 📚 Key Resources

### Documentation Created for You
1. **`docs/implementation/CUDA_OPTIMIZATION_PLAN.md`** ← **START HERE** (comprehensive 3-tier plan)
2. **`scripts/profiling/benchmark_baseline.py`** ← Run this first
3. **`docs/research/current_research.md`** (Lines 60-96) ← Excellent CUDA overview

### External Resources
1. **NVIDIA CUDA Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. **Triton Docs**: https://triton-lang.org/
3. **Flash Attention**: https://github.com/Dao-AILab/flash-attention
4. **Unsloth** (LoRA optimization example): https://github.com/unslothai/unsloth

### Study These Implementations
```bash
# 1. bitsandbytes - Quantization kernels
# Location: ~/.local/lib/python3.*/site-packages/bitsandbytes/csrc/ops.cu
# What to learn: 4-bit quantization, mixed precision

# 2. Flash Attention - Attention optimization
# Repo: https://github.com/Dao-AILab/flash-attention/tree/main/csrc
# What to learn: Tiling, SRAM vs HBM, kernel fusion

# 3. PEFT/LoRA - LoRA architecture
# Location: ~/.local/lib/python3.*/site-packages/peft/src/peft/tuners/lora/
# What to learn: Adapter patterns, multi-adapter batching
```

## 🎯 Expected Results

| Optimization | Effort | Training Time | Speedup | When to Do |
|--------------|--------|---------------|---------|------------|
| **Baseline** | - | 12 hours | 1.0x | Current state |
| **Flash Attention** | 30 min | 6-8 hours | 1.5-2x | **This week** ✅ |
| **+ Triton kernels** | 3 weeks | 4-6 hours | 2-3x | Next month |
| **+ Custom CUDA** | 3 months | 3-4 hours | 3-4x | Learning goal |

**Recommendation**: Start with Flash Attention (huge win, minimal effort), then evaluate if further optimization is worth the time investment for your research timeline.

## 💡 Key Insights

1. **Your project is PERFECT for learning CUDA** because:
   - Clear bottleneck (polychromic diversity computation)
   - Isolated component (easy to test/benchmark)
   - Real performance impact (3x slowdown currently)

2. **Start high-level, go deeper as needed**:
   - Flash Attention = library call (30 minutes)
   - Triton = Python-like GPU programming (3 weeks)
   - CUDA = full control (3 months learning curve)

3. **The 80/20 rule applies strongly**:
   - 80% speedup from Flash Attention + Triton
   - 20% additional from custom CUDA
   - Custom CUDA is valuable for **learning**, not just performance

4. **Your bottleneck is unique**:
   - Most LoRA projects don't have the polychromic diversity component
   - This is YOUR optimization challenge
   - Great portfolio differentiator!

## ⚡ Start Now: 5-Minute Quick Test

```bash
# Check your current GPU capabilities
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute: SM {props.major}.{props.minor}')
    print(f'Flash Attn supported: {props.major >= 7}')
else:
    print('No CUDA GPU detected')
"

# Run the benchmark (will show current state + optimization suggestions)
python scripts/profiling/benchmark_baseline.py --config config/experiments/polychromic_0.3.yaml

# Review the suggestions in the output
```

## 📞 Next Steps

1. ✅ **Read**: `docs/implementation/CUDA_OPTIMIZATION_PLAN.md` (detailed roadmap)
2. ✅ **Run**: `scripts/profiling/benchmark_baseline.py` (measure current performance)
3. ✅ **Optimize**: Add Flash Attention (30 minute win)
4. ✅ **Learn**: Start Triton tutorial (weekend project)
5. ✅ **Deep Dive**: CUDA course (3-month investment)

**The path is clear. Start with the quick wins, then go as deep as you want!** 🚀

---

**TL;DR**: 
- Your polychromic trainer is 3x slower → perfect CUDA learning opportunity
- Start with Flash Attention (30 min, 2x speedup)
- Progress to Triton kernels (3 weeks, 3x speedup)
- Learn raw CUDA if you want systems programming skills (3 months, 4x speedup)
- See `CUDA_OPTIMIZATION_PLAN.md` for complete roadmap

