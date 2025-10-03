# CUDA Optimization Plan

## 🎯 Goal
Optimize the Polychromic Trainer's computational bottlenecks using custom CUDA kernels to reduce training time from 12hrs to ~6-8hrs.

---

## 📊 Current Bottlenecks

### Identified in `src/training/polychromic_trainer.py`:

1. **Multiple Generation** (Lines 233-277)
   - Current: Sequential generation of N=3 replies
   - Time: ~70% of training overhead
   - Optimization target: Batched generation

2. **Diversity Computation** (Lines 132-168)
   - Current: PyTorch cosine similarity with O(N²) comparisons
   - Time: ~20% of training overhead
   - Optimization target: Fused CUDA kernel

3. **Attention Mechanism** (Implicit)
   - Current: Standard PyTorch attention
   - Time: Built into model.generate()
   - Optimization target: Flash Attention v2/v3

---

## 🚀 Three-Tiered Optimization Strategy

### Tier 1: Library-Based Optimizations (Week 1) ✅ PRIORITY
**Expected speedup: 1.5-2x | Effort: Low | Risk: None**

- [ ] Install and integrate Flash Attention 2
  ```bash
  pip install flash-attn>=2.3.0
  ```
  
- [ ] Modify model loading in `scripts/training/train_model.py`:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      attn_implementation="flash_attention_2",
      ...
  )
  ```

- [ ] Consider Unsloth (2x speedup claimed):
  ```bash
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  ```

- [ ] Optimize diversity encoder batching in `polychromic_trainer.py`:
  ```python
  embeddings = self.diversity_encoder.encode(
      texts,
      batch_size=32,  # Add batching
      normalize_embeddings=True  # Pre-normalize
  )
  ```

**Benchmark**: Run training for 100 steps, measure time/step

---

### Tier 2: Triton Kernels (Weeks 2-6) 🔄 IN PROGRESS
**Expected speedup: 2-3x | Effort: Medium | Risk: Low**

#### Milestone 1: Pairwise Distance Kernel (Weeks 2-3)
- [ ] Setup Triton environment:
  ```bash
  pip install triton
  ```

- [ ] Implement `src/triton_kernels/diversity_triton.py`
  - Fused pairwise cosine distance computation
  - Target: Replace `_semantic_diversity()` in polychromic_trainer

- [ ] Unit tests comparing Triton vs PyTorch outputs
  ```python
  # tests/unit/test_triton_kernels.py
  def test_pairwise_distances_triton():
      embeddings = torch.randn(10, 384)
      pytorch_result = pytorch_pairwise_distances(embeddings)
      triton_result = triton_pairwise_distances(embeddings)
      assert torch.allclose(pytorch_result, triton_result, atol=1e-4)
  ```

- [ ] Benchmark: Compare kernel execution time
  ```python
  # Time PyTorch version
  # Time Triton version
  # Target: 2-3x speedup
  ```

#### Milestone 2: Batched Generation Kernel (Weeks 4-6)
- [ ] Research batched generation patterns in Transformers
- [ ] Implement Triton kernel for parallel sampling
- [ ] Integrate into `generate_multiple_replies()`
- [ ] Benchmark end-to-end training speedup

---

### Tier 3: Raw CUDA Kernels (Weeks 7-12) 📚 LEARNING PHASE
**Expected speedup: 3-4x | Effort: High | Risk: Medium**

#### Learning Track (Weeks 7-9)
- [ ] Complete NVIDIA's "An Even Easier Introduction to CUDA"
  - [Tutorial Link](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  
- [ ] Read CUDA Programming Guide Chapters 2-3
  - Focus: Memory hierarchy (SRAM vs HBM)
  - Focus: Kernel execution model
  
- [ ] Udacity/NVIDIA DLI Course
  - "Getting Started with Accelerated Computing in Modern CUDA C++"
  - 8-10 hours with GPU access

- [ ] Study reference implementations:
  - `bitsandbytes/csrc/ops.cu` - Quantization kernels
  - `flash-attention/csrc/` - Attention optimizations
  - `unsloth` repo - LoRA-specific kernels

#### Implementation Track (Weeks 10-12)
- [ ] Setup CUDA development environment:
  ```bash
  mkdir -p src/cuda_kernels
  # Install CUDA Toolkit (if not present)
  # Setup torch.utils.cpp_extension
  ```

- [ ] Implement diversity kernel in `src/cuda_kernels/diversity_kernels.cu`
  - Start with naive implementation
  - Profile with `nvprof`
  - Optimize: memory coalescing, shared memory, etc.

- [ ] Build system setup:
  ```python
  # src/cuda_kernels/setup.py
  from torch.utils.cpp_extension import BuildExtension, CUDAExtension
  setup(name='diversity_cuda', ext_modules=[...])
  ```

- [ ] Integration with fallback:
  ```python
  try:
      import diversity_cuda
      USE_CUDA_KERNELS = True
  except ImportError:
      USE_CUDA_KERNELS = False
      logger.warning("CUDA kernels not available, using PyTorch")
  ```

- [ ] End-to-end training benchmark:
  - Baseline (PyTorch): 12 hours
  - Target (CUDA optimized): 6-8 hours

---

## 🔍 Profiling Strategy

### Tools
1. **NVIDIA Nsight Systems** (system-wide profiling)
   ```bash
   nsys profile --trace=cuda,nvtx \
     python scripts/training/train_model.py \
     --config config/experiments/polychromic_0.3.yaml \
     --data data/processed/training_data_*.jsonl
   ```

2. **PyTorch Profiler** (Python-level profiling)
   ```python
   from torch.profiler import profile, ProfilerActivity
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       trainer.train()
   prof.export_chrome_trace("trace.json")
   ```

3. **Custom Timing**
   ```python
   import time
   start = time.time()
   diversity_score = self._compute_batch_diversity(inputs)
   logger.info(f"Diversity computation: {time.time() - start:.3f}s")
   ```

### Key Metrics to Track
- Time per training step (seconds)
- GPU memory usage (GB)
- GPU utilization (%)
- Kernel execution time (ms)
- Memory bandwidth utilization (%)

---

## 📊 Success Criteria

| Optimization Tier | Speedup Target | Time Target | Status |
|-------------------|----------------|-------------|--------|
| Baseline (current) | 1.0x | 12 hours | ✅ Measured |
| Tier 1 (Flash Attn) | 1.5-2x | 6-8 hours | ⏳ Pending |
| Tier 2 (Triton) | 2-3x | 4-6 hours | ⏳ Pending |
| Tier 3 (CUDA) | 3-4x | 3-4 hours | ⏳ Pending |

**Minimum viable**: Tier 1 (Flash Attention) - Low effort, high impact  
**Optimal**: Tier 2 (Triton kernels) - Best effort/reward ratio  
**Stretch goal**: Tier 3 (Raw CUDA) - Learning investment for future projects

---

## 📚 Learning Resources

### CUDA Basics
- ✅ **Read**: `docs/research/current_research.md` (Lines 60-96) - Excellent CUDA overview
- [ ] [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ ] [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [ ] Udacity Course: "Intro to Parallel Programming" (free)

### Triton
- [ ] [OpenAI Triton Documentation](https://triton-lang.org/)
- [ ] [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [ ] Study: Unsloth's Triton kernels (real-world LoRA examples)

### Advanced CUDA
- [ ] Study `bitsandbytes/csrc/ops.cu` - 4-bit quantization
- [ ] Study `flash-attention/csrc/` - Attention optimization
- [ ] Study `PEFT/src/peft/tuners/lora/` - LoRA architecture

### Profiling
- [ ] [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [ ] [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)

---

## 🗂️ New Files to Create

```
src/
├── cuda_kernels/           # Tier 3
│   ├── __init__.py
│   ├── diversity_kernels.cu
│   ├── setup.py
│   └── tests/
│       └── test_cuda_kernels.py
│
├── triton_kernels/         # Tier 2
│   ├── __init__.py
│   ├── diversity_triton.py
│   ├── generation_triton.py
│   └── tests/
│       └── test_triton_kernels.py
│
└── training/
    └── optimized_trainer.py  # Unified optimized trainer

scripts/
└── profiling/
    ├── profile_training.py
    └── benchmark_kernels.py

docs/
└── implementation/
    └── CUDA_OPTIMIZATION_RESULTS.md  # Results tracking
```

---

## ⚠️ Risks and Mitigations

### Risk 1: CUDA Development Time
- **Impact**: Could take 3+ months to learn and implement
- **Mitigation**: Start with Tier 1 (immediate value), pursue Tier 2/3 as learning project
- **Alternative**: Use Tier 1 optimizations for research, learn CUDA for future projects

### Risk 2: Hardware Compatibility
- **Impact**: Custom CUDA kernels may not work on all GPUs
- **Mitigation**: Always implement PyTorch fallback
- **Testing**: Test on multiple GPU architectures (V100, A100, H100)

### Risk 3: Correctness
- **Impact**: Custom kernels could have subtle bugs
- **Mitigation**: 
  - Extensive unit tests comparing against PyTorch
  - Numerical stability checks (atol=1e-4)
  - Gradual rollout (single example → batch → full training)

### Risk 4: Maintenance Burden
- **Impact**: Custom kernels require ongoing maintenance
- **Mitigation**: 
  - Thorough documentation
  - Keep PyTorch fallback maintained
  - Only optimize true bottlenecks (80/20 rule)

---

## 📈 Progress Tracking

### Week 1: ⏳ In Progress
- [ ] Install Flash Attention 2
- [ ] Benchmark baseline training (100 steps)
- [ ] Integrate Flash Attention into model loading
- [ ] Benchmark with Flash Attention (100 steps)
- [ ] Document speedup

### Weeks 2-3: 📅 Scheduled
- [ ] Setup Triton environment
- [ ] Implement pairwise distance kernel
- [ ] Unit tests for Triton kernel
- [ ] Benchmark kernel execution time

### Weeks 4-6: 📅 Planned
- [ ] Batched generation kernel (Triton)
- [ ] Integration into polychromic_trainer
- [ ] End-to-end training benchmark

### Weeks 7-12: 📚 Learning Phase
- [ ] CUDA tutorials and courses
- [ ] Study reference implementations
- [ ] Implement raw CUDA kernels
- [ ] Final benchmarks

---

## 🎯 Next Steps

**This week:**
1. Install Flash Attention: `pip install flash-attn>=2.3.0`
2. Run baseline benchmark: `scripts/profiling/benchmark_baseline.py`
3. Integrate Flash Attention into `train_model.py`
4. Measure speedup

**This month:**
1. Complete Tier 1 optimizations
2. Start Triton learning track
3. Implement first Triton kernel

**This quarter:**
1. Complete Tier 2 (Triton optimizations)
2. Begin CUDA learning track
3. Prototype raw CUDA kernels

---

**Status**: 📍 Ready to start Tier 1 optimizations  
**Next Review**: After Tier 1 completion  
**Goal**: Reduce polychromic training time from 12hrs → 6-8hrs


