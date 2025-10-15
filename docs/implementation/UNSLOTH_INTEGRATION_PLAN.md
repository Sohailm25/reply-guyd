# Unsloth Integration Plan for Qwen3 Polychromic Training

## Purpose
Accelerate our LoRA fine-tuning & RLHF workflows by leveraging Unsloth’s fused kernels, tokenizer optimisations, and memory-efficient adapters. The end-state is a public PR to `unslothai/unsloth` adding first-class Qwen (Qwen2/3 8B) support, plus a gated integration path in our training stack (`train_model.py`, polychromic trainer, GRPO trainer).

---

## Current Stack Overview

| Layer | Location | Notes |
| --- | --- | --- |
| Model loading | `scripts/training/train_model.py::setup_model_and_tokenizer` | Uses vanilla `transformers.AutoModelForCausalLM` + bitsandbytes 4-bit config. |
| LoRA prep | `src/training/trainers/base.py::setup_lora_model` | Calls `prepare_model_for_kbit_training` then `peft.get_peft_model`. |
| Custom trainers | `src/training/trainers/polychromic.py`, `.../grpo.py` | Tight coupling to `transformers.Trainer`. Offline loss/generation mix. |
| Generation | Deep inside `Trainer.compute_loss` overrides | Requires KV cache (now re-enabled) and `model.generate`. |

Unsloth relies on:

1. `FastLanguageModel.from_pretrained` (wraps HF loading + kernel patches).
2. `FastLanguageModel.get_peft_model` (custom PEFT/LoRA attach).
3. Optional `SFTTrainer`/`DPOTrainer` wrappers (built on TRL’s trainers).
4. Model registry: `unsloth/models/__init__.py` enumerates supported base configurations.

Missing pieces for Qwen include rotary scaling params, RMSNorm epsilon, logits processing, and chat templates. We must upstream these to avoid maintaining a fork.

---

## Integration Strategy

We’ll work in two parallel tracks:

1. **Upstream Enablement (Unsloth repo)**
   - Add Qwen model metadata (hidden size, layered dims, vocab, RMSNorm). Likely touches `unsloth/models/_model_config.py` and `fast_language_model.py`.
   - Implement Qwen rotary embedding support (`rope_scaling`, `rope_theta`) and ensure attention kernels pick correct block size.
   - Extend tokenizer mapping: map Qwen tokenizer (tiktoken-based) or reuse HF `AutoTokenizer`.
   - Add regression tests: forward pass, LoRA attach, sample generate. Mirror existing tests for LLaMA/Mistral.
   - Update docs & README (supported models table) and release notes.

2. **Consumption in Our Repo**
   - Feature-gate via config flag `model.use_unsloth: bool`.
   - Abstract model loading: create `src/models/loading.py` helper calling either HF or Unsloth path (retaining quantization parity).
   - Update LoRA setup to delegate to Unsloth when active (`FastLanguageModel.get_peft_model`). Maintain dropout/rank parity with current YAML.
   - Ensure trainers accept Unsloth-wrapped models (they subclass `PreTrainedModel` so `Trainer` should work, but verify gradient checkpointing, caching toggles).
   - Handle generation: use `FastLanguageModel` helper to enable `use_cache=True` and check unsloth-specific `model._supports_flash_attn`. Keep our vectorised `generate_multiple_replies`.
   - Provide CLI toggle so baseline/polychromic/GRPO experiments can opt-in without separate configs.

---

## Detailed Task Breakdown

### Phase 0 – Research & Design (This Document)
1. Audit Unsloth code paths for LLaMA to understand required hooks (loader, config, LoRA, train).  
2. Extract Qwen model hyperparameters (layer counts, RMSNorm epsilon, rotary base) from HF config.  
3. Identify compatibility gaps: e.g., Qwen uses SwiGLU gating & built-in attention scaling.  
4. Draft upstream API changes and confirm no licensing conflicts (Apache-2.0 vs MIT).

### Phase 1 – Upstream Prototype (Fork of Unsloth)
1. Fork `unslothai/unsloth`, create branch `feature/qwen-support`.  
2. Implement Qwen config dataclass, register under `MODEL_TYPES`.  
3. Patch `FastLanguageModel._load_model` to recognize `QwenForCausalLM`. Ensure weight tying and RMS norms re-use.  
4. Modify LoRA helper to map Qwen attention/feed-forward module names (they differ from LLaMA).  
5. Validate with synthetic tests:
   - Forward pass on dummy input (no LoRA).  
   - Attach LoRA, run `loss.backward()` to ensure gradients propagate.  
   - Run `model.generate()` up to 32 tokens to verify KV caching and sampling speed.  
6. Add CI test to upstream repo (likely under `tests/models/test_fast_language_model.py`).  
7. Document new support in README & docs; open PR requesting maintainer review.

### Phase 2 – Integrate Unsloth Option Locally
1. Add dependency toggle (requirements file & optional extras). Provide installation script for GPU nodes (ensuring CUDA versions align).  
2. Introduce `ModelLoader` abstraction:
   ```python
   def load_base_model(config):
       if config.model.get("use_unsloth"):
           return load_unsloth_model(config)
       return load_hf_model(config)
   ```
3. Implement `load_unsloth_model` bridging:
   - Call `FastLanguageModel.from_pretrained(...)` with `max_seq_length=config['data']['max_length']`.  
   - Mirror quantization: Unsloth already supports QLoRA; verify 4-bit NF4 mapping.  
   - Cache CPU fallback for diversity encoder to avoid GPU conflicts.
4. Replace `setup_lora_model` with conditional path:
   - For Unsloth, call `FastLanguageModel.get_peft_model(model, r=..., lora_alpha=..., target_modules=[...])`.  
   - Ensure gradient checkpointing toggles use `model.gradient_checkpointing_enable`.  
5. Update polychromic & GRPO trainers:
   - Ensure `model.generate` signature matches (Unsloth extends HF model but confirm).  
   - Re-test `use_cache` toggling; unsloth already keeps KV caches enabled, so adapt our guard logic.
6. Wiring: CLI flag `--unsloth` or config key `model.framework: {"type": "hf"|"unsloth"}`.

### Phase 3 – Validation & Benchmarking
1. **Unit Tests**
   - Add tests under `tests/unit` verifying loader selects correct path and attaches LoRA without warning.  
   - Mock Unsloth objects if package absent to keep CPU CI afloat.
2. **Integration Tests**
   - Short polychromic training run (5 steps) on GPU verifying logs, avg generation time, loss progression.  
   - Compare throughput vs HF path (tokens/sec) using profiling script.
3. **Regression**
   - Ensure baseline training & evaluation still succeed when Unsloth disabled.  
   - Validate W&B logging unaffected.
4. **Upstream PR feedback loop**
   - Address maintainer review, update docs accordingly.  
   - When merged, bump dependency commit/tag here.

### Phase 4 – Rollout
1. Default to HF path; keep Unsloth opt-in until full soak.  
2. Document usage in `docs/implementation/` (setup instructions, GPU requirements).  
3. Update run scripts (`train_polychromic_variants.sh`) with optional `--unsloth`.  
4. Coordinate with infra: ensure RunPod base image contains Unsloth dependencies.

---

## Testing & Tooling Considerations
- **Pre-commit hook**: add lint/test target ensuring Unsloth import checks run only when dependency present.
- **CI Matrix**:  
  - CPU job: Unsloth path skipped (dependency optional).  
  - GPU job: run short training (polychromic + GRPO).
- **Benchmark Script**: create `scripts/profiling/benchmark_unsloth.py` to measure tokens/sec & GPU memory.

---

## Risks & Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Unsloth kernels incompatible with Qwen attention | Training crash | Start with small sequence lengths; fall back to HF path. |
| Divergent generation semantics (sampling differences) | Behavior drift | Run evaluation suite comparing outputs for fixed seeds. |
| Dependency bloat on RunPod images | Longer setup | Bake Unsloth wheel into custom Docker image; update `requirements-runpod.txt`. |
| Upstream PR delays | Blocked adoption | Maintain thin compatibility layer in repo while awaiting upstream merge. |

---

## Open Questions
1. Can Unsloth’s fused kernels coexist with our diversity encoder CPU usage without reintroducing cache stalls?  
2. Does Unsloth expose hooks needed for GRPO reference model cloning (frozen copy)?  
3. How does Unsloth handle Qwen’s `rope_scaling` defaults—do we need to implement custom patch?  
4. Should we upstream polychromic-specific benchmarking scripts alongside Qwen support?

---

## Next Actions
1. Stand up fork of Unsloth & begin Phase 1 prototype.  
2. Draft RFC for upstream maintainers outlining intended contribution.  
3. Prepare internal benchmark harness to compare HF vs Unsloth once prototype ready.
