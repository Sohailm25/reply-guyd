# Unsloth Integration Plan for Qwen3 Polychromic Training

## Purpose
Accelerate our LoRA fine-tuning & RLHF workflows by leveraging Unsloth’s fused kernels, tokenizer optimisations, and memory-efficient adapters. Unsloth already ships support for the Qwen family; our focus is to consume that path (and contribute upstream **only if** Qwen3-8B nuances are missing) while wiring a gated integration path in our training stack (`train_model.py`, polychromic trainer, GRPO trainer).

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

According to the Unsloth docs, Qwen 1.5/2 series models are supported out of the box. We must confirm whether the specific Qwen3-8B checkpoint we use aligns with their coverage (rope scaling, tokenizer behaviour, etc.). Any gaps discovered become upstream contribution candidates; otherwise, we only adapt our codebase.

---

## Integration Strategy

We’ll work in two parallel tracks:

1. **Compatibility Verification / Upstream Delta (Unsloth repo, conditional)**
   - Validate Unsloth’s current Qwen path against our Qwen3-8B checkpoint (rope config, SwiGLU variants, RMSNorm eps).
   - If unsupported, add the minimal deltas (model metadata, rotary parameters, tokenizer tweaks) and upstream PR; otherwise document that stock Unsloth is sufficient.
   - Ensure unit tests in Unsloth cover the Qwen3(8B) case we depend on before integration.

2. **Consumption in Our Repo**
   - Feature-gate via config flag `model.use_unsloth: bool`.
   - Abstract model & tokenizer loading so we can swap between HF and Unsloth without touching trainer code.
   - Update LoRA setup to delegate to Unsloth when active (`FastLanguageModel.get_peft_model`). Maintain dropout/rank parity with current YAML.
   - Ensure trainers accept Unsloth-wrapped models (they subclass `PreTrainedModel` so `Trainer` should work, but verify gradient checkpointing, caching toggles).
   - Handle generation: use `FastLanguageModel` helper to enable `use_cache=True` and check Unsloth-specific attributes. Keep our vectorised `generate_multiple_replies`.
   - Provide CLI/config toggles so baseline/polychromic/GRPO experiments can opt-in without separate configs.

---

## Detailed Task Breakdown

### Phase 0 – Research & Design (This Document)
1. Audit Unsloth code paths for Qwen (verify Qwen3-8B coverage).  
2. Extract any missing hyperparameters from our checkpoint and compare with Unsloth defaults.  
3. Decide if upstream work is necessary (record findings here).

### Phase 1 – Upstream Delta (only if required)
1. Fork `unslothai/unsloth`, create branch `feature/qwen3-support` (or similar).  
2. Add/adjust configuration to match Qwen3-8B specifics (rope scaling, tokenizer, LoRA target modules) if gaps exist.  
3. Extend Unsloth test coverage for Qwen3-8B (forward pass, LoRA attach, generation).  
4. Submit PR to Unsloth maintainers; once merged, record the minimum version/commit we depend on.

*(If upstream work is unnecessary, skip directly to Phase 2 once compatibility is confirmed.)*

### Phase 2 – Integrate Unsloth Option Locally
1. Add dependency toggle (requirements file & optional extras). Provide install docs for GPU nodes (ensuring CUDA versions align).  
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
   - Re-test `use_cache` toggling; Unsloth keeps KV caches enabled, so adapt our guard logic.
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
   - When merged (if applicable), bump dependency commit/tag here.

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
| Upstream PR delays | Blocked adoption | Maintain thin compatibility layer in repo while awaiting upstream merge (if needed). |

---

## Open Questions
1. Does Unsloth’s published Qwen support already include Qwen3-8B? (Gather exact version/commit references.)  
2. Can Unsloth’s fused kernels coexist with our diversity encoder CPU usage without reintroducing cache stalls?  
3. Does Unsloth expose hooks needed for GRPO reference model cloning (frozen copy)?  
4. Should we upstream polychromic-specific benchmarking scripts alongside any Unsloth contributions?

---

## Next Actions
1. Validate Unsloth Qwen coverage against our checkpoint (catalog differences, if any).  
2. Stand up fork of Unsloth *only if* gaps are found.  
3. Draft RFC for internal stakeholders once integration path is confirmed.  
4. Prepare internal benchmark harness to compare HF vs Unsloth once prototype ready.
