# 🔧 Implementation Plan: Evaluation Framework Improvements

**Based on:** EVALUATION_FRAMEWORK_REVIEW.md  
**Date:** October 3, 2025  
**Status:** Planning phase - DO NOT implement yet

---

## 📊 Summary

**Overall Assessment:** 85% complete, need 2 critical improvements

**What works:**
- ✅ Test set design (stratified, holdout)
- ✅ Prompt strategy (consistent across models)
- ✅ Baseline comparison (4 models)
- ✅ Metric suite (diversity, quality, Pass@k)

**What needs fixing:**
- ⚠️ Temperature control (Priority 1 - CRITICAL)
- ⚠️ Multi-seed training (Priority 2 - IMPORTANT)

---

## 🎯 Priority 1: Temperature Strategy (CRITICAL)

### **Problem**
Current code uses `temperature=0.7` for all evaluation:
- Good for Pass@k (needs diversity) ✅
- Bad for quality metrics (needs determinism) ❌

### **Solution**
Split into two evaluation modes:

```python
# Mode 1: Deterministic single-generation (for ROUGE, BERTScore)
def generate_single_best(model, prompts):
    return generate(
        temperature=0.0,     # Greedy decoding
        do_sample=False,     # Deterministic
        n_per_prompt=1       # Single generation
    )

# Mode 2: Diverse multi-candidate (for Pass@k)
def generate_multiple_candidates(model, prompts):
    return generate(
        temperature=0.7,     # Sampling
        do_sample=True,      # Non-deterministic
        n_per_prompt=10      # Multiple candidates
    )
```

### **Files to Modify**

1. **`scripts/evaluation/evaluate_comprehensive.py`**
   - Lines 123-177: `generate_replies()` function
   - Add `evaluation_mode` parameter: "deterministic" or "diverse"
   - Update function signature:
     ```python
     def generate_replies(
         model,
         tokenizer,
         prompts: List[str],
         evaluation_mode: str = "diverse",  # ← NEW
         n_per_prompt: int = 10,
         max_new_tokens: int = 100
     ) -> List[List[str]]:
     ```
   - Add conditional logic:
     ```python
     if evaluation_mode == "deterministic":
         temperature = 0.0
         do_sample = False
         n_per_prompt = 1
     else:  # diverse
         temperature = 0.7
         do_sample = True
         # n_per_prompt from argument
     ```

2. **`scripts/evaluation/evaluate_comprehensive.py` (main loop)**
   - Lines 300-450: Main evaluation loop
   - Call `generate_replies()` twice:
     ```python
     # For quality metrics
     deterministic_replies = generate_replies(
         model, tokenizer, tweets,
         evaluation_mode="deterministic"
     )
     quality_metrics = compute_rouge_scores(...)
     
     # For Pass@k
     diverse_replies = generate_replies(
         model, tokenizer, tweets,
         evaluation_mode="diverse",
         n_per_prompt=10
     )
     passk_metrics = compute_passk(...)
     ```

3. **`generate_with_prompt_template()` function**
   - Lines 180-252: Similar changes
   - Add `evaluation_mode` parameter
   - Apply same temperature logic

### **Testing**
```bash
# Test deterministic mode is reproducible
python scripts/evaluation/evaluate_comprehensive.py \
    --baseline-lora path/to/model \
    --test-data data/test.jsonl \
    --output output/test_deterministic/ \
    --evaluation-mode deterministic

# Run twice, check if outputs are identical
diff output/test_deterministic/run1.json output/test_deterministic/run2.json
# Should be identical!
```

---

## 🎯 Priority 2: Multi-Seed Training (IMPORTANT)

### **Problem**
All experiments use single seed (42):
- No variance estimation
- Can't claim robustness
- Missing standard ML practice

### **Solution**
Train each experiment with 3 random seeds: 42, 123, 456

### **Files to Create**

1. **Config variants** (6 new files)
   ```
   config/experiments/baseline_seed42.yaml        (exists)
   config/experiments/baseline_seed123.yaml       (NEW)
   config/experiments/baseline_seed456.yaml       (NEW)
   config/experiments/polychromic_0.3_seed42.yaml (exists)
   config/experiments/polychromic_0.3_seed123.yaml (NEW)
   config/experiments/polychromic_0.3_seed456.yaml (NEW)
   ```

2. **`scripts/analysis/aggregate_seeds.py`** (NEW)
   ```python
   #!/usr/bin/env python3
   """
   Aggregate results across multiple random seeds.
   
   Usage:
       python scripts/analysis/aggregate_seeds.py \
           --results-dir output/evaluation/baseline/
           --seeds 42 123 456
   """
   
   def aggregate_metrics(results_dir, seeds):
       """
       Load results from each seed, compute mean ± std.
       
       Output:
           {
               "pass@10": {
                   "mean": 0.78,
                   "std": 0.03,
                   "seeds": [0.75, 0.79, 0.80]
               },
               ...
           }
       """
       pass
   ```

3. **Training script** (simple bash wrapper)
   ```bash
   #!/bin/bash
   # scripts/training/train_all_seeds.sh
   
   EXPERIMENT=$1  # "baseline" or "polychromic_0.3"
   SEEDS=(42 123 456)
   
   for seed in "${SEEDS[@]}"; do
       echo "Training ${EXPERIMENT} with seed ${seed}..."
       python scripts/training/train_model.py \
           --config config/experiments/${EXPERIMENT}_seed${seed}.yaml
   done
   ```

### **Config Changes**
For each new seed config:
1. Copy existing config
2. Change `random_seed: 42` → `123` or `456`
3. Update output paths:
   ```yaml
   output_dir: "./output/experiments/baseline/seed_123"
   name: "baseline-seed123"
   tags: ["baseline", "seed123"]
   ```

### **Evaluation Changes**
Update `evaluate_comprehensive.py`:
```python
# Add --seeds argument
parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                   help='Random seeds to evaluate (default: [42])')

# Evaluate each seed
for seed in args.seeds:
    model_path = f"{args.baseline_lora}/seed_{seed}"
    results = evaluate_model(model_path)
    save_results(results, f"results_seed_{seed}.json")

# Aggregate if multiple seeds
if len(args.seeds) > 1:
    aggregate_results(args.output, args.seeds)
```

---

## 🔍 Priority 3: Prompt Consistency Verification (LOW)

### **Status**
Likely already correct, just needs verification.

### **Check**
```bash
# Training prompt (data_module.py line 78)
"Generate an engaging Twitter reply to this tweet:\n\n{example.tweet}\n\nReply:"

# Evaluation prompt (evaluate_comprehensive.py line 143)
"Generate an engaging Twitter reply to this tweet:\n\n{prompt}\n\nReply:"
```

**Result:** ✅ IDENTICAL (just variable name differs: `example.tweet` vs `prompt`)

**Action:** No changes needed!

---

## 📝 Priority 4: Documentation (NICE TO HAVE)

### **Files to Create**

1. **`docs/implementation/EVALUATION_STRATEGY.md`**
   - Explain two-mode evaluation
   - Document why temp=0 for quality, temp=0.7 for Pass@k
   - Multi-seed rationale
   - Prompt consistency

2. **Update `RESEARCH_IMPLEMENTATION.md`**
   - Add evaluation methodology section
   - Reference new EVALUATION_STRATEGY.md
   - Show example results table with mean ± std

---

## 📋 Implementation Checklist

### **Phase 1: Temperature Control** (2-3 hours)
- [ ] Modify `generate_replies()` to accept `evaluation_mode`
- [ ] Add temperature/sampling logic for each mode
- [ ] Update `generate_with_prompt_template()` similarly
- [ ] Update main evaluation loop to call twice
- [ ] Test deterministic mode reproducibility
- [ ] Test diverse mode for Pass@k

### **Phase 2: Multi-Seed Configs** (30 minutes)
- [ ] Create `baseline_seed123.yaml`
- [ ] Create `baseline_seed456.yaml`
- [ ] Create `polychromic_0.3_seed123.yaml`
- [ ] Create `polychromic_0.3_seed456.yaml`
- [ ] Verify all paths updated correctly

### **Phase 3: Seed Aggregation** (2 hours)
- [ ] Create `scripts/analysis/aggregate_seeds.py`
- [ ] Implement `aggregate_metrics()`
- [ ] Add mean/std computation
- [ ] Create visualization of seed variance
- [ ] Update `evaluate_comprehensive.py` for multi-seed support

### **Phase 4: Training Wrapper** (15 minutes)
- [ ] Create `scripts/training/train_all_seeds.sh`
- [ ] Test with small dataset first
- [ ] Document usage in README

### **Phase 5: Documentation** (1 hour)
- [ ] Create `docs/implementation/EVALUATION_STRATEGY.md`
- [ ] Update `RESEARCH_IMPLEMENTATION.md`
- [ ] Add to `docs/README.md` index

---

## ⏱️ Time Estimates

| Phase | Time | When |
|-------|------|------|
| Phase 1: Temperature | 2-3 hours | Before evaluation |
| Phase 2: Configs | 30 min | Before training |
| Phase 3: Aggregation | 2 hours | Before paper writing |
| Phase 4: Wrapper | 15 min | Convenience |
| Phase 5: Docs | 1 hour | Before submission |
| **TOTAL** | **6 hours** | Spread over project |

---

## 🎯 Minimal vs. Complete Implementation

### **Minimal (Required for valid results)**
- ✅ Phase 1: Temperature control
- ✅ Phase 2: Multi-seed configs
- ⏭️ Phase 3: Manual aggregation
- ⏭️ Phase 4: Skip (train manually)
- ⏭️ Phase 5: Skip (explain in paper methods)

**Time:** 3 hours + training time

### **Complete (Publication-ready)**
- ✅ All 5 phases
- ✅ Automated workflows
- ✅ Comprehensive documentation

**Time:** 6 hours + training time

---

## 📊 Impact Analysis

### **Without These Changes:**
- Results: Valid but less rigorous
- Paper: Reviewers will ask about:
  - "Why temp=0.7 for ROUGE?" → Non-deterministic
  - "Only one seed?" → Not robust
  - Risk: Major revision or rejection

### **With These Changes:**
- Results: Statistically robust
- Paper: Addresses all standard concerns
- Reporting: "Pass@10: 0.78 ± 0.03 (mean ± std over 3 seeds)"
- Risk: Minimal pushback on methodology

---

## ✅ Decision Point

**Recommendation:** Implement **Phase 1** (temperature) + **Phase 2** (configs) now.

**Why:**
1. Phase 1: 3 hours, critical for fair comparison
2. Phase 2: 30 min, required for statistical robustness
3. Phases 3-5: Can be done while training runs

**Next Steps:**
1. Review this plan
2. If approved, implement Phase 1
3. Test deterministic mode
4. Create seed configs (Phase 2)
5. Begin training with all seeds

---

## 🚨 Important Notes

### **Temperature=0 Caveat**
Qwen3 documentation says:
> "DO NOT use greedy decoding, as it can lead to performance degradation"

**Our response:**
- This warning is for thinking mode (enable_thinking=True)
- We use enable_thinking=False ✅
- Greedy decoding (temp=0) is safe for our use case
- If concerned, use temp=0.3 instead of 0.0

### **Training Time**
With 3 seeds:
- Baseline: 4 hours × 3 = 12 hours
- Polychromic: 12 hours × 3 = 36 hours
- **Total: ~48 hours** (2 days on RunPod)
- Cost: ~$3 × 3 + $9 × 3 = $36 total

### **Prompt Verification Result**
✅ Training and evaluation prompts are identical!
- No changes needed
- Consistency verified

---

**Status:** Ready for review and approval  
**Estimated Implementation Time:** 3.5 hours (minimal) or 6 hours (complete)  
**Estimated Training Time:** 48 hours (with 3 seeds)  
**Estimated Cost:** $36 (with 3 seeds)

---

**Questions before proceeding?**
1. Should we use temp=0.0 or temp=0.3 for deterministic mode?
2. 3 seeds sufficient, or should we do 5?
3. Implement minimal or complete version?
4. Any other concerns about evaluation methodology?
