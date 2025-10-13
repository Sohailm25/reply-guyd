# Complete Re-run Instructions

## Summary of What Happened

✅ **Generation completed successfully** on RunPod
❌ **Anthropic API ran out of credits** after 76/500 examples  
❌ **JSON bug caused crash** when saving results  
✅ **All bugs now fixed!**

## Files That Were Successfully Created

Check on RunPod at `/workspace/Qwen3-8/output/evaluation/`:
- `diversity_metrics.json` ✓
- `quality_metrics.json` ✓  
- `passk_results.json` ✓
- `llm_judge_results.json` (partial - only 76 examples)
- `statistical_tests.json` (corrupt due to bug)

**The critical data (Pass@k, diversity, quality) exists!**

---

## Two Options to Complete

### Option 1: Skip LLM-Judge (Recommended) ⚡

**Pros:**
- FREE (no API costs)
- Fast (30-60 minutes)
- LLM-judge is nice-to-have, not essential for paper
- All main contributions work without it

**Steps:**

1. **Transfer bug fixes to RunPod** (9.2KB file)
   - I've created `~/Desktop/qwen_bugfixes.tar.gz` on your Mac
   - Upload to RunPod via web terminal
   - Extract in `/workspace/Qwen3-8/`

2. **Re-run evaluation (RunPod)**
```bash
cd /workspace/Qwen3-8

# Extract fixes
tar -xzf ~/qwen_bugfixes.tar.gz  # wherever you uploaded it

# Re-run WITHOUT LLM-judge
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge
```

3. **Continue with remaining steps**
```bash
# LoRA analysis
python scripts/analysis/analyze_lora_parameters.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --output output/analysis/lora_comparison

# Novel metrics
python scripts/evaluation/evaluate_novel_metrics.py \
  --results-dir output/evaluation/ \
  --output output/evaluation/novel_metrics

# Organize results
mkdir -p paper/figures paper/data
cp output/evaluation/pareto_frontier.pdf paper/figures/figure3_pareto_frontier.pdf
cp output/analysis/lora_comparison/lora_comparison.pdf paper/figures/figure4_lora_analysis.pdf  
cp output/evaluation/novel_metrics/der_comparison.pdf paper/figures/figure5_der_comparison.pdf
cp output/evaluation/novel_metrics/collapse_points.pdf paper/figures/figure6_collapse_points.pdf
cp output/evaluation/*.json paper/data/
cp output/analysis/lora_comparison/*.json paper/data/
cp output/evaluation/novel_metrics/*.json paper/data/
```

**Total time:** 1 hour  
**Total cost:** ~$0.75 (GPU only)

---

### Option 2: With LLM-Judge (After Adding Credits)

**Pros:**
- Complete evaluation
- LLM-judge comparison in paper

**Cons:**
- Costs $3-5 for API calls
- Requires adding Anthropic credits
- Takes same amount of time

**Steps:**

Same as Option 1, but:
- Add $10 Anthropic credits first
- Remove `--skip-llm-judge` flag
- Provide API key: `--anthropic-key your_key`

**Total cost:** $0.75 (GPU) + $3-5 (API) = $4-6

---

## Detailed Transfer Steps

Since SCP/rsync don't work with RunPod, use the web terminal:

### Method 1: Copy-Paste (Fastest for small fixes)

**On RunPod web terminal:**

```bash
cd /workspace/Qwen3-8

# Update statistical_tests.py
nano src/evaluation/statistical_tests.py
# Find line 53 and 92, change:
# 'significant': p_value < 0.05
# To:
# 'significant': bool(p_value < 0.05)
```

Or just download the fixed file from GitHub if you push it.

### Method 2: RunPod File Upload

1. Go to RunPod → Connect → Jupyter Lab
2. Upload `~/Desktop/qwen_bugfixes.tar.gz`
3. In terminal: `cd /workspace/Qwen3-8 && tar -xzf ~/qwen_bugfixes.tar.gz`

---

## What About the Existing Partial Results?

The existing files on RunPod are:
- ✅ Safe to keep: `diversity_metrics.json`, `quality_metrics.json`, `passk_results.json`
- ❌ Will be overwritten: `statistical_tests.json` (good - it was corrupt)
- ⚠️ Partial but usable: `llm_judge_results.json` (76 examples)

**Recommendation:** Just re-run and let it overwrite everything. The generation is the slow part, and that will be redone anyway.

---

## Expected Results After Re-run

**Files you'll have:**
1. `output/evaluation/diversity_metrics.json` ✓
2. `output/evaluation/quality_metrics.json` ✓
3. `output/evaluation/passk_results.json` ✓
4. `output/evaluation/statistical_tests.json` ✓ (FIXED)
5. `output/evaluation/pareto_frontier.pdf` ✓ **Figure 3**
6. `output/analysis/lora_comparison/lora_comparison.pdf` ✓ **Figure 4**
7. `output/evaluation/novel_metrics/der_comparison.pdf` ✓ **Figure 5**
8. `output/evaluation/novel_metrics/collapse_points.pdf` ✓ **Figure 6**

**Total: 4 figures + 6 JSON files**

---

## My Strong Recommendation

**Skip LLM-judge** (Option 1):
- Your main contributions are diversity metrics, Pass@k, DER, collapse point
- LLM-judge is subjective and costs money
- You can always add it later if reviewers ask
- Saves $5 and 30 minutes

**Then after getting results:**
- Download everything to Mac
- Review the 4 figures
- Extract numbers from JSON
- Start writing paper sections 5.3-5.5

---

## Questions?

- **Do I need to re-generate?** Probably yes, but it's only 30-60 min on GPU
- **Can I salvage partial results?** Not easily - cleaner to just re-run
- **Should I cache generations?** Yes for future work, but optional for now
- **LLM-judge or skip?** Skip it - saves time and money

Let me know when you've:
1. Added Anthropic credits (if doing LLM-judge) OR
2. Decided to skip LLM-judge

And I'll guide you through the re-run!


