# RunPod Evaluation Re-run Guide

## What Went Wrong

1. **Anthropic API credits too low** - Ran out after 76/500 examples
2. **JSON serialization bug** - Script crashed when saving statistics (FIXED ✅)
3. **LLM-judge incomplete** - Only $0.41 spent (not the full $3-5 needed)

## What Actually Worked ✅

Looking at your RunPod files:
- ✅ `diversity_metrics.json` (503 bytes) - SAVED
- ✅ `quality_metrics.json` (320 bytes) - SAVED
- ✅ `passk_results.json` (150 bytes) - SAVED
- ⚠️ `llm_judge_results.json` (178K) - Partial (76/500)
- ⚠️ `statistical_tests.json` (300 bytes) - Incomplete/corrupt

**Good news:** The expensive part (model generation) likely completed! We just need to re-run the LLM-judge and statistical parts.

---

## Bugs Fixed ✅

1. **JSON serialization bug** - Added `sanitize_for_json()` to handle numpy bool, NaN, Inf
2. **Type conversion** - All numpy types properly converted to Python native types

---

## Can We Save Generations? YES!

### Answer to Your Question:

**YES - We should absolutely save generations!** Here's why:

| Without Caching | With Caching |
|----------------|--------------|
| Crash → Re-generate all 10,000 outputs (1hr) | Crash → Just re-run analysis (5min) |
| Try new metric → Re-generate (1hr) | Try new metric → Load saved (5min) |
| Different evaluation → Re-generate (1hr) | Different evaluation → Load saved (5min) |
| Cost: GPU time | Cost: Storage (~500MB JSON) |

**I've created two approaches for you:**

---

## Approach 1: Simple Re-run (No Caching) - Quickest

Just re-run the fixed evaluation with more Anthropic credits.

### On RunPod:
```bash
cd /workspace/Qwen3-8

# Pull the latest code with fixes
# (You'll need to get the updated files to RunPod)

# Or just re-run with skip-llm-judge to avoid API costs
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge

# This skips the $5 LLM-judge cost, completes in 30-60 minutes
```

**Time:** 30-60 minutes  
**Cost:** $0.50-1.00 (GPU only)  
**No API credits needed**

---

## Approach 2: Save Generations First (Smarter) - Recommended

Save generations once, then run multiple analyses without re-generating.

### Step 1: Save Generations (30-60 min, one-time)

```bash
cd /workspace/Qwen3-8

python scripts/evaluation/save_generations.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/generations/ \
  --n-generations 10 \
  --max-examples 500
```

**Output:**
- `output/generations/baseline_lora_generations.json` (~250MB)
- `output/generations/polychromic_lora_generations.json` (~250MB)
- `output/generations/prompts.json` (reference)

### Step 2: Run Evaluation on Saved Generations (5 min, repeatable)

```bash
python scripts/evaluation/evaluate_from_saved.py \
  --generations output/generations/ \
  --output output/evaluation/ \
  --skip-llm-judge
```

**Benefits:**
- If evaluation crashes → Just re-run Step 2 (5 min vs 60 min)
- Try different metrics → Just re-run Step 2
- Add LLM-judge later → Just re-run Step 2 with API key

---

## Which Files Need Updating on RunPod?

### Files with bug fixes:
1. ✅ `src/evaluation/statistical_tests.py` - JSON serialization fix
2. ✅ `scripts/evaluation/evaluate_comprehensive.py` - CPU support (already there)

### New files (optional, for caching):
3. ⭐ `scripts/evaluation/save_generations.py` - Save generations
4. ⭐ `scripts/evaluation/evaluate_from_saved.py` - Analyze saved generations

---

## Quick Re-run Instructions (After Adding Credits)

### Option A: Just Fix and Re-run (Simplest)

```bash
# On Mac - update the bug fix on RunPod
cat > /tmp/fix_stats.py << 'EOF'
import fileinput
import sys

for line in fileinput.input('/workspace/Qwen3-8/src/evaluation/statistical_tests.py', inplace=True):
    line = line.replace("'significant': p_value < 0.05", "'significant': bool(p_value < 0.05)")
    sys.stdout.write(line)
EOF

# Transfer to RunPod
scp -i ~/.ssh/id_ed25519 /tmp/fix_stats.py kz142xqyz00wbl-644111eb@ssh.runpod.io:/tmp/

# On RunPod - apply fix
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io
cd /workspace/Qwen3-8
python /tmp/fix_stats.py

# Re-run evaluation
tmux new -s qwen-eval
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge

# Detach: Ctrl+b, d
```

### Option B: Transfer All Fixed Files (Better)

Since I've fixed the files on your Mac, let's create a minimal update package:

```bash
# On Mac
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8

# Create patch with only updated files
tar -czf /tmp/qwen_fixes.tar.gz \
  src/evaluation/statistical_tests.py \
  scripts/evaluation/evaluate_comprehensive.py \
  scripts/evaluation/save_generations.py

# Manual transfer via RunPod web terminal:
# 1. Upload /tmp/qwen_fixes.tar.gz to /workspace/ in RunPod
# 2. Extract: cd /workspace/Qwen3-8 && tar -xzf ../qwen_fixes.tar.gz
```

---

## Recommended Workflow (After Fixing)

**On RunPod:**

```bash
cd /workspace/Qwen3-8

# Method 1: Skip LLM-judge (FREE, FAST)
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge

# Method 2: With LLM-judge (after adding credits)
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --anthropic-key $(cat .env | grep ANTHROPIC_API_KEY | cut -d= -f2)
```

**My recommendation:** Method 1 (skip LLM-judge). You don't need it for the paper's main contributions.

---

## After Completion

```bash
# Still on RunPod - continue with other analyses
python scripts/analysis/analyze_lora_parameters.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --output output/analysis/lora_comparison

python scripts/evaluation/evaluate_novel_metrics.py \
  --results-dir output/evaluation/ \
  --output output/evaluation/novel_metrics

# Download to Mac
# (On Mac):
./sync_from_runpod.sh
```

---

## Summary

**Bug fixes:** ✅ Done  
**Should cache generations:** ✅ YES - I created `save_generations.py` for this  
**Next step:** Apply fixes to RunPod and re-run  

Let me know when you've added Anthropic credits, and I'll help you transfer the fixes and re-run!


