# Quick Fix for RunPod (Copy-Paste These Commands)

## 🎯 The Problem

Your evaluation ran but:
- ❌ **Diversity metrics all 0.0** ← NLTK data missing
- ❌ **Script crashed** ← JSON bug
- ⚠️ **LLM-judge incomplete** ← API credits (optional)

**But generation worked!** Pass@k shows 0.722 vs 0.64 (polychromic winning!)

---

## ⚡ The Fix (3 Minutes)

### On RunPod, copy-paste this entire block:

```bash
cd /workspace/Qwen3-8

echo "=== Fixing NLTK Issue ==="
python << 'EOF'
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except: pass

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
print("✓ NLTK data downloaded")
EOF

echo ""
echo "=== Fixing JSON Bug ==="
python << 'EOF'
content = open('src/evaluation/statistical_tests.py').read()
content = content.replace("'significant': p_value < 0.05", "'significant': bool(p_value < 0.05)")
open('src/evaluation/statistical_tests.py', 'w').write(content)
print("✓ JSON bug fixed")
EOF

echo ""
echo "=== Testing Fixes ==="
python << 'EOF'
import sys
sys.path.insert(0, '.')

# Test NLTK
from nltk.tokenize import word_tokenize
tokens = word_tokenize("test sentence")
print(f"NLTK test: {tokens}")

# Test diversity
from src.evaluation.diversity_metrics import compute_distinct_n
texts = ["hello world", "hello there", "goodbye world"]
distinct = compute_distinct_n(texts, 2)
print(f"Diversity test: Distinct-2 = {distinct:.4f}")

if distinct > 0:
    print("✓✓✓ ALL FIXES WORKING! ✓✓✓")
else:
    print("✗✗✗ STILL BROKEN ✗✗✗")
    exit(1)
EOF

echo ""
echo "========================================="
echo "✓ READY TO RUN EVALUATION!"
echo "========================================="
```

**If you see `✓✓✓ ALL FIXES WORKING!` → proceed to re-run**

---

## 🏃 Re-run Evaluation

```bash
# Still on RunPod
cd /workspace/Qwen3-8

# Clean old broken results
rm -f output/evaluation/diversity_metrics.json
rm -f output/evaluation/statistical_tests.json

# Run in tmux
tmux new -s eval

# Inside tmux, run this:
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge 2>&1 | tee eval.log

# Detach: Ctrl+b, then d
```

---

## 👀 Monitor (While It Runs)

### After 15 minutes, check if diversity is working:

```bash
# Check diversity metrics
cat output/evaluation/diversity_metrics.json
```

**Should see:**
```json
{
  "baseline_lora": {
    "self_bleu": 0.35,    ← NON-ZERO!
    "distinct_2": 0.62,   ← NON-ZERO!
    ...
  },
  "polychromic_lora": {
    "self_bleu": 0.28,    ← Lower than baseline (more diverse!)
    "distinct_2": 0.75,   ← Higher than baseline (more diverse!)
    ...
  }
}
```

**If still zeros:**
- Attach to tmux: `tmux attach -t eval`
- Stop it (Ctrl+C)
- Check errors: `grep -i "error\|nltk" eval.log`

### Monitor progress:

```bash
# From another SSH session (or detached from tmux)
tail -f /workspace/Qwen3-8/eval.log
```

---

## ✅ After Completion (~1 hour)

### Check results:

```bash
cd /workspace/Qwen3-8

# Verify diversity metrics are NOT zero
cat output/evaluation/diversity_metrics.json | grep distinct_2

# Should show something like:
# "distinct_2": 0.6234,  (baseline)
# "distinct_2": 0.7456,  (polychromic)
```

### Run remaining analyses:

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

# Organize
mkdir -p paper/figures paper/data
cp output/evaluation/pareto_frontier.pdf paper/figures/figure3_pareto_frontier.pdf
cp output/analysis/lora_comparison/lora_comparison.pdf paper/figures/figure4_lora_analysis.pdf
cp output/evaluation/novel_metrics/der_comparison.pdf paper/figures/figure5_der_comparison.pdf
cp output/evaluation/novel_metrics/collapse_points.pdf paper/figures/figure6_collapse_points.pdf
cp output/evaluation/*.json output/analysis/lora_comparison/*.json output/evaluation/novel_metrics/*.json paper/data/

# Verify
ls paper/figures/*.pdf
# Should show 4 PDF files
```

---

## 💾 Download to Mac

```bash
# On Mac
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8

# If sync_from_runpod.sh doesn't work, use manual download via web terminal
# Or create new tar on RunPod:
# tar -czf ~/results.tar.gz paper/

# Then download via web interface
```

---

## 🎓 What You Learned

### Problem Detection:

✅ **Check diversity_metrics.json first** - if all 0.0, something is wrong  
✅ **Check logs for NLTK errors** - `LookupError: punkt`  
✅ **Run pre-flight tests** - before long evaluations

### Prevention:

✅ **Always run preflight_check.py** before expensive evaluations  
✅ **Test with small sample first** - `--max-examples 10` to verify setup  
✅ **Check output files incrementally** - don't wait till end

---

## Files Created

- `~/Desktop/qwen_evaluation_fixed.tar.gz` (18KB) - Complete fix package
- `preflight_check.py` - Comprehensive pre-run test
- `test_runpod_nltk.py` - NLTK-specific test
- `runpod_complete_setup.sh` - All-in-one setup
- `DIAGNOSTIC_GUIDE.md` - Detailed diagnostics
- `RUNPOD_FIX_AND_RERUN.md` - Complete instructions
- `QUICK_FIX_RUNPOD.md` - This file

---

## TL;DR

**On RunPod:**

```bash
# Fix everything (3 commands)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
python -c "c=open('src/evaluation/statistical_tests.py').read(); c=c.replace(\"'significant': p_value < 0.05\",\"'significant': bool(p_value < 0.05)\"); open('src/evaluation/statistical_tests.py','w').write(c); print('✓')"
python -c "from nltk.tokenize import word_tokenize; print('✓ Works:', word_tokenize('test'))"

# Re-run evaluation (in tmux)
tmux new -s eval
cd /workspace/Qwen3-8 && python scripts/evaluation/evaluate_comprehensive.py --baseline-lora output/experiments/baseline/seed_42 --polychromic-lora output/experiments/polychromic_0.3/seed_42 --test-data data/processed/test_data.jsonl --output output/evaluation/ --n-generations 10 --max-examples 500 --skip-llm-judge 2>&1 | tee eval.log
# Ctrl+b, d to detach

# Check after 15min
cat output/evaluation/diversity_metrics.json
# Should NOT be all zeros!
```

Done! 🎉


