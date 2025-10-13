# Evaluation Diagnostic Guide

## Root Cause: NLTK Data Missing on RunPod

### What Went Wrong

Looking at your results:
```json
"diversity_metrics.json": {
  "self_bleu": 0.0,
  "distinct_1": 0.0,
  "distinct_2": 0.0,
  ...all zeros...
}
```

But your Pass@k results are GOOD:
```json
"passk_results.json": {
  "baseline_lora": {"1": 0.64, "5": 0.992, "10": 1.0},
  "polychromic_lora": {"1": 0.722, "5": 0.986, "10": 0.998}
}
```

**This means:**
- ✅ Models loaded correctly
- ✅ Generation worked (10,000 replies generated)
- ✅ Pass@k computed correctly
- ❌ Diversity metrics failed silently → returned all 0.0

**Root cause:** NLTK's `punkt` tokenizer data wasn't downloaded on RunPod

### Why This Happens

The diversity metrics code uses:
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)  # ← Requires punkt data
```

When `punkt` data is missing:
- `word_tokenize()` throws `LookupError`
- Code catches exception and returns `0.0`
- No warning shown (silent failure)
- All diversity metrics become 0.0

---

## How to Test Before Re-running (CRITICAL)

### On RunPod, run this FIRST:

```bash
cd /workspace/Qwen3-8

# Quick test
python -c "from nltk.tokenize import word_tokenize; print('✓ NLTK works:',  word_tokenize('test sentence'))"
```

**If you see error:** NLTK data missing
**If you see tokens:** NLTK working

### Better: Use the Pre-flight Check

I've created a comprehensive test script:

```bash
cd /workspace/Qwen3-8
python preflight_check.py
```

This will:
1. Check all required files
2. Check all Python packages
3. **Test NLTK punkt tokenizer**
4. **Actually compute diversity metrics on test data**
5. Check CUDA/GPU

**ONLY proceed with evaluation if all checks pass!**

---

## How to Fix

### Fix Option 1: Auto-download (Easiest)

I've updated `src/evaluation/diversity_metrics.py` to auto-download NLTK data.

**Just transfer the updated file to RunPod and it will fix itself.**

### Fix Option 2: Manual download (RunPod)

```bash
cd /workspace/Qwen3-8

# Download NLTK data
python << 'EOF'
import nltk
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('punkt_tab')  # For newer NLTK versions
nltk.download('stopwords')
print("✓ Done!")
EOF

# Test it works
python -c "from nltk.tokenize import word_tokenize; print(word_tokenize('test'))"
# Should output: ['test']
```

### Fix Option 3: Use the test script

```bash
python test_runpod_nltk.py
```

This will diagnose the issue and download what's missing.

---

## Complete Re-run Checklist

### Before Starting Evaluation:

```bash
cd /workspace/Qwen3-8

# 1. Apply bug fixes
python << 'EOF'
import re
content = open('src/evaluation/statistical_tests.py').read()
content = content.replace("'significant': p_value < 0.05", "'significant': bool(p_value < 0.05)")
if 'import math' not in content:
    content = content.replace('import logging\n', 'import logging\nimport math\n')
open('src/evaluation/statistical_tests.py', 'w').write(content)
print("✓ Statistical tests bug fixed")
EOF

# 2. Download NLTK data
python << 'EOF'
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print("✓ NLTK data downloaded")
EOF

# 3. Run pre-flight check
python preflight_check.py

# If all pass ✓, proceed to evaluation
```

### Expected Pre-flight Output:

```
✓ PASS: Files
✓ PASS: Python Packages  
✓ PASS: NLTK Data
✓ PASS: Diversity Metrics
✓ PASS: CUDA/GPU

✓ ALL CHECKS PASSED!
```

---

## Re-run Evaluation (After Fixes)

```bash
# In tmux
tmux new -s qwen-eval

cd /workspace/Qwen3-8

# Clean old corrupt results
rm output/evaluation/diversity_metrics.json
rm output/evaluation/statistical_tests.json

# Re-run evaluation
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --skip-llm-judge 2>&1 | tee eval_rerun.log

# Detach: Ctrl+b, d
```

---

## Monitoring Re-run

### Check if diversity metrics are working:

```bash
# After 10 minutes, check diversity metrics file
cat output/evaluation/diversity_metrics.json

# Should see NON-ZERO values like:
# "distinct_2": 0.65  (NOT 0.0!)
# "self_bleu": 0.35   (NOT 0.0!)
```

### Check progress:

```bash
# See latest progress
tail -50 eval_rerun.log | grep -E "Progress:|Step|diversity|ERROR"

# Watch live
tail -f eval_rerun.log
```

---

## What to Look For in Logs

### ✓ GOOD Signs:

```
✓ NLTK punkt data downloaded
Computing diversity metrics...
Self-BLEU: 0.3245
Distinct-2: 0.6789
```

### ✗ BAD Signs:

```
LookupError: punkt
Resource punkt not found
All diversity metrics: 0.0
```

---

## Files to Transfer to RunPod

I've created **`~/Desktop/qwen_fixes_complete.tar.gz`** (13KB) with:

1. ✅ `src/evaluation/statistical_tests.py` - JSON bug fixed
2. ✅ `src/evaluation/diversity_metrics.py` - Auto-downloads NLTK data
3. ✅ `scripts/evaluation/evaluate_comprehensive.py` - CPU support
4. ✅ `preflight_check.py` - Comprehensive pre-flight test
5. ✅ `test_runpod_nltk.py` - NLTK-specific test

### Transfer via RunPod Web Terminal:

1. Upload `qwen_fixes_complete.tar.gz` to `/workspace/` in RunPod
2. Extract:
```bash
cd /workspace/Qwen3-8
tar -xzf ../qwen_fixes_complete.tar.gz
chmod +x preflight_check.py test_runpod_nltk.py
```

3. Run pre-flight:
```bash
python preflight_check.py
```

4. If all pass, run evaluation

---

## Summary: Two Critical Issues

### Issue 1: NLTK Data Missing ⚠️
- **Symptom:** All diversity metrics = 0.0
- **Cause:** `punkt` tokenizer not downloaded
- **Fix:** `nltk.download('punkt')`
- **Prevention:** Run `preflight_check.py` first
- **Auto-fix:** Updated `diversity_metrics.py` auto-downloads

### Issue 2: JSON Serialization Bug ⚠️
- **Symptom:** TypeError when saving results
- **Cause:** numpy bool not JSON-serializable
- **Fix:** Convert to Python bool
- **Prevention:** Use `sanitize_for_json()` function

---

## Simple Test on RunPod (Right Now)

Before re-running everything, test if NLTK works:

```bash
# On RunPod
cd /workspace/Qwen3-8

python << 'EOF'
# Test 1: NLTK
from nltk.tokenize import word_tokenize
print("Test 1:", word_tokenize("hello world"))

# Test 2: Diversity metrics
import sys
sys.path.insert(0, '.')
from src.evaluation.diversity_metrics import compute_distinct_n

texts = ["hello world", "hello there", "goodbye world"]
distinct2 = compute_distinct_n(texts, n=2)
print(f"Test 2: Distinct-2 = {distinct2}")

if distinct2 > 0:
    print("✓ Diversity metrics WORKING!")
else:
    print("✗ Diversity metrics BROKEN - all zeros")
EOF
```

**Expected output:**
```
Test 1: ['hello', 'world']
Test 2: Distinct-2 = 0.83333...
✓ Diversity metrics WORKING!
```

**If you see:**
```
LookupError: punkt
```

**Then run:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Next Steps

1. **Test NLTK on RunPod now** (30 seconds)
2. **If broken, download punkt** (1 minute)
3. **Run preflight_check.py** (1 minute)
4. **Re-run evaluation** (30-60 minutes)

Want me to help you create a simple one-liner command that does all the setup and testing automatically?


