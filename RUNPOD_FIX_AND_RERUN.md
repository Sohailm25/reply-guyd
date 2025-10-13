# RunPod: Fix Issues & Re-run Evaluation

## 🔍 Root Cause Analysis

### Issue 1: NLTK Data Missing (CRITICAL)
- **Symptom:** All diversity metrics = 0.0
- **Cause:** `punkt` tokenizer not downloaded on RunPod
- **Impact:** Diversity analysis completely broken

### Issue 2: JSON Serialization Bug
- **Symptom:** TypeError when saving statistical_tests.json
- **Cause:** numpy bool → JSON can't serialize
- **Impact:** Script crashes before completing

### Issue 3: Anthropic Credits
- **Symptom:** API 400 errors after 76 examples
- **Cause:** Credit balance too low
- **Impact:** LLM-judge incomplete (optional)

---

## ✅ What Actually Worked

Your evaluation DID work partially:
- ✅ **10,000 generations created** (5,000 per model)
- ✅ **Pass@k computed:** Polychromic 0.722 vs Baseline 0.64 (good!)
- ✅ **Quality metrics:** ROUGE scores computed
- ❌ **Diversity metrics:** All zeros (NLTK issue)
- ❌ **LLM-judge:** Only 76/500 (credits issue)

**The expensive GPU work (generation) succeeded!**

---

## 🚀 Complete Fix (3 Commands on RunPod)

### Step 1: Download NLTK Data (30 seconds)

```bash
cd /workspace/Qwen3-8

python << 'EOF'
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
print("✓ NLTK data downloaded")
EOF
```

### Step 2: Apply Bug Fixes (30 seconds)

```bash
# Fix JSON serialization
python << 'EOF'
content = open('src/evaluation/statistical_tests.py').read()
content = content.replace("'significant': p_value < 0.05", "'significant': bool(p_value < 0.05)")
if 'import math' not in content:
    content = content.replace('import logging\n', 'import logging\nimport math\n')
open('src/evaluation/statistical_tests.py', 'w').write(content)
print("✓ JSON bug fixed")
EOF
```

### Step 3: Verify Everything Works (30 seconds)

```bash
# Test NLTK
python -c "from nltk.tokenize import word_tokenize; print('NLTK:', word_tokenize('test sentence'))"

# Test diversity metrics  
python << 'EOF'
import sys
sys.path.insert(0, '.')
from src.evaluation.diversity_metrics import compute_distinct_n
texts = ["hello world", "hello there", "goodbye world"]
result = compute_distinct_n(texts, 2)
print(f"Distinct-2 test: {result:.4f}")
if result > 0:
    print("✓ Diversity metrics WORKING!")
else:
    print("✗ Still broken - check NLTK")
    exit(1)
EOF
```

**If all 3 commands succeed, you're ready!**

---

## 🔄 Re-run Evaluation (Method 1: Skip LLM-Judge)

**Recommended - No API costs, faster**

```bash
cd /workspace/Qwen3-8

# Clean old results
rm -f output/evaluation/diversity_metrics.json output/evaluation/statistical_tests.json

# Start in tmux
tmux new -s qwen-eval

# Inside tmux:
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

**Time:** 30-60 minutes  
**Cost:** $0.50-1.00 (GPU only, no API)

---

## 📊 Monitor Progress

### Check if diversity metrics are working (IMPORTANT):

```bash
# Wait 15 minutes, then check
tail -100 eval_rerun.log | grep -i "diversity\|distinct"

# After generation completes, check the file
cat output/evaluation/diversity_metrics.json
```

**Should see:**
```json
{
  "baseline_lora": {
    "self_bleu": 0.35,     ← NOT 0.0!
    "distinct_1": 0.45,    ← NOT 0.0!
    "distinct_2": 0.62,    ← NOT 0.0!
    ...
  }
}
```

**If still all zeros:**
- STOP the evaluation (Ctrl+C)
- Run the NLTK test again
- Check for errors: `grep -i "error\|nltk" eval_rerun.log`

---

## ⚡ After Evaluation Completes

### Continue with remaining analyses:

```bash
# Step 2: LoRA Parameter Analysis (10 min)
python scripts/analysis/analyze_lora_parameters.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --output output/analysis/lora_comparison

# Step 3: Novel Metrics (5 min)
python scripts/evaluation/evaluate_novel_metrics.py \
  --results-dir output/evaluation/ \
  --output output/evaluation/novel_metrics

# Step 4: Organize Results (1 min)
mkdir -p paper/figures paper/data

cp output/evaluation/pareto_frontier.pdf paper/figures/figure3_pareto_frontier.pdf 2>/dev/null || echo "Warning: pareto_frontier.pdf not found"
cp output/analysis/lora_comparison/lora_comparison.pdf paper/figures/figure4_lora_analysis.pdf 2>/dev/null || echo "Warning: lora_comparison.pdf not found"
cp output/evaluation/novel_metrics/der_comparison.pdf paper/figures/figure5_der_comparison.pdf 2>/dev/null || echo "Warning: der_comparison.pdf not found"
cp output/evaluation/novel_metrics/collapse_points.pdf paper/figures/figure6_collapse_points.pdf 2>/dev/null || echo "Warning: collapse_points.pdf not found"

cp output/evaluation/*.json paper/data/ 2>/dev/null || true
cp output/analysis/lora_comparison/*.json paper/data/ 2>/dev/null || true
cp output/evaluation/novel_metrics/*.json paper/data/ 2>/dev/null || true

echo ""
echo "✓ Results organized in paper/"
echo ""
echo "Figures created:"
ls -lh paper/figures/*.pdf
```

---

## 📥 Download Results to Mac

```bash
# On Mac terminal
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
./sync_from_runpod.sh

# Verify
ls paper/figures/
ls paper/data/
```

---

## 🧪 Testing Checklist (Copy This!)

Run these on RunPod BEFORE evaluation:

```bash
# Test 1: NLTK
python -c "from nltk.tokenize import word_tokenize; print('✓ NLTK works')"

# Test 2: Diversity metrics
python -c "import sys; sys.path.insert(0, '.'); from src.evaluation.diversity_metrics import compute_distinct_n; print('✓ Distinct-n:', compute_distinct_n(['hello world', 'hello there'], 2))"

# Test 3: Import check
python -c "import torch, transformers, peft, anthropic, nltk; print('✓ All packages')"

# Test 4: CUDA
python -c "import torch; print('✓ CUDA:', torch.cuda.is_available())"

# Test 5: Files
ls output/experiments/*/seed_42/adapter_model.safetensors && echo "✓ Adapters found"
```

**All 5 must pass before running evaluation!**

---

## 🎯 Summary

### Files to Create on RunPod:

Just copy-paste `runpod_complete_setup.sh` content into RunPod terminal, or:

1. **Download NLTK data**
2. **Fix JSON bug**
3. **Test everything**
4. **Run evaluation**

### Expected Results:

After successful run, you'll have:
- ✅ 4 PDF figures for paper
- ✅ 6+ JSON files with real (non-zero) metrics
- ✅ Polychromic showing higher diversity than baseline
- ✅ Statistical significance tests

### Timeline:

- Setup & testing: 2 minutes
- Evaluation: 30-60 minutes (GPU)
- LoRA + Novel metrics: 15 minutes
- Download to Mac: 2 minutes
- **Total: ~1 hour**

### Cost:

- GPU: ~$0.75 (1 hour @ $0.79/hr)
- API: $0 (skipping LLM-judge)
- **Total: $0.75**

---

## 💡 Pro Tip

Save the `runpod_complete_setup.sh` script on RunPod for future use. Any time you start a new RunPod instance, just run it to set everything up automatically!

---

**Ready?** Just run the 3 fix commands on RunPod, then start the evaluation!


