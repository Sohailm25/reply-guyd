# 🚀 Quick Start: Evaluation Improvements

**What changed:** Temperature control + Multi-seed training now implemented

---

## ⚡ Quick Commands

### **Train All Seeds (Recommended)**

```bash
# Train baseline with all 3 seeds (42, 123, 456)
bash scripts/training/train_all_seeds.sh baseline

# Train polychromic with all 3 seeds
bash scripts/training/train_all_seeds.sh polychromic_0.3
```

### **Evaluate (Automatic Temperature Handling)**

```bash
# Evaluation now automatically uses:
#   - temp=0 (deterministic) for quality metrics
#   - temp=0.7 (diverse) for Pass@k

python scripts/evaluation/evaluate_comprehensive.py \
    --baseline-lora output/experiments/baseline/seed_42 \
    --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
    --test-data data/processed/test_data.jsonl \
    --output output/evaluation/seed_42/
```

### **Aggregate Results Across Seeds**

```bash
# Compute mean ± std for all metrics
python scripts/analysis/aggregate_seeds.py \
    --results-dir output/evaluation/ \
    --seeds 42 123 456 \
    --output output/evaluation/aggregated.json \
    --latex-output output/evaluation/table.tex
```

---

## 📊 What's Automatic

**You DON'T need to:**
- Manually specify temperature (handled automatically)
- Generate twice (script does it for you)
- Choose which mode for which metric (automatic)

**The script automatically:**
1. Generates with temp=0 (deterministic) for ROUGE, BERTScore
2. Generates with temp=0.7 (diverse) for Pass@k, diversity
3. Uses correct generations for each metric type

---

## 💰 Cost & Time (3 Seeds)

| Experiment | Time | Cost |
|------------|------|------|
| Baseline × 3 | 12 hrs | $9 |
| Polychromic × 3 | 36 hrs | $27 |
| **Total** | **48 hrs** | **$36** |

---

## 📝 Reporting in Paper

**Methods:**
```
We report metrics as mean ± standard deviation across 3 random 
seeds (42, 123, 456). Quality metrics use greedy decoding 
(temperature=0) for reproducibility. Pass@k uses temperature 
sampling (temperature=0.7) for diversity.
```

**Results:**
```
Pass@10: 0.78 ± 0.03 (mean ± std over 3 seeds)
```

---

## ✅ What's Different

### **Before:**
- Single seed (no variance)
- temp=0.7 for everything (non-reproducible quality metrics)

### **After:**
- 3 seeds (can report mean ± std)
- temp=0 for quality metrics (reproducible)
- temp=0.7 for Pass@k (diverse, correct)

---

## 📚 Full Documentation

See `docs/implementation/IMPLEMENTATION_COMPLETE_EVAL_IMPROVEMENTS.md` for complete details.

---

**Status:** ✅ Ready to use!

