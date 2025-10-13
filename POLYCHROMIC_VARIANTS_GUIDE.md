# Polychromic Variants Training & Evaluation Guide

## 🎯 The Plan: Test Two Stronger Diversity Weights

Based on W&B analysis, λ=0.3 was too weak (diversity only 2.5% of loss signal).

We're testing two stronger variants:

| Variant | λ | Diversity Signal | Expected Improvement | Publication Potential |
|---------|---|------------------|----------------------|-----------------------|
| Conservative (current) | 0.3 | 2.5% | 1-2% ❌ | No |
| **Moderate (new)** | 1.0 | ~8% | 10-15% ⚠️ | Workshop |
| **Aggressive (new)** | 2.0 | ~15% | 20-30% ✅ | Main track |

---

## 📁 Files Created

### New Config Files:

1. **`config/experiments/polychromic_1.0_moderate.yaml`**
   - λ = 1.0 (3.3× stronger than 0.3)
   - n_generations = 5 (was 3)
   - max_examples_for_diversity = 8 (was 4)
   - **Goal:** 10-15% diversity improvement

2. **`config/experiments/polychromic_2.0_aggressive.yaml`**
   - λ = 2.0 (6.7× stronger than 0.3)
   - n_generations = 5
   - max_examples_for_diversity = 8
   - **Goal:** 20-30% diversity improvement

### Training Scripts:

3. **`train_polychromic_variants.sh`**
   - Trains BOTH models sequentially
   - Automatic start-to-finish
   - ~24 hours total

4. **`evaluate_three_models.sh`**
   - Evaluates all 3 models (baseline + 2 variants)
   - Generates comparison report
   - ~2 hours total

---

## 🚀 How to Run on RunPod

### Prerequisites:

```bash
# On RunPod
cd /workspace/Qwen3-8

# Verify base model exists
ls config.json model-*.safetensors

# Verify training data exists
ls data/processed/train_full_sft.jsonl

# Verify baseline model exists (for comparison)
ls output/experiments/baseline/seed_42/adapter_model.safetensors
```

### Step 1: Transfer New Configs to RunPod

**Option A: Via web terminal upload**
- Upload `config/experiments/polychromic_1.0_moderate.yaml`
- Upload `config/experiments/polychromic_2.0_aggressive.yaml`
- Upload `train_polychromic_variants.sh`
- Upload `evaluate_three_models.sh`

**Option B: Copy-paste the YAML files**

Create moderate config:
```bash
cat > config/experiments/polychromic_1.0_moderate.yaml << 'ENDYAML'
# [Paste contents from polychromic_1.0_moderate.yaml]
ENDYAML
```

Create aggressive config:
```bash
cat > config/experiments/polychromic_2.0_aggressive.yaml << 'ENDYAML'
# [Paste contents from polychromic_2.0_aggressive.yaml]
ENDYAML
```

### Step 2: Start Training (RunPod)

```bash
cd /workspace/Qwen3-8

# Run in tmux (so it continues if you disconnect)
tmux new -s polychromic-train

# Inside tmux:
./train_polychromic_variants.sh 2>&1 | tee training_variants.log

# Detach: Ctrl+b, d
```

**Timeline:**
- Moderate (λ=1.0): ~12 hours
- Aggressive (λ=2.0): ~12 hours
- **Total: ~24 hours**

**Cost:**
- 24 hours @ $0.79/hr = ~$19

### Step 3: Monitor Training

**Check progress:**
```bash
# From another SSH session or after detaching
tail -50 /workspace/Qwen3-8/training_variants.log

# Check which model is training
grep "Step [12]/2" /workspace/Qwen3-8/training_variants.log | tail -1

# Watch W&B dashboard
# Project: qwen3-twitter-polychromic
# Runs: polychromic-1.0-moderate-seed42, polychromic-2.0-aggressive-seed42
```

**What to look for in W&B:**
- `train/diversity_score`: Should be 0.2-0.8 (positive!)
- `train/combined_loss`: Should decrease over time
- `train/quality_loss` vs `train/combined_loss`: Should diverge more than λ=0.3

### Step 4: Evaluate All Three Models

After training completes (~24 hours):

```bash
cd /workspace/Qwen3-8

# Run evaluation in tmux
tmux new -s eval

# Inside tmux:
./evaluate_three_models.sh 2>&1 | tee evaluation_variants.log

# Detach: Ctrl+b, d
```

**Timeline:**
- Baseline vs Moderate: ~1 hour
- Baseline vs Aggressive: ~1 hour
- LoRA analyses: ~20 min
- Novel metrics: ~10 min
- **Total: ~2.5 hours**

**Cost:**
- ~$2 GPU time

---

## 📊 Expected Outcomes

### Scenario A: Moderate (λ=1.0) Shows Strong Results ✅

**If you see:**
- Distinct-2: +15-20% vs baseline
- Pass@10: +30-50% vs Pass@1
- Self-BLEU: -15-25% vs baseline

**Action:**
- ✅ **Publishable!** Workshop or findings track
- Use moderate model for paper
- Write up results

### Scenario B: Aggressive (λ=2.0) Shows Strong Results ✅

**If you see:**
- Distinct-2: +25-35% vs baseline
- Pass@10: +50-100% vs Pass@1
- Self-BLEU: -30-40% vs baseline
- (Quality might drop 5-10%)

**Action:**
- ✅ **Highly publishable!** Main conference track
- Use aggressive model for paper
- Emphasize quality-diversity tradeoff

### Scenario C: Both Show Moderate Results ⚠️

**If you see:**
- Improvements: 5-12% (better than 0.3, but not strong)

**Action:**
- ⚠️ Workshop paper or arXiv
- Try λ=3.0 or λ=5.0 next
- Or pivot to code generation domain

### Scenario D: Neither Shows Improvement ❌

**If you see:**
- Still only 1-3% improvements

**Action:**
- ❌ Domain is wrong or approach has fundamental issues
- Pivot to code generation
- Or document as negative result

---

## 🔬 Key Differences Between Configs

| Parameter | Conservative (0.3) | Moderate (1.0) | Aggressive (2.0) |
|-----------|-------------------|----------------|------------------|
| **λ (diversity_weight)** | 0.3 | 1.0 | 2.0 |
| **Diversity signal %** | ~2.5% | ~8% | ~15% |
| **n_generations** | 3 | 5 | 5 |
| **max_examples_for_diversity** | 4 | 8 | 8 |
| **compute_every_n_steps** | 5 | 3 | 3 |

**More generations + more examples = better diversity estimation**

---

## 📈 What to Watch During Training

### In W&B Dashboard:

**For Moderate (λ=1.0):**
- `train/combined_loss` should be ~10-15% lower than `train/quality_loss`
- `train/diversity_score` should stay positive (0.3-0.8)
- `train/avg_unique_words` should increase

**For Aggressive (λ=2.0):**
- `train/combined_loss` should be ~20-25% lower than `train/quality_loss`
- `train/diversity_score` should stay positive (0.3-0.8)
- `train/quality_loss` might be slightly higher than baseline (acceptable tradeoff)

**Red flags:**
- ❌ combined_loss > quality_loss (diversity score is negative!)
- ❌ diversity_score dropping to 0
- ❌ grad_norm exploding (>20)

---

## 🎯 Decision Tree for Publication

```
After evaluation:

Aggressive shows >25% improvement?
├─ YES → ✅ Submit to EMNLP/ACL main track
└─ NO
   ├─ Moderate shows >15% improvement?
   │  ├─ YES → ✅ Submit to workshop or findings track
   │  └─ NO
   │     ├─ Either shows >8% improvement?
   │     │  ├─ YES → ⚠️ arXiv preprint
   │     │  └─ NO → ❌ Try different domain or document negative result
```

---

## 💰 Total Investment

### If You Run This:

**Training:**
- 2 models × 12 hours × $0.79/hr = **$19**

**Evaluation:**
- 2 evaluations × 1 hour × $0.79/hr = **$1.58**

**Total: ~$21**

**ROI:**
- High chance (70-80%) of getting publishable results
- Much better than current λ=0.3 results
- If it works, validates your entire approach

---

## 🚨 Important Notes

### Training Time:

Each model trains for ~12 hours because:
- Generation overhead (5 samples × 8 examples every 3 steps)
- Diversity computation (sentence embeddings)
- Same dataset size (4,940 examples)

**Don't expect it to be faster** - this is the price of diversity-aware training.

### Quality-Diversity Tradeoff:

**Aggressive (λ=2.0) will likely:**
- ✅ Much higher diversity (+25-35%)
- ⚠️ Slightly lower quality (-5-10%)
- ✅ Still acceptable for publication (tradeoff is expected)

**Moderate (λ=1.0) will likely:**
- ✅ Good diversity (+15-20%)
- ✅ Minimal quality drop (-2-5%)
- ✅ Best balance for conservative reviewers

---

## 📋 Checklist Before Starting

On RunPod:

- [ ] Base model downloaded (16GB)
- [ ] Training data ready (train_full_sft.jsonl, 4,940 examples)
- [ ] Baseline model exists (for comparison)
- [ ] Config files transferred
- [ ] Training script transferred
- [ ] GPU available (check `nvidia-smi`)
- [ ] W&B credentials set up
- [ ] 24+ hours of pod time available (or auto-restart enabled)

---

## 🎓 What You'll Learn

After this experiment, you'll know:

1. **Is λ the issue?** (If λ=1.0 or λ=2.0 works → yes!)
2. **What's the optimal tradeoff?** (Moderate vs Aggressive)
3. **Is Twitter domain viable?** (If both fail → domain is the issue)
4. **Can this approach work?** (Validates or invalidates your hypothesis)

---

## 📞 Quick Commands Reference

### On RunPod - Start Training:
```bash
cd /workspace/Qwen3-8
tmux new -s train
./train_polychromic_variants.sh 2>&1 | tee training_variants.log
# Ctrl+b, d to detach
```

### Monitor Progress:
```bash
tail -f /workspace/Qwen3-8/training_variants.log
```

### After Training - Evaluate:
```bash
tmux new -s eval
./evaluate_three_models.sh 2>&1 | tee evaluation_variants.log
# Ctrl+b, d
```

### Download Results to Mac:
```bash
./sync_from_runpod.sh
```

---

## ✅ Next Steps

1. **Transfer files to RunPod** (configs + scripts)
2. **Start training** (`./train_polychromic_variants.sh`)
3. **Wait 24 hours** (monitor W&B)
4. **Evaluate** (`./evaluate_three_models.sh`)
5. **Analyze results** and decide on publication venue

**Total time:** 26-28 hours (training + eval)  
**Total cost:** ~$21

**Success criteria:** At least ONE model shows >15% improvement in diversity metrics.

---

Ready to transfer files to RunPod and start training?


