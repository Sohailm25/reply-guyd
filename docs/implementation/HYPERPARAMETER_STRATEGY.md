# Hyperparameter Strategy for Small Datasets

## 🎯 **Decision: Start Standard, Adjust if Needed**

### **Recommended Approach**

**Phase 1: Run with current configs (rank=16, λ=0.3)**
- Learn actual overfitting patterns
- Understand quality-diversity tradeoff
- Get baseline performance
- Cost: ~$12

**Phase 2: Evaluate results**
- Check train/eval loss gap
- Examine W&B curves
- Review sample outputs

**Phase 3: Decide next step**
- If working well → Multiple seeds
- If overfitting → Conservative configs
- If underfitting → More aggressive configs

---

## 📊 **Current vs Conservative Configs**

### Standard Configuration (Use First)

**File:** `config/experiments/baseline.yaml` & `polychromic_0.3.yaml`

```yaml
lora:
  rank: 16
  alpha: 16
  dropout: 0.1

polychromic:
  diversity_weight: 0.3
  n_generations: 3

training:
  learning_rate: 2.0e-4
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

**Rationale:**
- Research-informed (QLoRA paper recommendations)
- Gives us baseline to learn from
- Early stopping protects against severe overfitting
- Statistical comparison is cleaner

---

### Conservative Configuration (Use If Overfitting)

**File:** `config/experiments/baseline_conservative.yaml` & `polychromic_conservative.yaml`

```yaml
lora:
  rank: 8              # ↓ from 16 - fewer parameters
  alpha: 16            # Keep 2x rank ratio
  dropout: 0.15        # ↑ from 0.1 - more regularization

polychromic:
  diversity_weight: 0.15  # ↓ from 0.3 - prioritize quality
  n_generations: 2        # ↓ from 3 - faster, less diversity pressure

training:
  learning_rate: 1.0e-4   # ↓ from 2e-4 - safer convergence
  train_split: 0.7        # ↓ from 0.8 - more eval data
  val_split: 0.15         # ↑ from 0.1 - better early stopping signal
  test_split: 0.15        # ↑ from 0.1 - more robust testing
```

**Use when:**
- Train/eval loss gap > 0.4
- Eval loss starts increasing after epoch 1
- Model memorizing training examples
- Validation perplexity increasing

---

## 🚨 **Overfitting Detection Guide**

### **Signs to Watch in W&B:**

**🟢 Healthy Training:**
```
Epoch 1: train_loss=0.8 → eval_loss=0.85
Epoch 2: train_loss=0.6 → eval_loss=0.65
Gap: 0.05 ✓ Good!
```

**🟡 Mild Overfitting (Acceptable):**
```
Epoch 1: train_loss=0.8 → eval_loss=0.9
Epoch 2: train_loss=0.5 → eval_loss=0.7
Gap: 0.2 ⚠️ Borderline, early stopping will help
```

**🔴 Severe Overfitting (Use Conservative):**
```
Epoch 1: train_loss=0.8 → eval_loss=1.0
Epoch 2: train_loss=0.3 → eval_loss=1.2 ↑
Gap: 0.9 ❌ Stop and use conservative configs
```

### **What to Check:**

1. **Training Curves (W&B)**
   - `train/loss` should decrease smoothly
   - `eval/loss` should track close to train
   - Gap < 0.3 is healthy

2. **Sample Outputs**
   - Are they generic/repetitive? → Overfitting
   - Do they vary appropriately? → Good
   - Are they on-topic? → Check quality

3. **Perplexity**
   - Should be reasonable (5-20)
   - If skyrocketing → Problem

---

## 🎓 **When to Use Each Configuration**

### **Scenario 1: Unknown Dataset Size (Current)**
→ **Use Standard configs first**
- We don't know final size yet (collection ongoing)
- Standard gives us learning baseline
- Can adjust after seeing results

### **Scenario 2: Final Dataset < 800 pairs**
→ **Consider Conservative immediately**
- Very small dataset
- Start with rank=8
- Lower learning rate

### **Scenario 3: Final Dataset 800-1,500 pairs**
→ **Use Standard, monitor closely**
- Should work with rank=16
- Early stopping will protect
- Adjust if gap > 0.3

### **Scenario 4: Final Dataset > 1,500 pairs**
→ **Standard is perfect**
- Enough data for rank=16
- May even consider rank=32 in ablations

---

## 📝 **Decision Matrix**

| Observed Behavior | Action |
|-------------------|--------|
| Train/eval gap < 0.2 | ✅ Continue with standard, try multiple seeds |
| Train/eval gap 0.2-0.4 | ⚠️ Monitor, early stopping will handle it |
| Train/eval gap > 0.4 | ❌ Stop, switch to conservative configs |
| Eval loss increasing | ❌ Overfitting, use conservative |
| Good diversity, poor quality | 🔄 Reduce diversity_weight to 0.15 |
| Good quality, poor diversity | 🔄 Keep or increase diversity_weight |
| Training too slow | ⏩ Reduce n_generations from 3→2 |

---

## 💡 **Recommended Workflow**

### **Week 1: Initial Training**

```bash
# After data curation complete

# 1. Train baseline (standard config)
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data.jsonl

# Monitor in W&B for 1-2 hours
# Check: Is train/eval gap reasonable?
```

### **Decision Point 1: After Baseline Epoch 1**

**If train/eval gap < 0.3:**
```bash
# Continue baseline, start polychromic (standard)
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data.jsonl
```

**If train/eval gap > 0.4:**
```bash
# Stop baseline, restart with conservative
python scripts/training/train_model.py \
  --config config/experiments/baseline_conservative.yaml \
  --data data/processed/training_data.jsonl
```

### **Week 2: Analysis & Iteration**

After both models trained:
1. Run comprehensive evaluation
2. Analyze results
3. Decide on ablations or multiple seeds

---

## 🔬 **Research Perspective**

### **Why Standard First is Better for Research:**

1. **Cleaner Comparison**
   - Baseline vs Polychromic with same hyperparams
   - Isolates effect of diversity objective
   - More interpretable results

2. **Learning Opportunity**
   - See actual overfitting patterns
   - Understand quality-diversity tradeoff
   - Inform future work

3. **Publishable Either Way**
   - Good results → Great!
   - Overfitting → "We found rank=16 too high for N<1000"
   - Both are valid contributions

4. **Cost is Minimal**
   - $12 for first run
   - $12 for conservative run if needed
   - Total $24 to explore both strategies

---

## ✅ **Immediate Action Items**

### **Now (Before Training):**
- [ ] Wait for data collection to complete
- [ ] Manual curation to 800-1,200 pairs
- [ ] Note actual dataset size

### **Decision Point:**
- [ ] If dataset < 800 pairs → Use conservative configs
- [ ] If dataset 800-1,500 pairs → Use standard configs first
- [ ] If dataset > 1,500 pairs → Definitely use standard

### **During Training:**
- [ ] Monitor W&B dashboard closely
- [ ] Check train/eval gap after epoch 1
- [ ] Be ready to stop and switch if severe overfitting

### **After First Run:**
- [ ] Evaluate results
- [ ] Decide: continue with standard, switch to conservative, or proceed to multiple seeds
- [ ] Document findings

---

## 📚 **Configuration Files Reference**

**Standard (Use First):**
- `config/experiments/baseline.yaml`
- `config/experiments/polychromic_0.3.yaml`

**Conservative (Use If Overfitting):**
- `config/experiments/baseline_conservative.yaml`
- `config/experiments/polychromic_conservative.yaml`

**All configs ready to use!**

---

## 🎯 **Final Recommendation**

**Start with standard configs (rank=16, λ=0.3).**

**Reasons:**
1. ✅ Don't know final dataset size yet
2. ✅ Research-informed baseline
3. ✅ Early stopping protects us
4. ✅ Learn more from first run
5. ✅ Cost is minimal ($12)
6. ✅ Conservative configs ready if needed

**Switch to conservative ONLY if:**
- Train/eval gap > 0.4
- Eval loss increases
- Severe overfitting observed

**You're set up for success either way!** 🚀

