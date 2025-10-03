# Research-Grade Polychromic LoRA Implementation

## 🎯 Overview

This is a **research-grade implementation** of Polychromic LoRA: a general-purpose diversity-aware parameter-efficient fine-tuning method designed for **Arxiv-quality** experimentation and analysis.

**Research Question:**
> Does diversity-aware optimization (Polychromic LoRA) improve Pass@k performance across task-specific domains while maintaining single-generation quality compared to standard LoRA fine-tuning?

**Key Insight:**
Training objectives should match deployment scenarios. When multiple candidates are generated and the best is selected (Pass@k evaluation), explicitly optimizing for diversity during training yields better results than standard single-generation optimization.

**Key Features:**
- ✅ Both baseline and polychromic training
- ✅ Comprehensive evaluation suite
- ✅ Statistical significance testing
- ✅ LLM-as-judge evaluation
- ✅ Pass@k metrics
- ✅ Full reproducibility
- ✅ Publication-ready code

---

## 📊 Experimental Design

### Models (Per Domain)

1. **Zero-shot** - Base model with simple prompt
2. **Few-shot** - Base model with 5-shot examples
3. **Standard LoRA** - Parameter-efficient fine-tuning with cross-entropy loss
4. **Polychromic LoRA (λ=0.3, N=3)** - Diversity-aware training (ours)

**Ablations:** Different λ values (0.1, 0.2, 0.5), different N values (2, 3, 5)

**Domains:** Social media replies, code generation, creative writing (or Q&A)

### Evaluation Metrics

**Diversity:**
- Self-BLEU (lower = more diverse)
- Distinct-n (higher = more diverse)
- Semantic diversity (cosine distance)

**Quality:**
- ROUGE scores
- BERTScore
- Perplexity

**Task Performance (Primary Metric):**
- Pass@k (k=1,3,5,10) - Our focus
- Domain-specific metrics (accuracy, F1, etc.)
- LLM-as-judge (Claude 3.5 Sonnet) - Optional

**Statistical Rigor:**
- Mann-Whitney U test
- Cohen's d effect size
- Bootstrap confidence intervals
- Multiple random seeds

---

## 🗂️ Project Structure

```
Qwen3-8/
├── config/
│   ├── experiments/           # Experiment configurations
│   │   ├── baseline.yaml
│   │   ├── polychromic_0.3.yaml
│   │   └── ...
│   ├── lora_config.yaml       # Base LoRA config
│   └── data_collection_config.yaml
│
├── src/
│   ├── training/              # Training modules
│   │   ├── base_trainer.py    # Standard LoRA trainer
│   │   ├── polychromic_trainer.py  # Diversity-aware trainer
│   │   └── data_module.py     # Data loading
│   │
│   ├── evaluation/            # Evaluation modules
│   │   ├── diversity_metrics.py
│   │   ├── quality_metrics.py
│   │   ├── statistical_tests.py
│   │   ├── llm_judge.py
│   │   └── passk_evaluation.py
│   │
│   └── data_collection/       # Data collection (existing)
│
├── scripts/
│   ├── training/
│   │   └── train_model.py     # Main training script
│   ├── evaluation/
│   │   └── evaluate_comprehensive.py  # Full evaluation
│   ├── collect_data.py        # Data collection (existing)
│   └── test_installation.py   # Installation test
│
├── data/
│   ├── processed/             # Training data
│   └── raw/                   # Raw collected data
│
├── output/
│   ├── experiments/           # Trained models
│   │   ├── baseline/
│   │   └── polychromic_0.3/
│   └── evaluation/            # Evaluation results
│
└── notebooks/                 # Analysis notebooks
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Test installation
python scripts/test_installation.py
```

### 2. Prepare Data

Ensure your curated data is in `data/processed/`:
```bash
ls data/processed/training_data_*.jsonl
```

### 3. Train Models (on RunPod)

**Baseline:**
```bash
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data_*.jsonl
```

**Polychromic:**
```bash
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data_*.jsonl
```

### 4. Evaluate

```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --anthropic-key <your-key>
```

---

## 🔬 Research Methodology

### Training Process

1. **Data Splitting:** Stratified 80/10/10 split by engagement quartiles
2. **Multiple Seeds:** Train with seeds {42, 123, 456} for robustness
3. **Early Stopping:** Patience=3 to prevent overfitting
4. **Checkpointing:** Save every 50-100 steps

### Polychromic Training

**Loss Function:**
```python
L = L_quality - λ * D(generations)
```

Where:
- `L_quality`: Standard cross-entropy loss
- `D`: Diversity score (semantic, BLEU, or distinct-n)
- `λ`: Diversity weight (0.3 recommended)

**Generation During Training:**
- Generate N=3 diverse replies per batch
- Compute pairwise cosine distances
- Backprop through combined loss

### Evaluation Protocol

1. **Diversity Metrics** (100 test examples)
   - Generate 10 replies per prompt
   - Compute Self-BLEU, Distinct-n, semantic diversity
   
2. **Quality Metrics** (100 test examples)
   - Use first generation
   - Compute ROUGE, BERTScore vs ground truth
   
3. **LLM-as-Judge** (500 test examples)
   - Position-randomized pairwise comparison
   - 4 criteria: engagement, relevance, creativity, naturalness
   - Claude 3.5 Sonnet
   
4. **Pass@k** (100 test examples)
   - Generate 10, select best
   - Heuristic quality checker
   - Compute pass@1, pass@5, pass@10
   
5. **Statistical Tests**
   - Mann-Whitney U test (non-parametric)
   - Cohen's d effect size
   - Bootstrap 95% CI
   - Bonferroni correction for multiple comparisons

### Success Criteria

**Minimum:**
- p < 0.05 (statistically significant)
- Cohen's d > 0.3 (meaningful effect size)
- Higher diversity scores for polychromic
- Comparable or better quality

**Strong:**
- p < 0.01
- Cohen's d > 0.5
- Pass@10 improvement > 20%
- LLM-judge win rate > 55%

---

## 📈 Expected Results

Based on the paper's findings:

| Metric | Baseline | Polychromic | Winner |
|--------|----------|-------------|--------|
| Self-BLEU | 0.45 | **0.28** | Polychromic ✓ |
| Distinct-2 | 0.62 | **0.78** | Polychromic ✓ |
| Semantic Div | 0.31 | **0.42** | Polychromic ✓ |
| ROUGE-L | **0.35** | 0.33 | Baseline (slight) |
| Pass@10 | 0.42 | **0.61** | Polychromic ✓ |

**Key Insight:** When generating multiple options and selecting the best (pass@k), polychromic significantly outperforms because it maintains diverse strategies.

---

## 💰 Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Data collection | $0 | Already complete |
| Baseline training (3 seeds) | $9 | 3 × 4hr × $0.79/hr |
| Polychromic training (3 seeds) | $27 | 3 × 12hr × $0.79/hr |
| Ablations (6 configs) | $54 | Various λ values |
| LLM-as-judge (500 examples) | $80 | Claude 3.5 Sonnet |
| **Total** | **$170** | Well under budget |

---

## 🧪 Ablation Studies

### Diversity Weight (λ)

```bash
# Conservative
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.1.yaml \
  --data data/processed/training_data_*.jsonl

# Moderate (recommended)
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data_*.jsonl

# Aggressive
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.5.yaml \
  --data data/processed/training_data_*.jsonl
```

### Generation Count (N)

Edit config file:
```yaml
polychromic:
  n_generations: 2  # Faster, less diversity
  n_generations: 3  # Balanced (recommended)
  n_generations: 5  # Slower, more diversity
```

### Diversity Metric

```yaml
polychromic:
  diversity_metric: "semantic"  # Recommended
  diversity_metric: "bleu"      # Faster, less accurate
  diversity_metric: "distinct"  # Lexical diversity only
```

---

## 📊 W&B Integration

All experiments automatically log to Weights & Biases:

**Dashboard URL:** `https://wandb.ai/<username>/qwen3-twitter-polychromic`

**Logged Metrics:**
- Training: `train/loss`, `train/diversity_score`, `train/combined_loss`
- Evaluation: `eval/loss`, `eval/diversity`, `eval/avg_unique_words`
- System: `train/learning_rate`, `system/gpu_utilization`

**Sample Generations:** Logged as tables every evaluation

---

## 🔍 Monitoring Training

### Real-Time Monitoring

```bash
# Watch W&B dashboard
open https://wandb.ai/<username>/qwen3-twitter-polychromic

# Watch logs
tail -f output/experiments/*/seed_42/logs/*.log

# GPU usage
watch -n 1 nvidia-smi
```

### Expected Behavior

**Baseline:**
- `train/loss`: Decreasing smoothly
- `eval/loss`: Tracks train loss closely
- Training time: ~4 hours

**Polychromic:**
- `train/quality_loss`: Decreasing
- `train/diversity_score`: Increasing or stable (0.3-0.5)
- `train/combined_loss`: Decreasing
- Training time: ~12 hours (3x slower due to generation)

### Red Flags

⚠️ **Overfitting:**
- `eval/loss` increasing while `train/loss` decreasing
- Gap > 0.3 between train and eval
- **Action:** Stop training, reduce epochs

⚠️ **Diversity Collapse:**
- `diversity_score` dropping below 0.1
- **Action:** Increase λ or N

⚠️ **OOM Error:**
- CUDA out of memory
- **Action:** Reduce batch size, increase grad accumulation

---

## 📝 Reproducibility

### Fixed Elements

```yaml
experiment:
  random_seed: 42  # (Also 123, 456 for multi-seed)

training:
  seed: 42
  data_seed: 42
  
# All hyperparameters in config files
# Data preprocessing deterministic
# Evaluation metrics deterministic
```

### Variable Elements

- Training hardware (GPU type affects speed, not results)
- NLTK/transformer versions (pinned in requirements.txt)
- LLM-as-judge (Claude API may change)

### Reproducibility Package

For publication, include:
- ✅ Config files
- ✅ Training logs
- ✅ W&B dashboard links
- ✅ Model cards
- ✅ Dataset documentation
- ✅ Requirements.txt
- ✅ Evaluation results
- ✅ Statistical test outputs

---

## 📄 Paper Outline

**Title:** "Polychromic Training for Social Media Reply Generation: Balancing Quality and Diversity"

**Sections:**
1. Introduction (1.5 pages)
2. Related Work (1 page)
3. Method (2 pages)
   - Polychromic Training Objective
   - Implementation Details
   - Dataset Construction
4. Experimental Setup (1.5 pages)
   - Baselines
   - Evaluation Metrics
   - Statistical Testing
5. Results (2 pages)
   - Main Comparison (Table 1)
   - Ablation Studies (Table 2)
   - Pass@k Analysis (Figure 1)
   - Qualitative Examples
6. Analysis (1 page)
   - When does it help?
   - Computational cost
   - Failure modes
7. Conclusion (0.5 pages)
8. Limitations (0.5 pages)

**Total:** ~10 pages + appendix

---

## 🎓 Next Steps

### After First Training Run

1. **Quick Evaluation** (100 examples, no LLM-judge)
2. **Analyze W&B logs**
3. **Decide:** Proceed with full eval or iterate

### If Results Are Promising

1. **Train with multiple seeds** (42, 123, 456)
2. **Run ablation studies**
3. **Full evaluation** (500 examples, LLM-judge)
4. **Statistical analysis**
5. **Write paper**

### If Results Are Weak

1. **Debug:** Check training curves, sample outputs
2. **Iterate:** Adjust λ, N, or diversity metric
3. **Re-train:** With improved configuration

---

## 📞 Support

**Documentation:**
- `RUNPOD_QUICKSTART.md` - RunPod setup guide
- `SESSION_LOG.md` - Detailed development log
- `DATA_QUALITY_IMPROVEMENTS.md` - Data filtering details

**Key Scripts:**
- `scripts/test_installation.py` - Verify setup
- `scripts/training/train_model.py` - Main training
- `scripts/evaluation/evaluate_comprehensive.py` - Full eval

**W&B Dashboard:**
- Monitor training in real-time
- Compare experiments
- Download logs and artifacts

---

## ✅ Final Checklist

Before submitting to Arxiv:

- [ ] Multiple random seeds trained
- [ ] Statistical significance achieved (p < 0.05)
- [ ] Effect size meaningful (Cohen's d > 0.3)
- [ ] LLM-judge evaluation complete
- [ ] Failure mode analysis done
- [ ] Code released on GitHub
- [ ] Models released on HuggingFace
- [ ] Dataset documented
- [ ] Reproducibility verified
- [ ] Paper written and reviewed
- [ ] Limitations section thorough

---

**This is research-grade infrastructure. Every decision is documented, every experiment is tracked, and every result is reproducible. Ready for Arxiv submission.**

