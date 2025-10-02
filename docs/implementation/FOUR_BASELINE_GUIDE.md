# Complete Four-Baseline Evaluation Guide

## 🎯 Overview

You can now evaluate up to **4 different approaches**:

1. **Zero-Shot** - Base Qwen3 with simple prompt (no training)
2. **Prompt-Engineered** - Base Qwen3 with optimized prompt (no training)
3. **Baseline LoRA** - Standard supervised fine-tuning
4. **Polychromic LoRA** - Diversity-aware fine-tuning

**This strengthens your research significantly!**

---

## 📊 Model Comparison Table

| Model | Training? | Parameters | Cost | Strengths |
|-------|-----------|------------|------|-----------|
| **Zero-Shot** | No | 0 | $0 | Fast baseline, no training needed |
| **Prompt-Engineered** | No | 0 | $0 | Tests prompt engineering limits |
| **Baseline LoRA** | Yes (4hrs) | ~100M | $3 | Standard fine-tuning approach |
| **Polychromic LoRA** | Yes (12hrs) | ~100M | $9 | Novel diversity-aware approach |

---

## 🚀 Usage Examples

### **All Four Models**

```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --include-zero-shot \
  --include-prompt-engineered \
  --prompt-variant with_examples \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --max-examples 100 \
  --n-generations 10 \
  --anthropic-key $ANTHROPIC_API_KEY
```

### **Just Prompt-Based (No Training)**

```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --include-zero-shot \
  --include-prompt-engineered \
  --prompt-variant with_examples \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --max-examples 100
```

### **Just LoRA Models**

```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/
```

### **Zero-Shot vs Baseline (Shows Training Benefit)**

```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --include-zero-shot \
  --baseline-lora output/experiments/baseline/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/
```

---

## 🎨 Customizing Prompts

### **Available Prompt Variants**

Edit `src/evaluation/prompt_templates.py` to customize:

```python
PROMPT_VARIANTS = {
    "zero_shot":        # Simple: "Generate a reply to: {tweet}"
    "engineered":       # Detailed instructions, no examples
    "with_examples":    # 5 hardcoded examples + instructions
    "helpful":          # Optimized for helpful replies
    "conversational":   # Optimized for discussion
    "technical":        # Optimized for technical depth
}
```

### **Adding Your Best Examples**

**Step 1: Find your best training examples**
```bash
# Sort by engagement
jq -s 'sort_by(.reply_likes) | reverse | .[0:10]' data/processed/training_data_*.jsonl

# Review manually
cat data/processed/training_data_*.jsonl | jq . | less
```

**Step 2: Edit the template**
```python
# src/evaluation/prompt_templates.py

def get_prompt_with_examples(tweet: str) -> str:
    prompt = """You are an expert at writing engaging Twitter replies.

Example 1:
Tweet: [PASTE YOUR BEST TWEET HERE]
Reply: [PASTE THE HIGH-ENGAGEMENT REPLY]

Example 2:
[ADD MORE EXAMPLES...]

Now reply to this tweet:
{tweet}

Your reply:"""
    
    return prompt.format(tweet=tweet)
```

**Step 3: Re-run evaluation**
```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --include-prompt-engineered \
  --prompt-variant with_examples \
  ...
```

---

## 📈 Expected Results

### **Hypothesis:**

```
Quality (single-shot):
Zero-Shot < Prompt-Engineered < Baseline LoRA ≈ Polychromic LoRA

Diversity:
Zero-Shot ≈ Prompt-Engineered < Baseline LoRA < Polychromic LoRA

Pass@10:
Zero-Shot < Prompt-Engineered < Baseline LoRA < Polychromic LoRA ← Best!
```

### **Why This Matters for Research:**

1. **Zero-Shot** - Proves fine-tuning is necessary
2. **Prompt-Engineered** - Shows in-context learning limits
3. **Baseline LoRA** - Standard approach baseline
4. **Polychromic LoRA** - Novel contribution

**Having all four strengthens your Arxiv paper significantly!**

---

## 🔬 Research Questions You Can Answer

1. **Is fine-tuning necessary?**
   - Compare Zero-Shot vs Baseline LoRA

2. **How much can prompting help?**
   - Compare Zero-Shot vs Prompt-Engineered

3. **Does diversity-aware training work?**
   - Compare Baseline LoRA vs Polychromic LoRA

4. **What's the best approach overall?**
   - Compare all four on Pass@10

---

## 📊 Output Structure

```
output/evaluation/
├── diversity_metrics.json
│   ├── zero_shot: {...}
│   ├── prompt_engineered: {...}
│   ├── baseline_lora: {...}
│   └── polychromic_lora: {...}
│
├── quality_metrics.json
│   ├── zero_shot: {...}
│   ├── prompt_engineered: {...}
│   ├── baseline_lora: {...}
│   └── polychromic_lora: {...}
│
├── passk_results.json
│   ├── zero_shot: {1: 0.42, 5: 0.58, 10: 0.68}
│   ├── prompt_engineered: {...}
│   ├── baseline_lora: {...}
│   └── polychromic_lora: {...}
│
└── summary.json
```

---

## 🎨 Visualization

```bash
python scripts/analysis/visualize_results.py \
  --evaluation-dir output/evaluation/ \
  --output-dir output/figures/
```

**Generates:**
- `diversity_comparison.pdf` - 4 bars per metric
- `passk_curves.pdf` - 4 lines
- `summary_table.csv` - 4 columns
- `summary_table.tex` - 4 columns (LaTeX)

---

## ✅ Complete Workflow

### **Step 1: Train LoRA Models** (RunPod)
```bash
# Baseline
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data.jsonl

# Polychromic
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data.jsonl
```

### **Step 2: Customize Prompts** (Mac)
```bash
# Edit prompt templates
nano src/evaluation/prompt_templates.py

# Add your best examples to get_prompt_with_examples()
```

### **Step 3: Evaluate All Four** (Mac or RunPod)
```bash
python scripts/evaluation/evaluate_comprehensive.py \
  --include-zero-shot \
  --include-prompt-engineered \
  --prompt-variant with_examples \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/
```

### **Step 4: Visualize** (Mac)
```bash
python scripts/analysis/visualize_results.py \
  --evaluation-dir output/evaluation/ \
  --output-dir output/figures/
```

---

## 💰 Cost Breakdown

| Component | Cost |
|-----------|------|
| Zero-Shot evaluation | $0 (just inference) |
| Prompt-Engineered evaluation | $0 (just inference) |
| Baseline LoRA training | $3 |
| Polychromic LoRA training | $9 |
| Comprehensive evaluation | $0-80 (depending on LLM-judge) |
| **Total** | **$12-92** |

**Much cheaper than training multiple models!**

---

## 🎓 Research Value

**For your Arxiv paper:**

### **Section 4: Baselines**

```latex
\subsection{Baselines}

We compare our polychromic training approach against three baselines:

1. \textbf{Zero-Shot}: Base Qwen3-8B with minimal prompting
2. \textbf{Prompt-Engineered}: Base Qwen3-8B with carefully crafted instructions and examples
3. \textbf{Standard LoRA}: Fine-tuned with cross-entropy loss only

Table~\ref{tab:results} shows that polychromic training achieves 
the best Pass@10 performance while maintaining competitive quality...
```

**This is much stronger than just comparing two LoRA variants!**

---

## 📋 Quick Reference

**Evaluate all four:**
```bash
./scripts/evaluation/evaluate_comprehensive.py \
  --include-zero-shot --include-prompt-engineered \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/
```

**Customize prompts:**
```bash
nano src/evaluation/prompt_templates.py
# Edit get_prompt_with_examples()
```

**Generate figures:**
```bash
./scripts/analysis/visualize_results.py \
  --evaluation-dir output/evaluation/ \
  --output-dir output/figures/
```

---

**Your evaluation is now much more comprehensive and research-grade!** 🔬
