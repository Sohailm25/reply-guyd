# 🎯 Complete Workflow: Data Collection → Paper Submission

**A step-by-step guide from data collection completion to paper submission**

**Last Updated:** October 3, 2025  
**Status:** Ready to use when data collection completes

---

## 📊 Current Project Status

**Phase:** Data Collection (60% complete)  
**Next Phase:** Manual Curation → Training → Evaluation → Paper

---

## 🗺️ Complete Workflow Overview

```
┌─────────────────┐
│ 1. Data         │  ⏳ IN PROGRESS
│    Collection   │     Running in background with tmux
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 2. Verify       │  ⏳ NEXT STEP (when collection completes)
│    Collection   │     Check data quality, count, stats
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 3. Manual       │  Manual review of collected data
│    Curation     │  Filter bad examples, ensure quality
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 4. Prepare      │  Create train/val/test splits
│    Dataset      │  Final JSONL file ready for training
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 5. Train        │  48 hours (3 seeds × 2 experiments)
│    Models       │  Baseline + Polychromic on RunPod
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 6. Evaluate     │  Run comprehensive evaluation
│    Models       │  All 4 baselines on test set
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 7. Aggregate    │  Compute mean ± std across seeds
│    Results      │  Generate LaTeX tables
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 8. Write        │  Draft paper sections
│    Paper        │  Incorporate results
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 9. Submit!      │  🎉 Arxiv/Conference
│                 │
└─────────────────┘
```

---

## ✅ Step 1: Data Collection (CURRENT)

### **Status Check:**
```bash
# Check if collection is still running
tmux ls

# Attach to see progress
tmux attach -t data_collection

# Check data file
wc -l data/raw/training_data_master_*.jsonl
```

### **When Complete:**
You'll see: "✅ Data collection complete! Collected XXXX examples."

### **Exit tmux:**
Press `Ctrl+B` then `D` (don't kill the session!)

---

## ✅ Step 2: Verify Data Collection

### **Run Verification Script:**
```bash
# Count total examples
wc -l data/raw/training_data_master_*.jsonl

# Check for duplicates
python scripts/data/verify_collection.py \
    --data data/raw/training_data_master_*.jsonl
```

### **Expected Output:**
- **Total examples:** 1,000-2,000 pairs
- **Unique tweets:** >80%
- **Author diversity:** >50 unique authors
- **Engagement:** All meet thresholds

### **Quality Check:**
```bash
# Sample random examples
python scripts/data/sample_data.py \
    --data data/raw/training_data_master_*.jsonl \
    --n-samples 20 \
    --output data/quality_check_samples.txt
```

### **Review Samples:**
Open `data/quality_check_samples.txt` and verify:
- ✅ Tweets are relevant
- ✅ Replies are high quality
- ✅ No spam or garbage
- ✅ Appropriate content

**If quality is poor:** Adjust filters in `config/data_collection_config.yaml` and re-run collection.

---

## ✅ Step 3: Manual Curation

### **Create Curation File:**
```bash
# Copy to curation directory
cp data/raw/training_data_master_*.jsonl data/curation/all_collected.jsonl

# Create curation script
python scripts/data/start_curation.py \
    --input data/curation/all_collected.jsonl \
    --output data/curation/curated_examples.jsonl
```

### **Manual Review Process:**

**For each example, check:**
1. **Tweet quality:**
   - [ ] Clear, understandable
   - [ ] Not spam or promotional
   - [ ] Appropriate content

2. **Reply quality:**
   - [ ] Relevant to tweet
   - [ ] Adds value (not just "agreed!")
   - [ ] Natural language
   - [ ] Good engagement signal

3. **Pairing quality:**
   - [ ] Reply makes sense for tweet
   - [ ] Good example of what we want to generate

### **Keep or Discard:**
```bash
# Mark examples as keep/discard
# Script will show each example and ask:
# Keep this example? (y/n/s for skip)
```

### **Target After Curation:**
- **Minimum:** 800 high-quality pairs
- **Ideal:** 1,000-1,500 pairs
- **Maximum:** 2,000 pairs (if available)

### **Save Curated Data:**
```bash
# Curated examples saved to:
# data/curation/curated_examples.jsonl
```

---

## ✅ Step 4: Prepare Final Dataset

### **Create Train/Val/Test Splits:**
```bash
python scripts/data/prepare_final_dataset.py \
    --input data/curation/curated_examples.jsonl \
    --output data/processed/training_data_final.jsonl \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1 \
    --seed 42
```

### **Verify Splits:**
```bash
# Check split statistics
python scripts/data/verify_splits.py \
    --data data/processed/training_data_final.jsonl
```

### **Expected Output:**
```
Dataset Statistics:
  Total examples: 1200
  Train: 960 (80%)
  Val: 120 (10%)
  Test: 120 (10%)

Engagement distribution:
  Train: mean=157, median=98
  Val: mean=162, median=102
  Test: mean=155, median=95

Author diversity:
  Train: 68 unique authors
  Val: 42 unique authors (some overlap OK)
  Test: 45 unique authors (some overlap OK)

✅ Splits look good!
```

---

## ✅ Step 5: Train Models (RunPod)

### **5.1 Upload Data to RunPod:**

```bash
# Start RunPod instance (24GB GPU recommended)
# See: docs/runpod-training/RUNPOD_QUICKSTART.md

# Upload final dataset
scp data/processed/training_data_final.jsonl \
    root@<RUNPOD_IP>:/workspace/Qwen3-8/data/processed/
```

### **5.2 Train All Seeds (Baseline):**

```bash
# SSH into RunPod
ssh root@<RUNPOD_IP>

# Navigate to project
cd /workspace/Qwen3-8

# Activate environment
source qwen-lora-env/bin/activate

# Train baseline with all 3 seeds
bash scripts/training/train_all_seeds.sh baseline
```

**Time:** 12 hours (4 hrs × 3 seeds)  
**Cost:** ~$9

**Monitor Progress:**
```bash
# Check W&B dashboard
# https://wandb.ai/your-username/qwen3-twitter-polychromic

# Or tail logs
tail -f output/experiments/baseline/seed_42/logs/train.log
```

### **5.3 Train All Seeds (Polychromic):**

```bash
# Train polychromic with all 3 seeds
bash scripts/training/train_all_seeds.sh polychromic_0.3
```

**Time:** 36 hours (12 hrs × 3 seeds)  
**Cost:** ~$27

### **5.4 Download Trained Models:**

```bash
# From your local machine
scp -r root@<RUNPOD_IP>:/workspace/Qwen3-8/output/experiments/ \
    output/experiments/
```

**Total Training:**
- **Time:** 48 hours (2 days)
- **Cost:** $36
- **Models:** 6 total (3 baseline + 3 polychromic)

---

## ✅ Step 6: Evaluate Models

### **6.1 Prepare Evaluation:**

```bash
# Verify test data exists
ls -lh data/processed/training_data_final.jsonl

# Verify models downloaded
ls -lh output/experiments/baseline/seed_*/
ls -lh output/experiments/polychromic_0.3/seed_*/
```

### **6.2 Run Evaluation (All Seeds):**

```bash
# Evaluate seed 42
python scripts/evaluation/evaluate_comprehensive.py \
    --include-zero-shot \
    --include-prompt-engineered \
    --prompt-variant with_examples \
    --baseline-lora output/experiments/baseline/seed_42 \
    --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
    --test-data data/processed/training_data_final.jsonl \
    --output output/evaluation/seed_42/ \
    --max-examples 100 \
    --n-generations 10
```

**Repeat for seeds 123 and 456:**
```bash
# Seed 123
python scripts/evaluation/evaluate_comprehensive.py \
    --include-zero-shot \
    --include-prompt-engineered \
    --baseline-lora output/experiments/baseline/seed_123 \
    --polychromic-lora output/experiments/polychromic_0.3/seed_123 \
    --test-data data/processed/training_data_final.jsonl \
    --output output/evaluation/seed_123/ \
    --max-examples 100 \
    --n-generations 10

# Seed 456
python scripts/evaluation/evaluate_comprehensive.py \
    --include-zero-shot \
    --include-prompt-engineered \
    --baseline-lora output/experiments/baseline/seed_456 \
    --polychromic-lora output/experiments/polychromic_0.3/seed_456 \
    --test-data data/processed/training_data_final.jsonl \
    --output output/evaluation/seed_456/ \
    --max-examples 100 \
    --n-generations 10
```

**Time per seed:** 2-4 hours  
**Total evaluation time:** 6-12 hours

### **6.3 Verify Evaluation Output:**

```bash
# Check each seed directory
ls -lh output/evaluation/seed_42/
# Should contain:
#   - diversity_metrics.json
#   - quality_metrics.json
#   - passk_results.json
#   - statistical_tests.json (if applicable)
```

---

## ✅ Step 7: Aggregate Results

### **7.1 Aggregate Across Seeds:**

```bash
python scripts/analysis/aggregate_seeds.py \
    --results-dir output/evaluation/ \
    --seeds 42 123 456 \
    --output output/evaluation/aggregated_results.json \
    --latex-output output/evaluation/results_table.tex
```

### **7.2 Review Aggregated Results:**

```bash
# View aggregated metrics
cat output/evaluation/aggregated_results.json | jq .

# Example output:
{
  "zero_shot": {
    "pass@10": {
      "mean": 0.45,
      "std": 0.02,
      "values": [0.43, 0.46, 0.46]
    },
    ...
  },
  "prompt_engineered": { ... },
  "baseline_lora": { ... },
  "polychromic_lora": { ... }
}
```

### **7.3 Generate Visualizations:**

```bash
python scripts/analysis/visualize_results.py \
    --aggregated output/evaluation/aggregated_results.json \
    --output output/figures/
```

**Generated figures:**
- `diversity_comparison.png` - Diversity metrics across models
- `passk_curves.png` - Pass@k performance
- `quality_comparison.png` - ROUGE, BERTScore
- `results_table.tex` - LaTeX table for paper

---

## ✅ Step 8: Write Paper

### **8.1 Paper Structure:**

```
1. Abstract
2. Introduction
3. Related Work
4. Method (Polychromic LoRA)
5. Experimental Setup
   - Dataset
   - Baselines (4 total)
   - Evaluation Metrics
   - Implementation Details
6. Results
   - Main Results (Table with mean ± std)
   - Pass@k Analysis
   - Diversity vs Quality Trade-off
   - Ablation Studies
7. Discussion
8. Conclusion
9. References
```

### **8.2 Key Results to Report:**

**Format:**
```latex
We report metrics as mean ± standard deviation across 3 random 
seeds (42, 123, 456). Quality metrics use greedy decoding 
(temperature=0) for reproducibility. Pass@k metrics use temperature 
sampling (temperature=0.7) for diversity.
```

**Main Results Table:**
```latex
\begin{table}[t]
\centering
\caption{Performance comparison across four baselines}
\begin{tabular}{lcccc}
\toprule
Metric & Zero-shot & Few-shot & LoRA & Polychromic \\
\midrule
Pass@10 & 0.45$\pm$0.02 & 0.58$\pm$0.03 & 0.72$\pm$0.03 & \textbf{0.82$\pm$0.02} \\
ROUGE-L & 0.31$\pm$0.01 & 0.39$\pm$0.02 & 0.48$\pm$0.02 & \textbf{0.51$\pm$0.01} \\
Self-BLEU & 0.78$\pm$0.02 & 0.71$\pm$0.03 & 0.65$\pm$0.02 & \textbf{0.58$\pm$0.02} \\
\bottomrule
\end{tabular}
\end{table}
```

### **8.3 Statistical Significance:**

Report p-values and effect sizes:
```
Polychromic LoRA significantly outperforms standard LoRA on 
Pass@10 (p < 0.001, Cohen's d = 1.2), while maintaining 
comparable single-generation quality (ROUGE-L p = 0.08).
```

### **8.4 Paper Sections to Complete:**

- [ ] Abstract (200 words)
- [ ] Introduction (1.5 pages)
- [ ] Related Work (1 page)
- [ ] Method (2 pages)
- [ ] Experimental Setup (1 page)
- [ ] Results (2 pages)
- [ ] Discussion (1 page)
- [ ] Conclusion (0.5 pages)

**Target Length:** 8-10 pages (excluding references)

---

## ✅ Step 9: Submit Paper

### **9.1 Pre-Submission Checklist:**

- [ ] All experiments completed (3 seeds)
- [ ] Results aggregated with mean ± std
- [ ] Figures generated
- [ ] Tables formatted in LaTeX
- [ ] Paper written and proofread
- [ ] References complete
- [ ] Code/data availability statement
- [ ] Supplementary materials prepared

### **9.2 Target Venues:**

**Top-Tier ML:**
- ICLR (Jan deadline)
- NeurIPS (May deadline)
- ICML (Feb deadline)

**Top-Tier NLP:**
- EMNLP (May deadline)
- ACL (Feb deadline)

**Arxiv:**
- Submit immediately (no deadline)
- Get arxiv ID for references

### **9.3 Submission:**

```bash
# Create submission package
python scripts/paper/prepare_submission.py \
    --paper paper/polychromic_lora.tex \
    --figures output/figures/ \
    --output submission/
```

---

## 📋 Quick Reference Checklist

```
Data Phase:
  [ ] Data collection complete (1000-2000 pairs)
  [ ] Quality verification passed
  [ ] Manual curation complete (800+ kept)
  [ ] Final dataset prepared
  [ ] Train/val/test splits created

Training Phase:
  [ ] Baseline trained (3 seeds)
  [ ] Polychromic trained (3 seeds)
  [ ] Models downloaded locally
  [ ] W&B logs reviewed

Evaluation Phase:
  [ ] Evaluation run (all 3 seeds)
  [ ] Results aggregated
  [ ] Visualizations generated
  [ ] LaTeX tables created

Paper Phase:
  [ ] Paper written
  [ ] Results incorporated
  [ ] Proofread and polished
  [ ] Submission ready

Submission:
  [ ] Venue selected
  [ ] Submission package prepared
  [ ] Paper submitted!
  [ ] Celebrate! 🎉
```

---

## ⏱️ Timeline Estimates

| Phase | Time | Dependencies |
|-------|------|--------------|
| Data Collection | 2-7 days | Running now |
| Manual Curation | 4-8 hours | After collection |
| Dataset Prep | 1 hour | After curation |
| Training | 48 hours | After prep |
| Evaluation | 6-12 hours | After training |
| Result Analysis | 4-8 hours | After evaluation |
| Paper Writing | 1-2 weeks | After analysis |
| **TOTAL** | **3-4 weeks** | From now |

---

## 🚨 Common Issues & Solutions

### **Issue: Not enough data after curation**
**Solution:** Adjust curation criteria or collect more data

### **Issue: Training crashes**
**Solution:** Check GPU memory, reduce batch size, use gradient checkpointing

### **Issue: Evaluation takes too long**
**Solution:** Reduce `--max-examples` to 50 for quick tests

### **Issue: Results not significant**
**Solution:** Check if diversity_weight (λ) needs adjustment, run ablations

---

## 📚 Key Documents to Reference

- **Training:** `docs/runpod-training/RUNPOD_QUICKSTART.md`
- **Evaluation:** `docs/implementation/QUICK_START_EVAL_IMPROVEMENTS.md`
- **Research:** `docs/implementation/RESEARCH_IMPLEMENTATION.md`
- **Status:** `docs/project-status/PROJECT_STATUS.md`

---

## ✅ You're Ready!

Everything is in place:
- ✅ Data collection running
- ✅ Training infrastructure ready
- ✅ Evaluation framework complete
- ✅ Multi-seed configs created
- ✅ Aggregation scripts ready
- ✅ Documentation complete

**Next action:** Wait for data collection to complete, then follow this workflow step by step!

---

**Created:** October 3, 2025  
**Status:** Ready for use  
**Estimated completion:** 3-4 weeks from data collection finish

