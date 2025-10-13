# Evaluation Status & Instructions

## Current Status: ✓ RUNNING

**Started:** October 9, 2025 at 12:40 PM CDT  
**Tmux Session:** `qwen-eval`  
**Log File:** `evaluation_run.log`

---

## ⚠️ Important: CPU vs GPU Performance

The evaluation is currently running on **CPU** (no CUDA), which means:

- **Expected time on CPU:** 12-24 hours (vs. 1-2 hours on GPU)
- **Memory usage:** ~16GB RAM for the full-precision model
- **Progress:** Generating 500 examples × 10 generations × 2 models = 10,000 total generations

### Why So Slow?

Your Mac doesn't have CUDA, so we're loading the 8B parameter Qwen3 model in full precision (float32) on CPU. This is necessary for the evaluation to work, but it's much slower than GPU inference.

### Options:

1. **Let it run overnight** ✓ Recommended
   - The tmux session will keep running even if your laptop sleeps
   - Check progress periodically with `./check_progress.sh`
   - It will complete eventually (12-24 hours)

2. **Run on GPU instead** ⚡ Faster
   - Transfer evaluation to RunPod with GPU
   - Would complete in 1-2 hours instead
   - Cost: ~$2-3 for 2 hours on RTX A6000
   
3. **Reduce scope** 📉 Faster but less robust
   - Stop current run and restart with fewer examples
   - `--max-examples 100` instead of 500
   - Would reduce time to ~3-5 hours on CPU

---

## Monitoring Progress

### Quick Status Check
```bash
./check_progress.sh
```

### Watch Live Log
```bash
tail -f evaluation_run.log
```

### Attach to Tmux Session
```bash
tmux attach -t qwen-eval
```
- To detach without stopping: Press `Ctrl+b`, then `d`

### Check What Stage We're At
```bash
grep "Step" evaluation_run.log | tail -5
```

### Check Output Files
```bash
ls -lh output/evaluation/
ls -lh output/analysis/lora_comparison/
ls -lh output/evaluation/novel_metrics/
```

---

## Evaluation Pipeline Steps

The script runs 4 sequential steps:

### ✓ Step 1: Comprehensive Evaluation (CURRENT - Running)
**Time estimate:** 12-24 hours on CPU, 1-2 hours on GPU  
**What it does:**
- Loads both LoRA models (baseline + polychromic)
- Generates 10 replies per test example (500 examples)
- Computes diversity metrics (Self-BLEU, Distinct-n)
- Computes quality metrics (ROUGE)
- Computes Pass@k
- Runs LLM-as-judge (Claude API)
- Generates Pareto frontier plot

**Output files:**
- `output/evaluation/diversity_metrics.json`
- `output/evaluation/quality_metrics.json`
- `output/evaluation/passk_results.json`
- `output/evaluation/llm_judge_results.json`
- `output/evaluation/pareto_frontier.pdf` → **Figure 3**

**API Cost:** ~$80 (Claude calls for LLM-judge)

---

### Step 2: LoRA Parameter Analysis (Next)
**Time estimate:** 10 minutes  
**What it does:**
- Analyzes LoRA adapter weights
- Compares layer-wise parameter changes
- Identifies where diversity is encoded

**Output files:**
- `output/analysis/lora_comparison/lora_comparison.pdf` → **Figure 4**
- `output/analysis/lora_comparison/comparison_summary.json`

---

### Step 3: Novel Metrics Evaluation (Next)
**Time estimate:** 5 minutes  
**What it does:**
- Computes Diversity Efficiency Ratio (DER)
- Finds collapse points
- Generates comparison visualizations

**Output files:**
- `output/evaluation/novel_metrics/der_comparison.pdf` → **Figure 5**
- `output/evaluation/novel_metrics/collapse_points.pdf` → **Figure 6**

---

### Step 4: Results Organization (Final)
**Time estimate:** 1 minute  
**What it does:**
- Copies all figures to `paper/figures/`
- Copies all JSON data to `paper/data/`
- Creates organized structure for paper writing

---

## Expected Completion

**On CPU:** Tomorrow morning (Oct 10, 2025 ~12:00 PM)  
**On GPU:** Today evening (Oct 9, 2025 ~2:00 PM)

---

## Troubleshooting

### If the evaluation crashes:
```bash
# Check the log for errors
tail -100 evaluation_run.log

# Restart the evaluation
tmux send-keys -t qwen-eval './run_evaluation.sh 2>&1 | tee evaluation_run.log' C-m
```

### If you want to stop it:
```bash
# Attach to session
tmux attach -t qwen-eval

# Press Ctrl+C to stop

# Or kill the entire session
tmux kill-session -t qwen-eval
```

### If you want to reduce scope (faster):
```bash
# Edit run_evaluation.sh
# Change: --max-examples 500
# To: --max-examples 100

# Restart evaluation
tmux send-keys -t qwen-eval C-c
tmux send-keys -t qwen-eval './run_evaluation.sh 2>&1 | tee evaluation_run.log' C-m
```

---

## Success Indicators

When complete, you should have:

- [ ] 4 PDF figures in `paper/figures/`
- [ ] 6+ JSON files in `paper/data/`
- [ ] Log shows "EVALUATION COMPLETE!"
- [ ] No error messages in log

---

## Next Steps After Completion

1. **Verify Results**
   ```bash
   ls paper/figures/
   ls paper/data/
   ```

2. **Review Figures**
   - Open PDFs in `paper/figures/`
   - Check they look publication-quality

3. **Extract Key Numbers**
   ```bash
   cat paper/data/diversity_metrics.json | jq .
   cat paper/data/passk_results.json | jq .
   ```

4. **Start Paper Writing**
   - Use figures for paper sections 5.3-5.5
   - Extract numbers for tables
   - Follow `DATA_TO_PAPER_COMPLETE_WORKFLOW.md`

---

## Files Created

- `run_evaluation.sh` - Main evaluation script
- `check_progress.sh` - Progress monitoring script
- `evaluation_run.log` - Live log of evaluation
- `EVALUATION_STATUS.md` - This file

---

**Questions?** Check the logs or review the plan in `/scientific-evaluation-phase.plan.md`


