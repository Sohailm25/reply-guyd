# RunPod Transfer Guide - Evaluation

## Quick Start (3 Commands)

**On your Mac:**
```bash
# 1. Transfer files to RunPod (~5 minutes)
./sync_to_runpod.sh

# 2. SSH to RunPod
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io

# 3. On RunPod: Setup and start evaluation
cd /workspace/Qwen3-8
./runpod_setup.sh
```

That's it! The evaluation will run in tmux on RunPod with GPU acceleration.

---

## Detailed Steps

### Step 1: Transfer Files to RunPod (On Mac)

```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
./sync_to_runpod.sh
```

**What gets transferred (~350MB):**
- Trained LoRA adapters (baseline + polychromic)
- Test data (500 examples)
- All scripts and source code
- Evaluation pipeline
- .env file with API key

**Time:** ~5 minutes (depends on internet speed)

**Note:** The base Qwen3-8B model (~16GB) is NOT transferred. It should already be on RunPod from your training runs, or will be downloaded automatically.

---

### Step 2: SSH to RunPod

```bash
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io
```

You should see the RunPod prompt.

---

### Step 3: Setup and Start Evaluation (On RunPod)

```bash
cd /workspace/Qwen3-8
./runpod_setup.sh
```

**What this script does:**
1. Checks for base Qwen3-8B model (downloads if missing)
2. Verifies Python dependencies
3. Confirms adapter files exist
4. Loads API key from .env
5. Creates tmux session `qwen-eval`
6. Starts the evaluation pipeline

**Expected output:**
```
=========================================
✓ Setup Complete!
=========================================

🚀 Evaluation Started!

Session: qwen-eval
Expected time: 1-2 hours on GPU
Expected cost: ~$2-3 (compute) + $80 (LLM-judge)
```

---

### Step 4: Monitor Progress (On RunPod)

**Quick check:**
```bash
./check_progress.sh
```

**Watch live log:**
```bash
tail -f evaluation_run.log
```

**Attach to tmux session:**
```bash
tmux attach -t qwen-eval
```
Press `Ctrl+b`, then `d` to detach without stopping.

**Check output files:**
```bash
ls -lh output/evaluation/
ls -lh paper/figures/
```

---

### Step 5: Download Results (Back on Mac)

After the evaluation completes on RunPod (1-2 hours):

```bash
# On your Mac
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
./sync_from_runpod.sh
```

**What gets downloaded:**
- 4 publication-quality figures
- 6+ JSON files with metrics
- Analysis results
- Log file

**Time:** ~2 minutes

---

## Timeline & Costs

### On RunPod GPU (RTX A6000)

**Time:**
- Transfer to RunPod: 5 minutes
- Setup: 2 minutes
- Evaluation: 1-2 hours
- Download results: 2 minutes
- **Total: ~2 hours**

**Cost:**
- Compute: $0.79/hr × 2 hrs = ~$1.58
- LLM-judge API: ~$80
- **Total: ~$82**

### Comparison: Mac CPU vs RunPod GPU

| Aspect | Mac CPU | RunPod GPU |
|--------|---------|------------|
| Time | 12-24 hours | 1-2 hours |
| Can close laptop | ❌ No (pauses) | ✅ Yes |
| Need WiFi | ⚠️ For LLM-judge | ✅ Always on |
| Compute cost | Free | ~$2 |
| Total cost | $80 | $82 |

**Recommendation:** RunPod GPU ✓

---

## Monitoring While Away

Once the evaluation is running on RunPod, you can:

1. **Close your laptop** ✅ - RunPod keeps running
2. **Disconnect from WiFi** ✅ - RunPod has internet
3. **Go to sleep** ✅ - Evaluation continues
4. **Check later** - Just SSH back in

**To check if it's done:**
```bash
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io
cd /workspace/Qwen3-8
grep "EVALUATION COMPLETE" evaluation_run.log
```

If that prints something, it's done! Download results.

---

## Troubleshooting

### If base model is missing on RunPod:

The setup script will offer to download it:
```bash
huggingface-cli download Qwen/Qwen2.5-8B --local-dir ./ --local-dir-use-symlinks False
```

This takes ~10 minutes on RunPod (fast internet).

### If sync fails:

Check SSH connection:
```bash
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io "echo Connected"
```

If connection fails, check RunPod dashboard:
- Is the pod running?
- Is SSH exposed on port 22?
- Is the connection string correct?

### If evaluation fails:

Check the log:
```bash
tail -100 evaluation_run.log
```

Common issues:
- Out of memory: Reduce `--max-examples` in `run_evaluation.sh`
- CUDA errors: Check GPU with `nvidia-smi`
- API errors: Check `.env` has valid `ANTHROPIC_API_KEY`

---

## After Evaluation Completes

### On RunPod:
```bash
# Verify all outputs exist
ls -lh paper/figures/
# Should see 4 PDF files

ls -lh paper/data/
# Should see 6+ JSON files

# Check the log
tail -50 evaluation_run.log
# Should see "EVALUATION COMPLETE!"
```

### On Mac:
```bash
# Download everything
./sync_from_runpod.sh

# Verify locally
ls paper/figures/
ls paper/data/

# Optional: Stop RunPod pod to save money
# (Go to RunPod dashboard and stop the pod)
```

---

## Next Steps After Results Downloaded

1. **Review figures:**
   ```bash
   open paper/figures/*.pdf
   ```

2. **Extract key numbers:**
   ```bash
   cat paper/data/diversity_metrics.json | jq .
   cat paper/data/passk_results.json | jq .
   ```

3. **Start writing paper** - Follow `DATA_TO_PAPER_COMPLETE_WORKFLOW.md`

---

## Files Created

- `sync_to_runpod.sh` - Upload to RunPod
- `sync_from_runpod.sh` - Download from RunPod
- `runpod_setup.sh` - Setup on RunPod
- `RUNPOD_TRANSFER_GUIDE.md` - This file

---

## Summary Commands

**Mac → RunPod:**
```bash
./sync_to_runpod.sh
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io
cd /workspace/Qwen3-8
./runpod_setup.sh
# Detach with Ctrl+b, d
# Close laptop, go do something else for 2 hours
```

**Check status:**
```bash
ssh -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io \
  "cd /workspace/Qwen3-8 && tail -20 evaluation_run.log"
```

**RunPod → Mac:**
```bash
./sync_from_runpod.sh
ls paper/figures/
```

Done! 🎉

