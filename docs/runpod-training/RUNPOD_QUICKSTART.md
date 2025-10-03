# RunPod Quick Start Guide for Polychromic Training

## 🚀 Overview

This guide shows you how to train both baseline and polychromic models on RunPod.

**Estimated Costs:**
- Baseline training: ~$3 (4 hours on A6000)
- Polychromic training: ~$9 (12 hours on A6000)
- Total for both: ~$12

---

## 📋 Prerequisites

1. RunPod account created
2. Data collection complete (800-1,200 curated pairs)
3. `.env` file with API keys (W&B, Anthropic)

---

## 🖥️ RunPod Setup

### Step 1: Create Pod

1. Go to [RunPod](https://www.runpod.io/)
2. Click "Deploy"
3. Select **GPU Instance**
4. Choose: **RTX A6000 (48GB)**
5. Select Template: **PyTorch 2.1**
6. Storage: **50GB Container Disk** + **50GB Volume Disk**
7. Click **Deploy**

### Step 2: Connect to Pod

```bash
# SSH into pod (use credentials from RunPod dashboard)
ssh root@<pod-ip> -p <port>
```

---

## 📦 Installation

### Quick Install Script

```bash
# Clone repository
cd /workspace
git clone <your-repo-url> qwen-training
cd qwen-training

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Verify installation
python -c "import torch; import transformers; import peft; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+cu121
CUDA: True
```

---

## 📊 Upload Data

### Option 1: Direct Upload (Small datasets < 100MB)

Use RunPod web interface to upload files to `/workspace/qwen-training/data/processed/`

### Option 2: rsync (Recommended)

```bash
# From your Mac:
rsync -avz -e "ssh -p <port>" \
  data/processed/training_data_*.jsonl \
  root@<pod-ip>:/workspace/qwen-training/data/processed/

# Verify
ssh root@<pod-ip> -p <port> "ls -lh /workspace/qwen-training/data/processed/"
```

### Option 3: Cloud Storage

```bash
# Upload to S3/GCS from Mac, download on RunPod
# Example with S3:
aws s3 cp data/processed/training_data_*.jsonl s3://your-bucket/

# On RunPod:
aws s3 cp s3://your-bucket/training_data_*.jsonl data/processed/
```

---

## 🎯 Training

### Baseline Model

```bash
cd /workspace/qwen-training

# Copy .env file with API keys
# (Upload via RunPod interface or paste content)
nano .env
# Add:
# WANDB_API_KEY=your_key_here

# Train baseline
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data_*.jsonl

# Monitor progress:
# - W&B dashboard: https://wandb.ai/<your-username>/qwen3-twitter-polychromic
# - Local logs: tail -f output/experiments/baseline/seed_42/logs/*.log
```

**Expected Duration:** 2-4 hours

### Polychromic Model

```bash
# After baseline completes:
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data_*.jsonl

# Monitor progress in W&B
```

**Expected Duration:** 6-12 hours

---

## 💾 Download Results

### Download Models

```bash
# From your Mac:
rsync -avz -e "ssh -p <port>" \
  root@<pod-ip>:/workspace/qwen-training/output/experiments/ \
  output/experiments/

# This downloads:
# - output/experiments/baseline/seed_42/
# - output/experiments/polychromic_0.3/seed_42/
```

### Verify Downloaded Models

```bash
# On Mac:
ls -lh output/experiments/baseline/seed_42/
# Should see:
# - adapter_config.json
# - adapter_model.safetensors
# - experiment_config.yaml
# - checkpoint-*/
```

---

## 🔍 Monitoring

### W&B Dashboard

Real-time monitoring from Mac:
- Open: `https://wandb.ai/<username>/qwen3-twitter-polychromic`
- Watch:
  - `train/loss` - should decrease
  - `train/diversity_score` - polychromic should increase
  - `eval/loss` - should track train loss
  - `train/learning_rate` - should follow cosine schedule

### SSH Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch logs
tail -f output/experiments/*/seed_42/logs/*.log

# Check training progress
grep "Training" output/experiments/*/seed_42/logs/*.log | tail -n 20
```

---

## 🎓 Training Tips

### If Training Fails

```bash
# Resume from checkpoint
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data_*.jsonl \
  --resume

# Training will automatically resume from last checkpoint
```

### If Out of Memory

Edit config file:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Speed Up Training

```bash
# Reduce evaluation frequency
# Edit config/experiments/*.yaml:
training:
  eval_steps: 200  # Increase from 50/100
```

---

## 📊 Evaluation on RunPod (Optional)

You can evaluate on RunPod or download models and evaluate on Mac.

### On RunPod:

```bash
# Comprehensive evaluation
python scripts/evaluation/evaluate_comprehensive.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --anthropic-key <your-key>

# Download results
rsync -avz -e "ssh -p <port>" \
  root@<pod-ip>:/workspace/qwen-training/output/evaluation/ \
  output/evaluation/
```

---

## 💰 Cost Management

### Stop Pod When Idle

```bash
# Always stop pod when not training!
# From RunPod dashboard: Stop → Terminate

# Or keep paused if resuming soon:
# Stop (don't terminate)
```

### Estimated Costs

| Task | Duration | GPU | Cost |
|------|----------|-----|------|
| Setup | 30 min | A6000 | $0.40 |
| Baseline training | 4 hours | A6000 | $3.20 |
| Polychromic training | 12 hours | A6000 | $9.50 |
| Evaluation | 2 hours | A6000 | $1.60 |
| **Total** | **18.5 hours** | | **$14.70** |

**Tip:** Run evaluation on Mac to save $1.60

---

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'transformers'`

```bash
pip install -r requirements.txt
```

### `RuntimeError: CUDA out of memory`

Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### `Connection timeout` or `SSH disconnected`

Use `tmux` to keep training running:
```bash
# Start tmux session
tmux new -s training

# Run training
python scripts/training/train_model.py ...

# Detach: Ctrl+B, then D

# Reconnect later
tmux attach -s training
```

### Models not saving

Check disk space:
```bash
df -h
# If low, increase volume disk in RunPod settings
```

---

## 📋 Complete RunPod Workflow

```bash
# 1. Create Pod (A6000, 50GB disk)

# 2. Setup
cd /workspace
git clone <repo> qwen-training
cd qwen-training
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 3. Upload data
# (use rsync from Mac)

# 4. Train baseline
tmux new -s baseline
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --data data/processed/training_data_*.jsonl
# Detach: Ctrl+B, D

# 5. Monitor via W&B
# (open on Mac browser)

# 6. Train polychromic (after baseline completes)
tmux new -s polychromic
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml \
  --data data/processed/training_data_*.jsonl
# Detach: Ctrl+B, D

# 7. Download results
# (rsync from Mac)

# 8. STOP POD (important!)
# RunPod dashboard → Stop → Terminate
```

---

## ✅ Checklist

Before training:
- [ ] Data uploaded to RunPod
- [ ] .env file with W&B API key
- [ ] Config files reviewed
- [ ] Enough disk space (50GB+)

During training:
- [ ] W&B dashboard showing metrics
- [ ] tmux session active
- [ ] GPU utilization >90%

After training:
- [ ] Models downloaded to Mac
- [ ] W&B logs saved
- [ ] Pod stopped/terminated
- [ ] Checkpoint files verified

---

## 🎯 Next Steps

After training completes:
1. Download models to Mac
2. Run evaluation (on Mac or RunPod)
3. Analyze results
4. If needed, train with different hyperparameters

See `scripts/evaluation/evaluate_comprehensive.py` for evaluation options.

