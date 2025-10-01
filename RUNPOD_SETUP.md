# RunPod Setup Guide

**Setting up Qwen3-8B LoRA training on RunPod**

---

## 🚀 Why RunPod?

- **Cost**: ~$0.39-0.79/hour for A6000/A100 (vs $1000+ for local GPU)
- **Performance**: 48GB VRAM, perfect for 8B model with QLoRA
- **Flexibility**: Pay only when training
- **No local setup**: Mac stays free for development

**Budget for this project**: $10-20 for 2-3 training runs

---

## 📋 RunPod Pod Configuration

### Recommended Pod Setup

**Template**: PyTorch 2.0+ with CUDA 12.1+

**GPU Options** (pick one):
- **RTX A6000** (48GB) - $0.39-0.79/hr ⭐ **Best value**
- **A100 PCIe** (40GB) - $1.29/hr - Faster but pricier
- **RTX 4090** (24GB) - $0.39/hr - Works with QLoRA 4-bit

**Storage**: 50GB minimum (model is 16GB, data ~5GB, outputs ~10GB)

**Container**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`

---

## 🛠️ Quick Start on RunPod

### Step 1: Launch Pod

1. Go to [runpod.io](https://runpod.io)
2. Click "Deploy" → "GPU Instances"
3. Select **RTX A6000** (48GB)
4. Template: PyTorch 2.x
5. Storage: 50GB
6. Click "Deploy"

### Step 2: Connect to Pod

```bash
# Option A: Web Terminal (in browser)
Click "Connect" → "Start Web Terminal"

# Option B: SSH (from your Mac)
ssh root@<pod-ip> -p <ssh-port>
# Port shown in RunPod dashboard
```

### Step 3: Clone Repository

```bash
# In RunPod terminal
cd /workspace

# Clone your repo (or upload via SSH)
git clone <your-repo-url>
# OR upload via scp from Mac:
# scp -P <port> -r ~/Repos/experiments/cuda/Qwen3-8 root@<pod-ip>:/workspace/

cd Qwen3-8
```

### Step 4: Run Setup

```bash
# Install dependencies
chmod +x setup.sh
./setup.sh

# Or manual installation:
pip install -r requirements.txt
```

### Step 5: Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
# API Keys (copy from your local .env)
WANDB_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Model paths
MODEL_PATH=/workspace/Qwen3-8
OUTPUT_DIR=/workspace/Qwen3-8/output/models
CACHE_DIR=/workspace/Qwen3-8/data/cache

# W&B
WANDB_PROJECT=qwen3-twitter-lora
EOF
```

### Step 6: Upload Training Data

```bash
# From your Mac, upload processed data
scp -P <port> data/processed/training_data_*.jsonl root@<pod-ip>:/workspace/Qwen3-8/data/processed/
```

### Step 7: Start Training

```bash
# Run training (Phase 3 - script to be created)
python scripts/train_lora.py --config config/lora_config.yaml

# Training will:
# - Load model in 4-bit (uses ~6GB VRAM)
# - Train LoRA adapters
# - Log to W&B
# - Save checkpoints every 50 steps
```

---

## 📦 What to Upload to RunPod

You **DON'T** need to upload the base model to RunPod! Download it there directly.

### Upload from Mac:

```bash
# 1. Processed training data (~5MB for 1,000 pairs)
scp -P <port> data/processed/training_data_*.jsonl root@<pod-ip>:/workspace/Qwen3-8/data/processed/

# 2. Configuration files
scp -P <port> config/*.yaml root@<pod-ip>:/workspace/Qwen3-8/config/

# 3. .env file (with API keys)
scp -P <port> .env root@<pod-ip>:/workspace/Qwen3-8/
```

### Download on RunPod:

```bash
# In RunPod terminal, download base model
cd /workspace/Qwen3-8
huggingface-cli download Qwen/Qwen3-8B --local-dir ./ --local-dir-use-symlinks False

# This downloads the 16GB model directly to RunPod (takes 5-10 min)
```

---

## 🔧 RunPod-Optimized Configuration

### Update `config/lora_config.yaml` for RunPod:

```yaml
# Paths (RunPod-specific)
model:
  path: "/workspace/Qwen3-8"
  
output:
  output_dir: "/workspace/Qwen3-8/output/models/lora-v1"
  logging_dir: "/workspace/Qwen3-8/output/logs/lora-v1"

# Training settings (optimized for A6000)
training:
  per_device_train_batch_size: 8  # Can increase on A6000
  gradient_accumulation_steps: 2  # Still effective batch = 16
  
  # Enable for faster training on RunPod
  bf16: true
  fp16: false
  gradient_checkpointing: true
```

---

## 💡 RunPod Workflow

### Development Loop (Recommended)

**On Mac** (free):
1. Develop code
2. Collect data via Apify
3. Review and clean data
4. Test scripts locally (no training)

**On RunPod** (paid):
1. Upload data + configs
2. Run training (2-4 hours)
3. Download trained model
4. Stop pod (save $)

**On Mac** (free):
1. Load trained model
2. Run evaluation
3. Analyze results
4. Adjust hyperparameters

**Repeat on RunPod** for v2, v3 training runs

### Cost Optimization

```bash
# Start pod only when ready to train
# Have everything prepared:
# ✓ Data collected and cleaned
# ✓ Config finalized
# ✓ Scripts tested

# Train efficiently:
# ✓ Use W&B to monitor remotely
# ✓ Enable early stopping
# ✓ Save checkpoints frequently

# Stop pod immediately after:
# ✓ Download LoRA adapters (small, ~50MB)
# ✓ Don't pay for idle time
```

---

## 📊 Training on RunPod (Step-by-Step)

### 1. Pre-Training Checks

```bash
# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Expected: NVIDIA RTX A6000

# Verify model loads
python -c "
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    '/workspace/Qwen3-8',
    quantization_config=config,
    device_map='auto'
)
print(f'✓ Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')
"
# Expected: ~4-6GB
```

### 2. Start Training

```bash
# Login to W&B (to monitor from Mac)
wandb login
# Paste your API key

# Start training
nohup python scripts/train_lora.py \
    --config config/lora_config.yaml \
    > output/logs/training.log 2>&1 &

# Get process ID
echo $! > training.pid
```

### 3. Monitor from Mac

```bash
# Watch W&B dashboard (real-time)
open https://wandb.ai/<username>/qwen3-twitter-lora

# Or SSH and tail logs
ssh root@<pod-ip> -p <port>
tail -f /workspace/Qwen3-8/output/logs/training.log
```

### 4. Training Complete

```bash
# Check if done
tail /workspace/Qwen3-8/output/logs/training.log

# Download LoRA adapters
scp -P <port> -r root@<pod-ip>:/workspace/Qwen3-8/output/models/lora-v1 ./output/models/

# Stop pod (IMPORTANT!)
# Go to RunPod dashboard → Stop Pod
```

---

## 🔍 Troubleshooting RunPod

### "CUDA Out of Memory"

```bash
# Reduce batch size in config/lora_config.yaml:
training:
  per_device_train_batch_size: 4  # or even 2
  gradient_accumulation_steps: 4  # keep effective = 16
```

### "Model download slow"

```bash
# Use mirror or resume
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-8B

# Or download specific files only
huggingface-cli download Qwen/Qwen3-8B --include "*.safetensors" "*.json"
```

### "Lost connection during training"

```bash
# Use tmux/screen for persistent sessions
tmux new -s training
python scripts/train_lora.py ...
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### "Pod crashed, lost everything"

```bash
# Enable auto-checkpoint saving
# Already configured in lora_config.yaml:
save_steps: 50
save_total_limit: 3

# Checkpoints saved to /workspace/Qwen3-8/output/models/lora-v1/
# Download periodically during training
```

---

## 💰 Cost Calculator

### Training Time Estimates

**Dataset: 1,000 pairs, 2 epochs**

| GPU | Time | Cost/hr | Total Cost |
|-----|------|---------|------------|
| RTX A6000 (48GB) | 2-3 hrs | $0.79 | $1.58-$2.37 |
| A100 PCIe (40GB) | 1.5-2 hrs | $1.29 | $1.94-$2.58 |
| RTX 4090 (24GB) | 3-4 hrs | $0.39 | $1.17-$1.56 |

**With Unsloth** (if working): 2x faster, half the cost

**For 3 training runs** (v1, v2, v3):
- Optimistic: $5-8
- Realistic: $10-15
- Pessimistic: $15-25

**Total project budget** (data + training + eval): $400-600

---

## 📁 File Transfer Best Practices

### Upload Strategy

```bash
# Compress data before upload (faster)
tar -czf training_data.tar.gz data/processed/training_data_*.jsonl

# Upload compressed
scp -P <port> training_data.tar.gz root@<pod-ip>:/workspace/Qwen3-8/

# Extract on RunPod
ssh root@<pod-ip> -p <port>
cd /workspace/Qwen3-8
tar -xzf training_data.tar.gz
```

### Download Strategy

```bash
# Download only LoRA adapters (small, ~50MB)
scp -P <port> -r root@<pod-ip>:/workspace/Qwen3-8/output/models/lora-v1 ./output/models/

# Download logs
scp -P <port> root@<pod-ip>:/workspace/Qwen3-8/output/logs/training.log ./output/logs/

# Don't download base model (already have it locally)
```

---

## ✅ RunPod Checklist

### Before Starting Pod:
- [ ] Data collected and cleaned locally
- [ ] Configs finalized
- [ ] Scripts tested (dry run locally)
- [ ] W&B logged in from Mac
- [ ] .env file prepared

### After Starting Pod:
- [ ] Clone/upload repo
- [ ] Run setup.sh
- [ ] Upload training data
- [ ] Download base model on RunPod
- [ ] Verify GPU and model loading
- [ ] Start training in tmux

### During Training:
- [ ] Monitor W&B dashboard
- [ ] Check logs periodically
- [ ] Download checkpoints every hour

### After Training:
- [ ] Download LoRA adapters
- [ ] Download logs
- [ ] Verify files received
- [ ] **STOP POD IMMEDIATELY**

---

## 🎯 Local (Mac) vs RunPod Division

| Task | Where | Why |
|------|-------|-----|
| Data collection | Mac | One-time, not GPU-intensive |
| Data cleaning | Mac | CPU task, free |
| Code development | Mac | Free, comfortable environment |
| Testing scripts | Mac | Can test without training |
| **Training** | **RunPod** | **Needs GPU, 2-4 hours** |
| Evaluation | Mac | Can load 4-bit locally |
| Analysis | Mac | CPU task, Jupyter notebooks |
| Hyperparameter tuning | Mac | Planning, then train on RunPod |

**Key**: Minimize RunPod time to minimize cost. Do everything else locally.

---

## 🚀 Quick Reference

### Start RunPod Training

```bash
# 1. Launch A6000 pod on runpod.io
# 2. SSH in
ssh root@<pod-ip> -p <port>

# 3. Setup
cd /workspace
git clone <repo-url>
cd Qwen3-8
pip install -r requirements.txt

# 4. Download model
huggingface-cli download Qwen/Qwen3-8B --local-dir ./ --local-dir-use-symlinks False

# 5. Upload data from Mac (separate terminal)
scp -P <port> data/processed/training_data_*.jsonl root@<pod-ip>:/workspace/Qwen3-8/data/processed/

# 6. Train
tmux new -s training
python scripts/train_lora.py --config config/lora_config.yaml

# 7. Monitor from Mac
open https://wandb.ai/<username>/qwen3-twitter-lora

# 8. When done, download adapters and STOP POD
```

---

## 📞 Support

- **RunPod Docs**: https://docs.runpod.io/
- **W&B Remote Monitoring**: https://docs.wandb.ai/
- **This Project**: See `lora_implementation_plan.md`

**Remember**: RunPod charges by the hour. Plan, prepare, execute, download, stop!

