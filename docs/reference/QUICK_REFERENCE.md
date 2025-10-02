# Quick Reference Card

**Essential commands for daily use**

---

## 🔧 Virtual Environment

The project uses a virtual environment with all dependencies installed.

### Option 1: Use Wrapper Script (Easiest)

```bash
# Run any command with venv activated automatically
./run.sh python scripts/collect_data.py --method apify --target 1
./run.sh python scripts/train_lora.py
./run.sh python -c "import torch; print(torch.__version__)"
```

### Option 2: Manual Activation (Recommended for Multiple Commands)

```bash
# Activate virtual environment
source qwen-lora-env/bin/activate

# Now run any commands
python scripts/collect_data.py --method apify --target 1
python scripts/train_lora.py

# Deactivate when done
deactivate
```

### Verify You're in the Virtual Environment

```bash
# Check Python location
which python
# Should show: /Users/sohailmo/Repos/experiments/cuda/Qwen3-8/qwen-lora-env/bin/python

# NOT: /usr/bin/python or /opt/homebrew/bin/python
```

---

## 📊 Data Collection

### Test Collection (1 tweet)
```bash
source qwen-lora-env/bin/activate
python scripts/collect_data.py --method apify --target 1
```

### Full Collection (1,000 pairs, ~$200-300)
```bash
source qwen-lora-env/bin/activate
python scripts/collect_data.py --method apify --target 1000
```

### View Collected Data
```bash
# See latest collection
ls -lh data/processed/

# Preview data
head -n 5 data/processed/training_data_*.jsonl
```

---

## 🚀 Training (RunPod)

See `RUNPOD_SETUP.md` for complete guide.

Quick version:
```bash
# 1. Launch pod on runpod.io (A6000 recommended)

# 2. On pod:
cd /workspace
git clone <your-repo>
cd Qwen3-8
pip install -r requirements-runpod.txt

# 3. Download model on pod (don't upload from Mac):
huggingface-cli download Qwen/Qwen3-8B --local-dir ./ --local-dir-use-symlinks False

# 4. Upload data from Mac (separate terminal):
scp -P <port> data/processed/training_data_*.jsonl root@<pod-ip>:/workspace/Qwen3-8/data/processed/

# 5. Train on pod:
python scripts/train_lora.py --config config/lora_config.yaml

# 6. Download results and STOP POD:
scp -P <port> -r root@<pod-ip>:/workspace/Qwen3-8/output/models/lora-v1 ./output/models/
```

---

## 🔍 Verification Commands

### Check Installation
```bash
source qwen-lora-env/bin/activate

python -c "
import transformers, peft, torch, bitsandbytes
print('✓ transformers:', transformers.__version__)
print('✓ peft:', peft.__version__)
print('✓ torch:', torch.__version__)
print('✓ bitsandbytes:', bitsandbytes.__version__)
"
```

### Check Data Collection Modules
```bash
source qwen-lora-env/bin/activate

python -c "
from src.data_collection import ApifyCollector, DataValidator, DataCleaner
print('✓ All data collection modules loaded')
"
```

### Check GPU (If Local)
```bash
source qwen-lora-env/bin/activate

python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## 📁 File Locations

```
Project Root: /Users/sohailmo/Repos/experiments/cuda/Qwen3-8/

Key Directories:
├── config/                  # YAML configuration files
├── data/
│   ├── raw/                # Raw collected data
│   ├── processed/          # Cleaned training data ← YOUR DATA HERE
│   └── cache/             # Temporary files
├── output/
│   ├── models/            # Trained LoRA adapters
│   ├── logs/              # Training/collection logs
│   └── figures/           # Plots and visualizations
├── scripts/               # Executable scripts
└── src/                   # Source code modules
```

---

## ⚙️ Configuration Files

### Data Collection: `config/data_collection_config.yaml`
```bash
nano config/data_collection_config.yaml

# Key settings:
# - collection.target_pairs: 1000
# - tweet_filters.min_likes: 200
# - reply_filters.min_likes: 5
```

### Training: `config/lora_config.yaml`
```bash
nano config/lora_config.yaml

# Key settings:
# - lora.rank: 16 (reduce to 8 if overfitting)
# - training.num_epochs: 2 (reduce to 1 if overfitting)
# - training.learning_rate: 2e-4
```

### Environment: `.env`
```bash
nano .env

# Required keys:
# - APIFY_API_TOKEN
# - WANDB_API_KEY
# - ANTHROPIC_API_KEY
```

---

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'X'"

**Problem**: Virtual environment not activated

**Solution**:
```bash
source qwen-lora-env/bin/activate
# Then run your command
```

### "APIFY_API_TOKEN not found"

**Problem**: `.env` file not configured

**Solution**:
```bash
cp .env.example .env
nano .env
# Add your API keys
```

### "No such file or directory: data/processed/training_data_*.jsonl"

**Problem**: Haven't collected data yet

**Solution**:
```bash
source qwen-lora-env/bin/activate
python scripts/collect_data.py --method apify --target 1
```

---

## 📚 Documentation Map

**Getting Started**:
- `GETTING_STARTED.md` - Setup and first steps
- `SETUP_STATUS.md` - What was installed
- `QUICK_REFERENCE.md` - This file

**Detailed Guides**:
- `lora_implementation_plan.md` - Complete 6-phase roadmap
- `RUNPOD_SETUP.md` - RunPod training guide
- `PROJECT_STRUCTURE.md` - Code organization

**Background**:
- `current_research.md` - Research synthesis
- `gameplan.md` - Realistic expectations

---

## 💡 Daily Workflow

### Morning (Mac, Free)
```bash
# Activate environment once
source qwen-lora-env/bin/activate

# Collect data
python scripts/collect_data.py --method apify --target 100

# Review quality
head -n 10 data/processed/training_data_*.jsonl

# Adjust configs if needed
nano config/data_collection_config.yaml
```

### Afternoon (RunPod, Paid)
```bash
# Launch RunPod A6000 pod
# Upload data
# Start training
# Monitor via W&B from Mac
# Download results
# STOP POD
```

### Evening (Mac, Free)
```bash
source qwen-lora-env/bin/activate

# Evaluate results
python scripts/evaluate_model.py --model output/models/lora-v1

# Analyze
jupyter notebook notebooks/03_results_analysis.ipynb

# Plan next iteration
```

---

## 🎯 Essential Commands Summary

```bash
# Setup (once)
./setup.sh

# Activate environment (every session)
source qwen-lora-env/bin/activate

# Collect data
python scripts/collect_data.py --method apify --target 1000

# Train (on RunPod, not Mac)
# See RUNPOD_SETUP.md

# Evaluate
python scripts/evaluate_model.py --model output/models/lora-v1

# Deactivate when done
deactivate
```

---

## 📞 Help

- **Activation issues**: Check `which python` shows venv path
- **API errors**: Verify `.env` has correct keys
- **Import errors**: Make sure you're in venv
- **Training**: See `RUNPOD_SETUP.md`

**Remember**: Always activate the virtual environment first! 🚀

