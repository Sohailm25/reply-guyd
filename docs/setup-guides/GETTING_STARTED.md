# Getting Started Guide

**Quick reference for running the Qwen3-8B LoRA fine-tuning project**

## ⚡ Quick Start (5 minutes)

```bash
# 1. Setup environment
./setup.sh

# 2. Configure API keys
cp .env.example .env
nano .env  # Add APIFY_API_TOKEN, ANTHROPIC_API_KEY, WANDB_API_KEY

# 3. Test installation
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Run test collection (10 tweets)
python scripts/collect_data.py --method apify --target 10
```

## 📋 Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] CUDA-capable GPU with 16GB+ VRAM (or RunPod account for training)
- [ ] ~20GB free disk space
- [ ] Apify account with API token ([apify.com](https://apify.com))
- [ ] Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- [ ] W&B account ([wandb.ai](https://wandb.ai))

## 🛠️ Detailed Setup

### Step 1: Environment Setup

```bash
# Navigate to project
cd Qwen3-8

# Run setup (installs all dependencies)
./setup.sh

# This will:
# ✓ Create virtual environment (qwen-lora-env)
# ✓ Install PyTorch, Transformers, PEFT, etc.
# ✓ Install evaluation tools
# ✓ Optionally install Unsloth (2x speedup)
# ✓ Optionally install twscrape (for scraping)
```

### Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your keys
nano .env  # or use any text editor
```

Required API keys:
```bash
# For data collection (REQUIRED)
APIFY_API_TOKEN=your_apify_token_here

# For evaluation (REQUIRED)  
ANTHROPIC_API_KEY=your_anthropic_key_here

# For experiment tracking (REQUIRED)
WANDB_API_KEY=your_wandb_key_here
```

Get API keys:
- **Apify**: Sign up at apify.com, go to Settings → API tokens
- **Anthropic**: console.anthropic.com → API Keys
- **W&B**: wandb.ai/authorize

### Step 3: Review Configuration Files

```bash
# Data collection settings
cat config/data_collection_config.yaml

# Training hyperparameters
cat config/lora_config.yaml
```

**Key settings to review:**
- `collection.target_pairs`: 1000 (adjust based on budget)
- `collection.search_queries`: Modify for your target domain
- `lora.rank`: 16 (start here, adjust if overfitting/underfitting)
- `training.num_epochs`: 2 (reduce to 1 if overfitting)

## 🎯 Phase-by-Phase Execution

### Phase 1: ✅ Infrastructure Setup (COMPLETE)

You're here! Infrastructure is ready to use.

### Phase 2: Data Collection (Current Phase)

**Budget**: $200-300 for ~1,000 training pairs

```bash
# Start small to test (10 tweets)
python scripts/collect_data.py --method apify --target 10

# Review output
cat data/processed/training_data_*.jsonl | head -n 5

# If satisfied, run full collection (1,000 pairs)
python scripts/collect_data.py --method apify --target 1000
```

**Expected duration**: 2-4 hours for 1,000 pairs (includes rate limiting)

**Manual review** (CRITICAL):
```bash
# Review collected data
python -c "
import json
with open('data/processed/training_data_<timestamp>.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        pair = json.loads(line)
        print(f'Tweet: {pair[\"tweet\"][:60]}...')
        print(f'Reply: {pair[\"reply\"][:60]}...')
        print(f'Likes: {pair[\"reply_likes\"]}\n')
"
```

**Decision point**: Does data look good? If yes → Phase 3. If no → adjust filters and recollect.

### Phase 3: Training (Coming Soon)

Training scripts will be created in this phase.

**Preview**:
```bash
# Train LoRA adapters
python scripts/train_lora.py \
    --config config/lora_config.yaml \
    --data data/processed/training_data_<timestamp>.jsonl

# Monitor in W&B
wandb login
# Then open: https://wandb.ai/<username>/qwen3-twitter-lora
```

### Phase 4: Evaluation (Coming Soon)

Evaluation framework will be implemented in this phase.

**Preview**:
```bash
# Evaluate fine-tuned vs base
python scripts/evaluate_model.py \
    --base-model ./ \
    --lora-model output/models/lora-v1 \
    --test-data data/processed/test_set.jsonl
```

## 🔍 Testing Your Setup

### Test 1: Python Environment

```bash
python3 -c "
import sys
import torch
import transformers
import peft

print('✓ Python version:', sys.version)
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ PEFT:', peft.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
"
```

Expected output:
```
✓ Python version: 3.x.x
✓ PyTorch: 2.x.x
✓ Transformers: 4.51.x
✓ PEFT: 0.7.x
✓ CUDA available: True
✓ GPU: NVIDIA ...
```

### Test 2: Model Loading

```bash
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Loading Qwen3-8B tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('./')
print('✓ Tokenizer loaded')

print('Loading model in 4-bit...')
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    './',
    quantization_config=quant_config,
    device_map='auto'
)
print('✓ Model loaded')
print(f'✓ Model memory: ~{torch.cuda.memory_allocated() / 1e9:.2f} GB')
"
```

### Test 3: Data Collection (Dry Run)

```bash
# Test Apify connection (collects 1 tweet)
python scripts/collect_data.py --method apify --target 1

# Check output
ls -lh data/processed/
```

## 📊 Monitoring Progress

### Weights & Biases Dashboard

1. Login to W&B:
```bash
wandb login
# Paste your API key when prompted
```

2. View experiments:
   - Go to https://wandb.ai/
   - Navigate to project: `qwen3-twitter-lora`
   - View runs, metrics, and artifacts

### Logs

All operations log to files:
```bash
# Data collection logs
tail -f output/logs/data_collection_*.log

# Training logs (Phase 3)
tail -f output/logs/training_*.log
```

## 🐛 Troubleshooting

### "APIFY_API_TOKEN not found"
```bash
# Check .env file exists
ls -la .env

# Verify token is set
cat .env | grep APIFY_API_TOKEN

# If missing, add it:
echo "APIFY_API_TOKEN=your_token_here" >> .env
```

### "CUDA out of memory"
```bash
# Use 4-bit quantization (should be enabled by default)
# If still OOM, reduce batch size in config/lora_config.yaml:
# training.per_device_train_batch_size: 2  # instead of 4
```

### "No module named 'src'"
```bash
# Run from project root
cd Qwen3-8
python scripts/collect_data.py ...
```

### Collection returns no data
```bash
# Check Apify token is valid
# Try simpler query:
python scripts/collect_data.py --method apify --target 5

# Review logs for errors
cat output/logs/data_collection_*.log
```

## 💡 Tips for Success

### Data Collection
1. **Start small**: Test with 10-50 tweets first
2. **Review early**: Look at first batch before collecting all 1,000
3. **Diverse queries**: Use all 7 search queries for topic diversity
4. **Manual review**: Spend 1-2 hours reviewing quality (worth it!)

### Training (Phase 3)
1. **Monitor from start**: Watch W&B dashboard for first 100 steps
2. **Early stopping**: Don't train longer if loss plateaus
3. **Save checkpoints**: Every 50 steps for small datasets
4. **Expect iteration**: First attempt has ~30% success rate

### Budget Management
- Apify: ~$200-300 for 1,000 pairs (primary cost)
- Training: ~$10-20 on RunPod (cheap!)
- Evaluation: ~$50-150 for Claude API
- Total realistic: $400-600

## 📚 Next Steps

After setup is complete:

1. **Read documentation**:
   - `lora_implementation_plan.md` - Detailed roadmap
   - `gameplan.md` - Realistic expectations
   - `PROJECT_STRUCTURE.md` - Code organization

2. **Collect data** (Phase 2):
   ```bash
   python scripts/collect_data.py --method apify --target 1000
   ```

3. **Manual review** (ESSENTIAL):
   - Review every example for quality
   - Budget 5-7 hours for 1,000 pairs
   - Remove spam, off-topic, or low-quality pairs

4. **Wait for Phase 3 implementation** (training scripts coming soon)

## 🆘 Getting Help

1. **Documentation**: Check `PROJECT_STRUCTURE.md` for troubleshooting
2. **Logs**: Review `output/logs/*.log` for error details
3. **Configuration**: Verify settings in `config/*.yaml`
4. **Research**: Consult `current_research.md` for methodology

## ✅ Ready to Start?

Checklist before data collection:
- [ ] `./setup.sh` completed successfully
- [ ] `.env` configured with all API keys
- [ ] Test collection ran successfully (1-10 tweets)
- [ ] W&B login working
- [ ] Reviewed and understood data collection config

If all checked, you're ready:
```bash
python scripts/collect_data.py --method apify --target 1000
```

Good luck! 🚀


