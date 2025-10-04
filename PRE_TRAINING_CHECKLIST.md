# ✅ Pre-Training Checklist

**Complete this checklist before starting training on RunPod**

**Date:** October 4, 2025  
**Status:** Ready for Phase 1 (SFT) Training

---

## 📊 Part 1: Data Verification (COMPLETE ✅)

### **Data Files Created**
- [x] `data/processed/test_data.jsonl` (500 examples)
- [x] `data/processed/train_full_sft.jsonl` (4,940 examples)
- [x] `data/processed/train_phase1_sft.jsonl` (2,500 examples)
- [x] `data/processed/train_phase2_grpo.jsonl` (2,440 examples)

### **Data Quality Verification**

Run these commands to verify:
```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
source qwen-lora-env/bin/activate

# Verify line counts
wc -l data/processed/test_data.jsonl  # Should be 500
wc -l data/processed/train_full_sft.jsonl  # Should be 4940
wc -l data/processed/train_phase1_sft.jsonl  # Should be 2500
wc -l data/processed/train_phase2_grpo.jsonl  # Should be 2440

# Sample data from each file
head -5 data/processed/test_data.jsonl | jq '.tweet, .reply' | head -10
head -5 data/processed/train_full_sft.jsonl | jq '.tweet, .reply' | head -10
```

**Expected:** All files exist, contain valid JSON, have correct line counts

- [x] All data files exist
- [x] Line counts correct
- [x] JSON format valid
- [x] Stratified by engagement (verified in splitting script)

---

## 📁 Part 2: Configuration Files (COMPLETE ✅)

### **Four Model Configs Created**
- [x] `config/experiments/baseline.yaml` (Model 1: Baseline SFT)
- [x] `config/experiments/polychromic_0.3.yaml` (Model 2: Polychromic SFT)
- [x] `config/experiments/baseline_grpo.yaml` (Model 3: Baseline → GRPO)
- [x] `config/experiments/polychromic_grpo.yaml` (Model 4: Polychromic → GRPO ⭐)

### **Config Verification**

Verify configs point to correct data files:
```bash
# Check baseline config
grep -A 2 "^data:" config/experiments/baseline.yaml
# Should show: train_file: data/processed/train_full_sft.jsonl

# Check polychromic config
grep -A 2 "^data:" config/experiments/polychromic_0.3.yaml
# Should show: train_file: data/processed/train_full_sft.jsonl

# Check baseline→GRPO config
grep "data_file" config/experiments/baseline_grpo.yaml
# Should show: phase1 → train_phase1_sft.jsonl, phase2 → train_phase2_grpo.jsonl

# Check polychromic→GRPO config
grep "data_file" config/experiments/polychromic_grpo.yaml
# Should show: phase1 → train_phase1_sft.jsonl, phase2 → train_phase2_grpo.jsonl
```

- [x] Model 1 config → train_full_sft.jsonl
- [x] Model 2 config → train_full_sft.jsonl
- [x] Model 3 config → phase1_sft.jsonl → phase2_grpo.jsonl
- [x] Model 4 config → phase1_sft.jsonl → phase2_grpo.jsonl
- [x] All configs → test_data.jsonl for evaluation

---

## 🖥️ Part 3: Local Environment (COMPLETE ✅)

### **Virtual Environment**
```bash
# Verify venv exists
ls -ld qwen-lora-env/
# Should show directory

# Activate and verify
source qwen-lora-env/bin/activate
which python
# Should show: /Users/sohailmo/Repos/experiments/cuda/Qwen3-8/qwen-lora-env/bin/python

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

- [x] Virtual environment exists
- [x] All dependencies installed
- [x] PyTorch available
- [x] Transformers available
- [x] PEFT available

---

## ☁️ Part 4: RunPod Setup (PENDING)

### **Before Starting RunPod**

**You need:**
1. RunPod account (signup at runpod.io)
2. Credit card or credits loaded
3. SSH key generated (if not already)

### **SSH Key Generation (If Needed)**
```bash
# Check if you have an SSH key
ls -la ~/.ssh/id_*.pub

# If not, generate one:
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for all prompts (default location, no passphrase)

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Add this to RunPod dashboard → Settings → SSH Keys
```

- [ ] RunPod account created
- [ ] Payment method added
- [ ] SSH key generated and added to RunPod

### **Create RunPod Instance**

**Recommended Configuration:**
- **GPU:** RTX A6000 (48GB VRAM) or A40 (48GB)
- **Template:** PyTorch 2.1+ or RunPod PyTorch
- **Container Disk:** 50GB minimum
- **Volume:** 50GB (optional but recommended for persistence)

**Cost Estimate:**
- A40/A6000: $0.39-0.79/hour
- Phase 1 training (all 4 models): ~20 hours = $8-16
- Keep pod running for analysis: add 2-4 hours = $1-3
- **Total Phase 1:** $9-19

- [ ] RunPod instance created
- [ ] GPU selected (A40 or A6000)
- [ ] Instance running and accessible

---

## 📦 Part 5: RunPod Environment Setup (PENDING)

### **Step 1: Connect to RunPod**
```bash
# Get connection details from RunPod dashboard
# SSH command format:
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### **Step 2: Clone Repository**
```bash
# On RunPod instance:
cd /workspace
git clone <your-github-repo-url> Qwen3-8

# Or if repo is private, use SSH:
git clone git@github.com:yourusername/Qwen3-8.git

cd Qwen3-8
```

### **Step 3: Install Dependencies**
```bash
# On RunPod:
pip install -r requirements-runpod.txt

# This installs:
# - transformers>=4.51.0
# - torch>=2.1.0
# - peft>=0.7.0
# - bitsandbytes>=0.41.0
# - accelerate>=0.24.0
# - datasets
# - wandb
# - sentence-transformers
# - And all other dependencies

# Should take 5-10 minutes
```

### **Step 4: Download Qwen3-8B Model**
```bash
# On RunPod (IMPORTANT: Download on pod, not upload from Mac!)
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /workspace/Qwen3-8/ \
  --local-dir-use-symlinks False

# This downloads ~16GB directly to pod (fast!)
# Takes 5-10 minutes on RunPod's fast connection
```

### **Step 5: Upload Data Files**
```bash
# From your Mac (separate terminal):
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8

# Upload all data files (total ~9MB, fast)
rsync -avz -e "ssh -p <runpod-port>" \
  data/processed/*.jsonl \
  root@<runpod-ip>:/workspace/Qwen3-8/data/processed/

# Verify upload (on RunPod):
ls -lh /workspace/Qwen3-8/data/processed/
wc -l /workspace/Qwen3-8/data/processed/*.jsonl
```

### **Step 6: Set Up W&B**
```bash
# On RunPod:
export WANDB_API_KEY=your_key_here
wandb login

# Or enter key when prompted
```

### **Step 7: Test Installation**
```bash
# On RunPod:
cd /workspace/Qwen3-8
python scripts/test_installation.py

# Should see:
# ✓ PyTorch available
# ✓ CUDA available
# ✓ Transformers available
# ✓ PEFT available
# ✓ W&B available
# ✓ Model files exist
# ✓ Data files exist
# ✓ ALL TESTS PASSED ✅
```

**Checklist:**
- [ ] Connected to RunPod via SSH
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Qwen3-8B model downloaded (16GB)
- [ ] Data files uploaded (9MB)
- [ ] W&B configured
- [ ] Test installation passed

---

## 🧪 Part 6: Quick Test Run (PENDING)

### **Run Quick Test (1 minute)**

Before full training, run a quick test to verify everything works:

```bash
# On RunPod:
cd /workspace/Qwen3-8

# Test baseline training (1 step only)
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml \
  --max-steps 1 \
  --output-dir /tmp/test_baseline

# Should complete in ~1 minute
# Check for errors
```

**Expected Output:**
```
✓ Config loaded
✓ Model loaded (quantized, 4-bit)
✓ LoRA adapters initialized
✓ Data loaded (4940 examples)
✓ Training started
✓ Step 1/1 complete
✓ Model saved to /tmp/test_baseline
```

**If errors occur:**
- Check error message
- Verify data files exist
- Verify model files exist
- Check GPU memory (should be sufficient for A40/A6000)

- [ ] Quick test run successful
- [ ] No errors
- [ ] Model can be loaded
- [ ] Data can be loaded
- [ ] Training step completes

---

## 📋 Part 7: Final Pre-Training Verification

### **Before Starting Full Training**

**Verify you have:**
- [x] ✅ Data: 500 test + 4940 train (full SFT) + 2500+2440 (two-phase)
- [x] ✅ Configs: 4 model configurations ready
- [x] ✅ Local: Virtual environment set up
- [ ] ⏳ RunPod: Instance running and accessible
- [ ] ⏳ RunPod: Environment set up completely
- [ ] ⏳ RunPod: Test run passed
- [ ] ⏳ W&B: Logged in and project created
- [ ] ⏳ Time: Ready to start (training takes ~20 hours for phase 1)

**Documents to Reference:**
- `DATA_TO_PAPER_COMPLETE_WORKFLOW.md` - Complete workflow guide
- `docs/runpod-training/RUNPOD_QUICKSTART.md` - RunPod detailed setup
- `README.md` - Experimental design overview

---

## 🚀 Part 8: Training Execution Order

### **Phase 1: SFT Training (Week 2)**

**Start with models 1 & 2 (full SFT, 4940 examples each):**

```bash
# Model 1: Baseline SFT (~4 hours)
python scripts/training/train_model.py \
  --config config/experiments/baseline.yaml

# Model 2: Polychromic SFT (~12 hours)
python scripts/training/train_model.py \
  --config config/experiments/polychromic_0.3.yaml
```

**Then train two-phase models, phase 1 only (2500 examples each):**

```bash
# Model 3: Baseline→GRPO Phase 1 (~2 hours)
python scripts/training/train_two_phase.py \
  --config config/experiments/baseline_grpo.yaml \
  --phase 1

# Model 4: Polychromic→GRPO Phase 1 (~6 hours)
python scripts/training/train_two_phase.py \
  --config config/experiments/polychromic_grpo.yaml \
  --phase 1
```

**Total Phase 1 Time:** ~24 hours  
**Total Phase 1 Cost:** $9-19 (depending on GPU)

### **Phase 2: GRPO Training (Later)**

**After Phase 1 complete and GRPO implementation ready:**

```bash
# Model 3: Baseline→GRPO Phase 2
python scripts/training/train_two_phase.py \
  --config config/experiments/baseline_grpo.yaml \
  --phase 2

# Model 4: Polychromic→GRPO Phase 2 ⭐
python scripts/training/train_two_phase.py \
  --config config/experiments/polychromic_grpo.yaml \
  --phase 2
```

---

## 📊 Expected Outputs

### **After Phase 1 Training Complete:**

You should have:
```
output/experiments/
├── baseline/
│   └── seed_42/
│       ├── adapter_model.safetensors
│       ├── diversity_dynamics.json
│       ├── diversity_trajectory.pdf
│       └── logs/
├── polychromic_0.3/
│   └── seed_42/
│       ├── adapter_model.safetensors
│       ├── diversity_dynamics.json
│       ├── diversity_trajectory.pdf
│       └── logs/
├── baseline_grpo/
│   └── seed_42/
│       └── phase1_sft/
│           ├── adapter_model.safetensors
│           ├── diversity_dynamics.json
│           └── logs/
└── polychromic_grpo/
    └── seed_42/
        └── phase1_sft/
            ├── adapter_model.safetensors
            ├── diversity_dynamics.json
            └── logs/
```

### **W&B Runs:**
- `baseline-seed42`
- `polychromic-0.3-seed42`
- `baseline-grpo-phase1-seed42`
- `polychromic-grpo-phase1-seed42`

---

## 🎯 Success Criteria

### **Phase 1 Training Success:**
- [ ] All 4 models trained successfully
- [ ] No OOM errors
- [ ] Training loss decreases
- [ ] Evaluation loss < 0.7
- [ ] Models saved correctly
- [ ] W&B logs complete
- [ ] Diversity dynamics tracked (for polychromic models)

### **Ready for Next Steps:**
- [ ] Phase 1 complete
- [ ] Models downloaded from RunPod
- [ ] Ready for evaluation or Phase 2

---

## 🚨 Troubleshooting

### **If Data Files Missing:**
```bash
# Re-run splitting script
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
source qwen-lora-env/bin/activate
python scripts/data/split_training_phases.py \
  --input data/processed/training_data_20251003_235332.jsonl \
  --output-dir data/processed
```

### **If RunPod Connection Fails:**
- Check pod is running (not stopped/terminated)
- Verify SSH port in connection command
- Try reconnecting after 1 minute

### **If Training OOM (Out of Memory):**
- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Ensure using 48GB GPU (A40/A6000)

### **If Training Too Slow:**
- This is expected for polychromic (3-4x slower than baseline)
- Polychromic generates 3 candidates per batch
- Consider reducing `n_generations` from 3 to 2 if needed

---

## ✅ Ready to Start?

**Complete checklist:**
- [x] Part 1: Data Verification ✅
- [x] Part 2: Configuration Files ✅
- [x] Part 3: Local Environment ✅
- [ ] Part 4: RunPod Setup ⏳
- [ ] Part 5: RunPod Environment Setup ⏳
- [ ] Part 6: Quick Test Run ⏳
- [ ] Part 7: Final Verification ⏳

**When all parts are ✅:**
→ Proceed to training execution (Part 8)!

---

**Created:** October 4, 2025  
**Status:** Ready for RunPod setup and training

**Next Step:** Follow Part 4 (RunPod Setup) when ready to start training

