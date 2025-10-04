# 🚀 Quick Start: Sync GRPO to RunPod

## ⚡ Fastest Method (One Command!)

```bash
# On your Mac
./sync_grpo_to_runpod.sh <RUNPOD_IP> <SSH_PORT>

# Example:
./sync_grpo_to_runpod.sh 123.45.67.89 12345
```

That's it! All GRPO files will sync automatically.

---

## 📦 What Gets Synced

**New Files (70 KB total):**
- `src/training/heuristic_reward.py` - Reward function (16 KB)
- `src/training/grpo_trainer.py` - Complete GRPO trainer (21 KB)
- `scripts/test_grpo.py` - Test suite (6 KB)
- `scripts/data/split_training_phases.py` - Data splitter (13 KB)

**Modified Files:**
- `src/training/__init__.py` - Added GRPO exports
- `scripts/training/train_model.py` - Added checkpoint loading

**Configs:**
- `config/experiments/grpo_from_baseline.yaml`
- `config/experiments/grpo_from_polychromic.yaml`

---

## ✅ Verify on RunPod

```bash
# SSH to RunPod
ssh -p <PORT> root@<IP>

# Test GRPO
cd /workspace/Qwen3-8
python scripts/test_grpo.py

# Should see:
# ✅ TEST 1 PASSED: Heuristic Reward Function Working!
# ✅ TEST 2 PASSED: GRPO Configuration Valid!
# ✅ TEST 3 PASSED: Batch Reward Computation Working!
```

---

## 🎯 Start Training

```bash
# Small test (1-2 hours)
python scripts/training/train_model.py \
  --config config/experiments/grpo_from_baseline.yaml \
  --data data/processed/training_data_*.jsonl \
  --checkpoint output/experiments/baseline/seed_42

# Watch for:
# "✅ Checkpoint loaded successfully!"
# "✅ Initialized GRPOTrainer"
```

---

## 🔧 Alternative: Git Method

```bash
# On Mac: Commit and push
git add src/training/*.py scripts/*.py
git commit -m "Add GRPO implementation"
git push origin feature/initialization

# On RunPod: Pull changes
cd /workspace/Qwen3-8
git pull origin feature/initialization
```

---

## 📊 Monitor Training

- **Console:** Watch for loss/reward metrics
- **W&B:** Check `train/policy_loss`, `train/avg_reward`, `train/kl_penalty`
- **Logs:** `tail -f training.log`

---

## ⚠️ Troubleshooting

**Imports fail?**
```bash
pip install sentence-transformers
```

**Checkpoint not found?**
```bash
ls -lh output/experiments/baseline/seed_42/adapter_model.safetensors
# Should see ~175MB file
```

**Out of memory?**
```yaml
# Edit config:
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

---

**Ready to train GRPO on RunPod! 🚀**
