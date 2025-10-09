# Manual Transfer Guide for RunPod

## Issue

RunPod's SSH connection doesn't support standard file transfer methods (rsync, scp). We need to use an alternative approach.

## Option 1: Use RunPod's Web Terminal & File Upload (Recommended)

### Step 1: Access RunPod Web Terminal
1. Go to RunPod dashboard: https://www.runpod.io/console/pods
2. Find your pod and click "Connect"
3. Choose "Start Web Terminal" or "Jupyter Lab"

### Step 2: Prepare Files on Mac
```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8

# Create minimal tar with only essential files
tar -czf ~/Desktop/qwen_eval_minimal.tar.gz \
  --exclude='*/checkpoint-*' \
  output/experiments/baseline/seed_42/adapter_model.safetensors \
  output/experiments/baseline/seed_42/adapter_config.json \
  output/experiments/baseline/seed_42/tokenizer* \
  output/experiments/baseline/seed_42/*.json \
  output/experiments/polychromic_0.3/seed_42/adapter_model.safetensors \
  output/experiments/polychromic_0.3/seed_42/adapter_config.json \
  output/experiments/polychromic_0.3/seed_42/tokenizer* \
  output/experiments/polychromic_0.3/seed_42/*.json \
  data/processed/test_data.jsonl \
  scripts/ \
  src/ \
  config/ \
  run_evaluation.sh \
  check_progress.sh \
  runpod_setup.sh \
  .env \
  requirements-runpod.txt

echo "Archive created: ~/Desktop/qwen_eval_minimal.tar.gz"
du -h ~/Desktop/qwen_eval_minimal.tar.gz
```

### Step 3: Upload via Web Interface
1. In RunPod web terminal or Jupyter:
   - Upload `qwen_eval_minimal.tar.gz` to `/workspace`
2. Extract:
```bash
cd /workspace
mkdir -p Qwen3-8
cd Qwen3-8
tar -xzf ../qwen_eval_minimal.tar.gz
rm ../qwen_eval_minimal.tar.gz
```

### Step 4: Run Setup
```bash
cd /workspace/Qwen3-8
chmod +x runpod_setup.sh check_progress.sh run_evaluation.sh
./runpod_setup.sh
```

---

## Option 2: Use SFTP (If available)

Try SFTP instead of SCP:
```bash
sftp -i ~/.ssh/id_ed25519 kz142xqyz00wbl-644111eb@ssh.runpod.io
# Then:
cd /workspace
mkdir Qwen3-8
put qwen_eval_minimal.tar.gz
```

---

## Option 3: GitHub as intermediary

If you have a private GitHub repo:

### On Mac:
```bash
# Create minimal archive
./create_minimal_archive.sh  # (will create this script)

# Upload to a private repo or Google Drive
```

### On RunPod:
```bash
cd /workspace/Qwen3-8
# Download from GitHub/Drive
wget YOUR_URL -O qwen_eval.tar.gz
tar -xzf qwen_eval.tar.gz
```

---

## Option 4: Run evaluation on Mac instead

Given the transfer difficulties, you could also:
1. Keep laptop awake and plugged in
2. Use `caffeinate` to prevent sleep:
```bash
caffeinate -d ./run_evaluation.sh 2>&1 | tee evaluation_run.log
```
3. Will take 12-24 hours on CPU but requires no transfer

---

## Recommendation

**Best approach:** Use RunPod's web terminal file upload (Option 1)
- Most reliable
- No SSH issues
- Takes ~5 minutes once archive is created

Let me know which option you'd like to try!

