#!/bin/bash
# Sync evaluation files to RunPod using tar (more reliable than rsync for RunPod)

RUNPOD_HOST="kz142xqyz00wbl-644111eb@ssh.runpod.io"
RUNPOD_KEY="$HOME/.ssh/id_ed25519"

echo "========================================="
echo "Syncing Evaluation to RunPod (via tar)"
echo "========================================="
echo ""

echo "1. Creating tar archives..."

# Create tar of adapters
echo "  - Baseline adapter..."
tar -czf /tmp/baseline_adapter.tar.gz -C output/experiments/baseline/seed_42 .

echo "  - Polychromic adapter..."
tar -czf /tmp/polychromic_adapter.tar.gz -C output/experiments/polychromic_0.3/seed_42 .

# Create tar of scripts and source
echo "  - Scripts and source code..."
tar -czf /tmp/scripts_src.tar.gz scripts/ src/ config/ run_evaluation.sh check_progress.sh runpod_setup.sh .env requirements*.txt

# Create tar of test data
echo "  - Test data..."
tar -czf /tmp/test_data.tar.gz -C data/processed test_data.jsonl

echo ""
echo "2. Transferring files to RunPod..."

# Transfer and extract in one go
echo "  - Baseline adapter (~175MB)..."
cat /tmp/baseline_adapter.tar.gz | ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" \
  "mkdir -p /workspace/Qwen3-8/output/experiments/baseline/seed_42 && cd /workspace/Qwen3-8/output/experiments/baseline/seed_42 && tar -xzf -"

echo "  - Polychromic adapter (~175MB)..."
cat /tmp/polychromic_adapter.tar.gz | ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" \
  "mkdir -p /workspace/Qwen3-8/output/experiments/polychromic_0.3/seed_42 && cd /workspace/Qwen3-8/output/experiments/polychromic_0.3/seed_42 && tar -xzf -"

echo "  - Scripts and source..."
cat /tmp/scripts_src.tar.gz | ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" \
  "mkdir -p /workspace/Qwen3-8 && cd /workspace/Qwen3-8 && tar -xzf -"

echo "  - Test data..."
cat /tmp/test_data.tar.gz | ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" \
  "mkdir -p /workspace/Qwen3-8/data/processed && cd /workspace/Qwen3-8/data && tar -xzf -"

echo ""
echo "3. Cleaning up local temp files..."
rm /tmp/baseline_adapter.tar.gz /tmp/polychromic_adapter.tar.gz /tmp/scripts_src.tar.gz /tmp/test_data.tar.gz

echo ""
echo "4. Verifying transfer..."
ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" "cd /workspace/Qwen3-8 && echo '✓ Files in /workspace/Qwen3-8:' && ls -lh output/experiments/*/seed_42/*.safetensors 2>&1 && ls -lh data/processed/test_data.jsonl 2>&1 && ls -lh run_evaluation.sh 2>&1"

echo ""
echo "========================================="
echo "✓ Sync Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. SSH to RunPod: ssh -i $RUNPOD_KEY $RUNPOD_HOST"
echo "2. cd /workspace/Qwen3-8"
echo "3. Run: ./runpod_setup.sh"
echo ""


