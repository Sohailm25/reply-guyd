#!/bin/bash
# Sync evaluation files to RunPod using scp

RUNPOD_HOST="kz142xqyz00wbl-644111eb@ssh.runpod.io"
RUNPOD_KEY="$HOME/.ssh/id_ed25519"

echo "========================================="
echo "Syncing Evaluation to RunPod (via scp)"
echo "========================================="
echo ""

echo "Creating tar archives..."
tar -czf /tmp/qwen_evaluation.tar.gz \
  output/experiments/baseline/seed_42/ \
  output/experiments/polychromic_0.3/seed_42/ \
  data/processed/test_data.jsonl \
  scripts/ \
  src/ \
  config/ \
  run_evaluation.sh \
  check_progress.sh \
  runpod_setup.sh \
  .env \
  requirements*.txt

echo "Archive size: $(du -h /tmp/qwen_evaluation.tar.gz | cut -f1)"
echo ""

echo "Transferring to RunPod (this may take 5-10 minutes)..."
scp -i "$RUNPOD_KEY" /tmp/qwen_evaluation.tar.gz "$RUNPOD_HOST:/tmp/"

if [ $? -ne 0 ]; then
    echo "❌ Transfer failed!"
    rm /tmp/qwen_evaluation.tar.gz
    exit 1
fi

echo ""
echo "Extracting on RunPod..."
ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" << 'ENDSSH'
cd /workspace
mkdir -p Qwen3-8
cd Qwen3-8
tar -xzf /tmp/qwen_evaluation.tar.gz
rm /tmp/qwen_evaluation.tar.gz
echo "✓ Files extracted to /workspace/Qwen3-8"
ls -lh output/experiments/*/seed_42/*.safetensors 2>&1
ENDSSH

echo ""
echo "Cleaning up local temp file..."
rm /tmp/qwen_evaluation.tar.gz

echo ""
echo "========================================="
echo "✓ Sync Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  ssh -i $RUNPOD_KEY $RUNPOD_HOST"
echo "  cd /workspace/Qwen3-8"
echo "  ./runpod_setup.sh"
echo ""


