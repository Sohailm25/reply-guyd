#!/bin/bash
# Sync evaluation files to RunPod

RUNPOD_HOST="kz142xqyz00wbl-644111eb@ssh.runpod.io"
RUNPOD_KEY="$HOME/.ssh/id_ed25519"
RUNPOD_DIR="/workspace/Qwen3-8"

echo "========================================="
echo "Syncing Evaluation to RunPod"
echo "========================================="
echo ""

# Create directories on RunPod
echo "1. Creating directories on RunPod..."
ssh -T -i "$RUNPOD_KEY" "$RUNPOD_HOST" << 'EOF'
mkdir -p /workspace/Qwen3-8/output/experiments/baseline/seed_42
mkdir -p /workspace/Qwen3-8/output/experiments/polychromic_0.3/seed_42
mkdir -p /workspace/Qwen3-8/data/processed
mkdir -p /workspace/Qwen3-8/scripts
mkdir -p /workspace/Qwen3-8/src
mkdir -p /workspace/Qwen3-8/config
echo "✓ Directories created"
EOF

echo ""
echo "2. Syncing trained adapters..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  output/experiments/baseline/seed_42/ \
  $RUNPOD_HOST:$RUNPOD_DIR/output/experiments/baseline/seed_42/

rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  output/experiments/polychromic_0.3/seed_42/ \
  $RUNPOD_HOST:$RUNPOD_DIR/output/experiments/polychromic_0.3/seed_42/

echo ""
echo "3. Syncing test data..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  data/processed/test_data.jsonl \
  $RUNPOD_HOST:$RUNPOD_DIR/data/processed/

echo ""
echo "4. Syncing scripts and configs..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  scripts/ \
  $RUNPOD_HOST:$RUNPOD_DIR/scripts/

rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  src/ \
  $RUNPOD_HOST:$RUNPOD_DIR/src/

rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  config/ \
  $RUNPOD_HOST:$RUNPOD_DIR/config/

echo ""
echo "5. Syncing evaluation scripts..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  run_evaluation.sh \
  check_progress.sh \
  runpod_setup.sh \
  .env \
  requirements*.txt \
  $RUNPOD_HOST:$RUNPOD_DIR/

echo ""
echo "========================================="
echo "✓ Sync Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. SSH to RunPod: ssh -i $RUNPOD_KEY $RUNPOD_HOST"
echo "2. cd $RUNPOD_DIR"
echo "3. Run setup script (will be provided)"
echo ""

