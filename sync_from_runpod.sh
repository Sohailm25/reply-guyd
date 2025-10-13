#!/bin/bash
# Download evaluation results from RunPod back to Mac

RUNPOD_HOST="kz142xqyz00wbl-644111eb@ssh.runpod.io"
RUNPOD_KEY="$HOME/.ssh/id_ed25519"
RUNPOD_DIR="/workspace/Qwen3-8"

echo "========================================="
echo "Downloading Results from RunPod"
echo "========================================="
echo ""

echo "1. Downloading evaluation results..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  $RUNPOD_HOST:$RUNPOD_DIR/output/evaluation/ \
  output/evaluation/

echo ""
echo "2. Downloading analysis results..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  $RUNPOD_HOST:$RUNPOD_DIR/output/analysis/ \
  output/analysis/

echo ""
echo "3. Downloading paper outputs..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  $RUNPOD_HOST:$RUNPOD_DIR/paper/ \
  paper/

echo ""
echo "4. Downloading log file..."
rsync -avz --progress -e "ssh -T -i $RUNPOD_KEY" \
  $RUNPOD_HOST:$RUNPOD_DIR/evaluation_run.log \
  ./

echo ""
echo "========================================="
echo "✓ Download Complete!"
echo "========================================="
echo ""
echo "Results available in:"
echo "  - Figures: paper/figures/"
echo "  - Data: paper/data/"
echo "  - Evaluation: output/evaluation/"
echo "  - Analysis: output/analysis/"
echo ""
echo "Verify results:"
echo "  ls -lh paper/figures/"
echo "  ls -lh paper/data/"
echo ""

