#!/bin/bash
#
# Sync GRPO Implementation to RunPod
#
# Usage:
#   ./sync_grpo_to_runpod.sh <runpod-host> <port>
#
# Example:
#   ./sync_grpo_to_runpod.sh 123.456.789.10 12345
#

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <runpod-host> <port>"
    echo "Example: $0 123.456.789.10 12345"
    exit 1
fi

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_USER="root"
REMOTE_DIR="/workspace/Qwen3-8"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "           Syncing GRPO Implementation to RunPod"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Target: ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Remote directory: ${REMOTE_DIR}"
echo ""

# Sync individual files
echo "📦 Syncing GRPO files..."
echo ""

# Core training files (NEW)
echo "  → src/training/heuristic_reward.py"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  src/training/heuristic_reward.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/src/training/

echo "  → src/training/grpo_trainer.py"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  src/training/grpo_trainer.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/src/training/

# Modified files
echo "  → src/training/__init__.py (updated)"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  src/training/__init__.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/src/training/

echo "  → scripts/training/train_model.py (updated)"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  scripts/training/train_model.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/scripts/training/

# Test and utility scripts (NEW)
echo "  → scripts/test_grpo.py"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  scripts/test_grpo.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/scripts/

echo "  → scripts/data/split_training_phases.py"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  scripts/data/split_training_phases.py \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/scripts/data/

# Config files (optional, but good to have)
echo "  → config/experiments/grpo_*.yaml"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  config/experiments/grpo_*.yaml \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/config/experiments/

# Documentation (optional)
echo "  → GRPO documentation"
rsync -avz -e "ssh -p ${RUNPOD_PORT}" \
  GRPO_*.md \
  ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/ 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Sync complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps on RunPod:"
echo ""
echo "  1. SSH into RunPod:"
echo "     ssh -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST}"
echo ""
echo "  2. Navigate to project:"
echo "     cd ${REMOTE_DIR}"
echo ""
echo "  3. Test GRPO implementation:"
echo "     python scripts/test_grpo.py"
echo ""
echo "  4. Start GRPO training:"
echo "     python scripts/training/train_model.py \\"
echo "       --config config/experiments/grpo_from_baseline.yaml \\"
echo "       --data data/processed/training_data_*.jsonl \\"
echo "       --checkpoint output/experiments/baseline/seed_42"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

