#!/bin/bash
# Train Two Polychromic Variants Back-to-Back
# This trains λ=1.0 (moderate) and λ=2.0 (aggressive) sequentially

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Polychromic Variants Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "This will train TWO models sequentially:"
echo "  1. Polychromic λ=1.0 (Moderate) - ~12 hours"
echo "  2. Polychromic λ=2.0 (Aggressive) - ~12 hours"
echo ""
echo "Total time: ~24 hours"
echo "Total cost: ~\$20 on RunPod RTX A6000"
echo ""
echo "Started: $(date)"
echo ""

# Verify we're in the right directory
if [ ! -f "scripts/training/train_model.py" ]; then
    echo -e "${RED}Error: Not in Qwen3-8 directory!${NC}"
    exit 1
fi

# Verify data exists
if [ ! -f "data/processed/train_full_sft.jsonl" ]; then
    echo -e "${RED}Error: Training data not found!${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 1/2: Training Moderate (λ=1.0)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Logging to: output/experiments/polychromic_1.0/seed_42/training.log"
echo ""

python scripts/training/train_model.py \
    --config config/experiments/polychromic_1.0_moderate.yaml \
    --data data/processed/train_full_sft.jsonl 2>&1 | tee output/experiments/polychromic_1.0/seed_42/training.log

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Moderate training complete!${NC}"
    echo -e "${GREEN}  Log: output/experiments/polychromic_1.0/seed_42/training.log${NC}"
    echo ""
else
    echo -e "${RED}✗ Moderate training failed!${NC}"
    echo -e "${RED}  Check: output/experiments/polychromic_1.0/seed_42/training.log${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Step 2/2: Training Aggressive (λ=2.0)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Logging to: output/experiments/polychromic_2.0/seed_42/training.log"
echo ""

python scripts/training/train_model.py \
    --config config/experiments/polychromic_2.0_aggressive.yaml \
    --data data/processed/train_full_sft.jsonl 2>&1 | tee output/experiments/polychromic_2.0/seed_42/training.log

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Aggressive training complete!${NC}"
    echo -e "${GREEN}  Log: output/experiments/polychromic_2.0/seed_42/training.log${NC}"
    echo ""
else
    echo -e "${RED}✗ Aggressive training failed!${NC}"
    echo -e "${RED}  Check: output/experiments/polychromic_2.0/seed_42/training.log${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}✓ ALL TRAINING COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Completed: $(date)"
echo ""
echo "Models trained:"
echo "  1. output/experiments/polychromic_1.0/seed_42/"
echo "  2. output/experiments/polychromic_2.0/seed_42/"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: ./evaluate_three_models.sh"
echo "  2. Review W&B dashboard for training curves"
echo "  3. Compare diversity metrics across all 3 models"
echo ""

