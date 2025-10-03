#!/bin/bash
# Train all seeds for a given experiment configuration
# Usage: bash scripts/training/train_all_seeds.sh baseline
#        bash scripts/training/train_all_seeds.sh polychromic_0.3

set -e  # Exit on error

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: bash scripts/training/train_all_seeds.sh <experiment_name>"
    echo "Examples:"
    echo "  bash scripts/training/train_all_seeds.sh baseline"
    echo "  bash scripts/training/train_all_seeds.sh polychromic_0.3"
    exit 1
fi

SEEDS=(42 123 456)

echo "═══════════════════════════════════════════════════════════════"
echo "  Training ${EXPERIMENT} with multiple seeds"
echo "═══════════════════════════════════════════════════════════════"
echo "Seeds: ${SEEDS[@]}"
echo "This will train 3 models sequentially."
echo ""

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Training ${EXPERIMENT} with seed ${seed}"
    echo "───────────────────────────────────────────────────────────────"
    
    CONFIG_FILE="config/experiments/${EXPERIMENT}_seed${seed}.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠️  Config file not found: $CONFIG_FILE"
        echo "Skipping seed ${seed}..."
        continue
    fi
    
    echo "Using config: $CONFIG_FILE"
    echo ""
    
    python scripts/training/train_model.py \
        --config "$CONFIG_FILE" \
        --data data/processed/training_data_final.jsonl
    
    echo ""
    echo "✅ Completed seed ${seed}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ All seeds completed for ${EXPERIMENT}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Run evaluation on all seeds:"
echo "   python scripts/evaluation/evaluate_comprehensive.py \\"
echo "     --baseline-lora output/experiments/baseline/seed_42 \\"
echo "     --polychromic-lora output/experiments/polychromic_0.3/seed_42 \\"
echo "     ..."
echo ""
echo "2. Aggregate results across seeds:"
echo "   python scripts/analysis/aggregate_seeds.py \\"
echo "     --experiment ${EXPERIMENT} \\"
echo "     --seeds 42 123 456"

