#!/bin/bash
# Quick verification script to check if ready for training

echo "═══════════════════════════════════════════════════════════════════"
echo "    VERIFICATION: Ready for RunPod Training?"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check data files
echo "✓ Checking data files..."
files=(
  "data/processed/test_data.jsonl"
  "data/processed/train_full_sft.jsonl"
  "data/processed/train_phase1_sft.jsonl"
  "data/processed/train_phase2_grpo.jsonl"
)

all_exist=true
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    lines=$(wc -l < "$file")
    echo "  ✅ $file ($lines lines)"
  else
    echo "  ❌ $file MISSING"
    all_exist=false
  fi
done

echo ""
echo "✓ Checking config files..."
configs=(
  "config/experiments/baseline.yaml"
  "config/experiments/polychromic_0.3.yaml"
  "config/experiments/baseline_grpo.yaml"
  "config/experiments/polychromic_grpo.yaml"
)

for config in "${configs[@]}"; do
  if [ -f "$config" ]; then
    echo "  ✅ $config"
  else
    echo "  ❌ $config MISSING"
    all_exist=false
  fi
done

echo ""
echo "✓ Checking documentation..."
if [ -f "PRE_TRAINING_CHECKLIST.md" ]; then
  echo "  ✅ PRE_TRAINING_CHECKLIST.md"
else
  echo "  ❌ PRE_TRAINING_CHECKLIST.md MISSING"
  all_exist=false
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
if [ "$all_exist" = true ]; then
  echo "✅ ALL FILES PRESENT - READY FOR RUNPOD!"
  echo ""
  echo "Next steps:"
  echo "  1. Open PRE_TRAINING_CHECKLIST.md"
  echo "  2. Follow Parts 4-8"
  echo "  3. Start training!"
else
  echo "❌ SOME FILES MISSING - CHECK ABOVE"
  echo ""
  echo "Re-run data splitting:"
  echo "  source qwen-lora-env/bin/activate"
  echo "  python scripts/data/split_training_phases.py \\"
  echo "    --input data/processed/training_data_20251003_235332.jsonl"
fi
echo "═══════════════════════════════════════════════════════════════════"
