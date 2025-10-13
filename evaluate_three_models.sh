#!/bin/bash
# Evaluate Three Models: Baseline vs Moderate vs Aggressive
# This compares all three polychromic variants at once

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Three-Model Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Models:"
echo "  1. Baseline (λ=0, standard SFT)"
echo "  2. Moderate (λ=1.0)"
echo "  3. Aggressive (λ=2.0)"
echo ""
echo "This will generate comprehensive comparison:"
echo "  - Diversity metrics for all 3"
echo "  - Pass@k for all 3"
echo "  - Novel metrics (DER, collapse points)"
echo "  - LoRA parameter analysis"
echo "  - 4 publication figures"
echo ""

# Verify models exist
echo "Checking models..."
if [ ! -f "output/experiments/baseline/seed_42/adapter_model.safetensors" ]; then
    echo "✗ Baseline model not found!"
    exit 1
fi
echo "  ✓ Baseline"

if [ ! -f "output/experiments/polychromic_1.0/seed_42/adapter_model.safetensors" ]; then
    echo "✗ Moderate model not found!"
    exit 1
fi
echo "  ✓ Moderate (λ=1.0)"

if [ ! -f "output/experiments/polychromic_2.0/seed_42/adapter_model.safetensors" ]; then
    echo "✗ Aggressive model not found!"
    exit 1
fi
echo "  ✓ Aggressive (λ=2.0)"

echo ""
echo -e "${GREEN}Step 1/4: Comprehensive Evaluation (60-90 min)${NC}"
echo ""

# Note: This would need a modified evaluate_comprehensive.py to handle 3 models
# For now, we'll run pairwise comparisons

echo "Evaluating Baseline vs Moderate..."
python scripts/evaluation/evaluate_comprehensive.py \
    --baseline-lora output/experiments/baseline/seed_42 \
    --polychromic-lora output/experiments/polychromic_1.0/seed_42 \
    --test-data data/processed/test_data.jsonl \
    --output output/evaluation/baseline_vs_moderate/ \
    --n-generations 10 \
    --max-examples 500 \
    --skip-llm-judge

echo ""
echo "Evaluating Baseline vs Aggressive..."
python scripts/evaluation/evaluate_comprehensive.py \
    --baseline-lora output/experiments/baseline/seed_42 \
    --polychromic-lora output/experiments/polychromic_2.0/seed_42 \
    --test-data data/processed/test_data.jsonl \
    --output output/evaluation/baseline_vs_aggressive/ \
    --n-generations 10 \
    --max-examples 500 \
    --skip-llm-judge

echo ""
echo -e "${GREEN}Step 2/4: LoRA Parameter Analysis (10 min)${NC}"
echo ""

python scripts/analysis/analyze_lora_parameters.py \
    --baseline output/experiments/baseline/seed_42 \
    --polychromic output/experiments/polychromic_1.0/seed_42 \
    --output output/analysis/lora_baseline_vs_moderate

python scripts/analysis/analyze_lora_parameters.py \
    --baseline output/experiments/baseline/seed_42 \
    --polychromic output/experiments/polychromic_2.0/seed_42 \
    --output output/analysis/lora_baseline_vs_aggressive

echo ""
echo -e "${GREEN}Step 3/4: Novel Metrics (5 min)${NC}"
echo ""

python scripts/evaluation/evaluate_novel_metrics.py \
    --results-dir output/evaluation/baseline_vs_moderate/ \
    --output output/evaluation/baseline_vs_moderate/novel_metrics

python scripts/evaluation/evaluate_novel_metrics.py \
    --results-dir output/evaluation/baseline_vs_aggressive/ \
    --output output/evaluation/baseline_vs_aggressive/novel_metrics

echo ""
echo -e "${GREEN}Step 4/4: Create Comparison Report${NC}"
echo ""

# Create a combined report
python << 'EOF'
import json
from pathlib import Path

print("\n" + "="*60)
print("RESULTS SUMMARY: Three-Model Comparison")
print("="*60)

# Load results
moderate = json.load(open('output/evaluation/baseline_vs_moderate/summary.json'))
aggressive = json.load(open('output/evaluation/baseline_vs_aggressive/summary.json'))

# Extract key metrics
def extract_metrics(data, model_key):
    return {
        'distinct_2': data['diversity'][model_key]['distinct_2'],
        'self_bleu': data['diversity'][model_key]['self_bleu'],
        'pass_1': data['passk'][model_key]['1'],
        'pass_10': data['passk'][model_key]['10'],
    }

baseline_mod = extract_metrics(moderate, 'baseline_lora')
moderate_res = extract_metrics(moderate, 'polychromic_lora')
aggressive_res = extract_metrics(aggressive, 'polychromic_lora')

print("\nDiversity Metrics (Distinct-2, higher=better):")
print(f"  Baseline:   {baseline_mod['distinct_2']:.4f}")
print(f"  Moderate:   {moderate_res['distinct_2']:.4f} ({(moderate_res['distinct_2']/baseline_mod['distinct_2']-1)*100:+.1f}%)")
print(f"  Aggressive: {aggressive_res['distinct_2']:.4f} ({(aggressive_res['distinct_2']/baseline_mod['distinct_2']-1)*100:+.1f}%)")

print("\nSelf-BLEU (lower=better):")
print(f"  Baseline:   {baseline_mod['self_bleu']:.4f}")
print(f"  Moderate:   {moderate_res['self_bleu']:.4f} ({(moderate_res['self_bleu']/baseline_mod['self_bleu']-1)*100:+.1f}%)")
print(f"  Aggressive: {aggressive_res['self_bleu']:.4f} ({(aggressive_res['self_bleu']/baseline_mod['self_bleu']-1)*100:+.1f}%)")

print("\nPass@10:")
print(f"  Baseline:   {baseline_mod['pass_10']:.4f}")
print(f"  Moderate:   {moderate_res['pass_10']:.4f} ({(moderate_res['pass_10']/baseline_mod['pass_10']-1)*100:+.1f}%)")
print(f"  Aggressive: {aggressive_res['pass_10']:.4f} ({(aggressive_res['pass_10']/baseline_mod['pass_10']-1)*100:+.1f}%)")

print("\n" + "="*60)
print("Best model for publication:")

# Determine best
if moderate_res['distinct_2'] > baseline_mod['distinct_2'] * 1.15:
    print("✓ MODERATE shows strong improvement (>15%)")
elif aggressive_res['distinct_2'] > baseline_mod['distinct_2'] * 1.15:
    print("✓ AGGRESSIVE shows strong improvement (>15%)")
else:
    print("⚠️  Neither shows strong enough improvement for top venue")
    print("   Consider workshop paper or arXiv")

print("="*60)
EOF

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}✓ EVALUATION COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results available in:"
echo "  - output/evaluation/baseline_vs_moderate/"
echo "  - output/evaluation/baseline_vs_aggressive/"
echo "  - output/analysis/lora_*/"
echo ""
echo "Review W&B dashboard for training curves"
echo ""


