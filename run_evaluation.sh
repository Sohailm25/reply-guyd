#!/bin/bash
# Comprehensive Evaluation Script for Tmux
# This runs all evaluation steps sequentially

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Scientific Evaluation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Started at: $(date)"
echo ""

# Activate virtual environment
source qwen-lora-env/bin/activate

# Load environment variables
export $(grep -v '^#' .env | xargs)

echo -e "${GREEN}Step 1: Comprehensive Evaluation${NC}"
echo "This will take 1-2 hours..."
echo ""

python scripts/evaluation/evaluate_comprehensive.py \
  --baseline-lora output/experiments/baseline/seed_42 \
  --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
  --test-data data/processed/test_data.jsonl \
  --output output/evaluation/ \
  --n-generations 10 \
  --max-examples 500 \
  --anthropic-key ${ANTHROPIC_API_KEY}

echo ""
echo -e "${GREEN}✓ Step 1 Complete: Comprehensive Evaluation${NC}"
echo ""

echo -e "${GREEN}Step 2: LoRA Parameter Analysis${NC}"
echo "This will take ~10 minutes..."
echo ""

python scripts/analysis/analyze_lora_parameters.py \
  --baseline output/experiments/baseline/seed_42 \
  --polychromic output/experiments/polychromic_0.3/seed_42 \
  --output output/analysis/lora_comparison

echo ""
echo -e "${GREEN}✓ Step 2 Complete: LoRA Analysis${NC}"
echo ""

echo -e "${GREEN}Step 3: Novel Metrics Evaluation${NC}"
echo "This will take ~5 minutes..."
echo ""

python scripts/evaluation/evaluate_novel_metrics.py \
  --results-dir output/evaluation/ \
  --output output/evaluation/novel_metrics

echo ""
echo -e "${GREEN}✓ Step 3 Complete: Novel Metrics${NC}"
echo ""

echo -e "${GREEN}Step 4: Organizing Results${NC}"
echo ""

# Create paper directory structure
mkdir -p paper/figures paper/data

# Copy figures with paper names
if [ -f output/evaluation/pareto_frontier.pdf ]; then
  cp output/evaluation/pareto_frontier.pdf paper/figures/figure3_pareto_frontier.pdf
  echo "  ✓ Copied Figure 3: Pareto frontier"
fi

if [ -f output/analysis/lora_comparison/lora_comparison.pdf ]; then
  cp output/analysis/lora_comparison/lora_comparison.pdf paper/figures/figure4_lora_analysis.pdf
  echo "  ✓ Copied Figure 4: LoRA analysis"
fi

if [ -f output/evaluation/novel_metrics/der_comparison.pdf ]; then
  cp output/evaluation/novel_metrics/der_comparison.pdf paper/figures/figure5_der_comparison.pdf
  echo "  ✓ Copied Figure 5: DER comparison"
fi

if [ -f output/evaluation/novel_metrics/collapse_points.pdf ]; then
  cp output/evaluation/novel_metrics/collapse_points.pdf paper/figures/figure6_collapse_points.pdf
  echo "  ✓ Copied Figure 6: Collapse points"
fi

# Copy numerical results
cp output/evaluation/*.json paper/data/ 2>/dev/null || true
cp output/analysis/lora_comparison/*.json paper/data/ 2>/dev/null || true
cp output/evaluation/novel_metrics/*.json paper/data/ 2>/dev/null || true

echo "  ✓ Copied all JSON data files"

echo ""
echo -e "${GREEN}✓ Step 4 Complete: Results Organized${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EVALUATION COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Completed at: $(date)"
echo ""
echo "Results available in:"
echo "  - Figures: paper/figures/"
echo "  - Data: paper/data/"
echo ""
echo "Next steps:"
echo "  1. Review figures in paper/figures/"
echo "  2. Extract key numbers from JSON files in paper/data/"
echo "  3. Start writing paper sections"
echo ""

