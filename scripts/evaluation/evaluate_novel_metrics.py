#!/usr/bin/env python3
"""
Evaluate Models Using Novel Metrics

Applies the three novel metrics (USQ, DER, Collapse Point) to evaluate models.

Usage:
    python scripts/evaluation/evaluate_novel_metrics.py \
        --models baseline:output/experiments/baseline/seed_42 \
                 polychromic:output/experiments/polychromic_0.3/seed_42 \
        --test-data data/processed/test_data.jsonl \
        --output output/evaluation/novel_metrics \
        --k-values 1,3,5,10
"""

import argparse
import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.novel_metrics import (
    compute_diversity_efficiency_ratio,
    find_collapse_point,
    analyze_diversity_efficiency,
    length_based_selector,
    diversity_based_selector,
    first_candidate_selector
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_passk_results(results_dir: str) -> Dict:
    """Load Pass@k results from evaluation directory."""
    results_path = Path(results_dir) / 'passk_results.json'
    
    if not results_path.exists():
        logger.error(f"Pass@k results not found at {results_path}")
        logger.error("Run evaluate_comprehensive.py first to generate Pass@k results")
        return {}
    
    with open(results_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models using novel metrics (USQ, DER, Collapse Point)"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory with existing Pass@k results (from evaluate_comprehensive.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for novel metrics results'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Novel Metrics Evaluation")
    logger.info("="*60)
    logger.info("📊 Metrics:")
    logger.info("  1. User Selection Quality (USQ)")
    logger.info("  2. Diversity Efficiency Ratio (DER)")
    logger.info("  3. Collapse Point Analysis")
    logger.info("="*60)
    
    # Load Pass@k results
    logger.info(f"\nLoading Pass@k results from {args.results_dir}...")
    passk_results = load_passk_results(args.results_dir)
    
    if not passk_results:
        logger.error("No Pass@k results found. Exiting.")
        sys.exit(1)
    
    logger.info(f"✅ Loaded Pass@k results for {len(passk_results)} models")
    for model_name in passk_results.keys():
        logger.info(f"  • {model_name}")
    
    # Analyze diversity efficiency
    logger.info("\n" + "="*60)
    logger.info("Computing Novel Metrics...")
    logger.info("="*60)
    
    results = analyze_diversity_efficiency(
        passk_curves=passk_results,
        save_dir=args.output
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 SUMMARY")
    logger.info("="*60)
    
    for model_name, data in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Collapse Point: k={data['collapse_point']}")
        logger.info(f"  DER@5: {data['der'].get(5, 'N/A'):.3f}" if 5 in data['der'] else "  DER@5: N/A")
        logger.info(f"  DER@10: {data['der'].get(10, 'N/A'):.3f}" if 10 in data['der'] else "  DER@10: N/A")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Novel Metrics Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {args.output}")
    logger.info(f"  • diversity_efficiency_analysis.json")
    logger.info(f"  • der_comparison.pdf")
    logger.info(f"  • collapse_points.pdf")
    logger.info("="*60)


if __name__ == "__main__":
    main()

