#!/usr/bin/env python3
"""
Analyze Diversity Dynamics from Training Runs

Compare how diversity evolved during training across different models.

Usage:
    python scripts/analysis/analyze_dynamics.py \
        --baseline output/experiments/baseline/seed_42/diversity_dynamics.json \
        --polychromic output/experiments/polychromic_0.3/seed_42/diversity_dynamics.json \
        --output output/figures/dynamics_comparison.pdf
"""

import argparse
import json
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.diversity_dynamics import (
    DiversitySnapshot, 
    compare_diversity_trajectories
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dynamics_history(filepath: str) -> list:
    """Load diversity dynamics history from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    history = [DiversitySnapshot(**snap) for snap in data['snapshots']]
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare diversity dynamics from training runs"
    )
    parser.add_argument(
        '--baseline', 
        type=str, 
        help='Path to baseline model diversity_dynamics.json'
    )
    parser.add_argument(
        '--polychromic', 
        type=str,
        help='Path to polychromic model diversity_dynamics.json'
    )
    parser.add_argument(
        '--grpo-baseline',
        type=str,
        help='Path to baseline→GRPO model diversity_dynamics.json (optional)'
    )
    parser.add_argument(
        '--grpo-polychromic',
        type=str,
        help='Path to polychromic→GRPO model diversity_dynamics.json (optional)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Output path for comparison plot'
    )
    parser.add_argument(
        '--metric', 
        type=str, 
        default='distinct_2',
        choices=['self_bleu', 'distinct_1', 'distinct_2', 
                'distinct_3', 'semantic_diversity', 'output_entropy'],
        help='Which metric to plot (default: distinct_2)'
    )
    parser.add_argument(
        '--phase-boundary',
        type=int,
        help='Step where Phase 1 ends and Phase 2 begins (for two-phase training)'
    )
    
    args = parser.parse_args()
    
    # Load histories
    histories = {}
    
    if args.baseline:
        logger.info(f"Loading baseline dynamics from {args.baseline}")
        histories['Baseline'] = load_dynamics_history(args.baseline)
        logger.info(f"  ✅ Loaded {len(histories['Baseline'])} snapshots")
    
    if args.polychromic:
        logger.info(f"Loading polychromic dynamics from {args.polychromic}")
        histories['Polychromic'] = load_dynamics_history(args.polychromic)
        logger.info(f"  ✅ Loaded {len(histories['Polychromic'])} snapshots")
    
    if args.grpo_baseline:
        logger.info(f"Loading baseline→GRPO dynamics from {args.grpo_baseline}")
        histories['Baseline→GRPO'] = load_dynamics_history(args.grpo_baseline)
        logger.info(f"  ✅ Loaded {len(histories['Baseline→GRPO'])} snapshots")
    
    if args.grpo_polychromic:
        logger.info(f"Loading polychromic→GRPO dynamics from {args.grpo_polychromic}")
        histories['Polychromic→GRPO'] = load_dynamics_history(args.grpo_polychromic)
        logger.info(f"  ✅ Loaded {len(histories['Polychromic→GRPO'])} snapshots")
    
    if not histories:
        logger.error("No dynamics files provided! Use --baseline or --polychromic")
        sys.exit(1)
    
    logger.info(f"\nComparing {len(histories)} models on metric: {args.metric}")
    
    # Plot comparison
    compare_diversity_trajectories(
        histories,
        metric=args.metric,
        save_path=args.output,
        phase_boundary=args.phase_boundary
    )
    
    logger.info(f"✅ Comparison plot saved to {args.output}")
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    
    for model_name, history in histories.items():
        if not history:
            continue
            
        metric_values = [getattr(s, args.metric) for s in history]
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Initial {args.metric}: {metric_values[0]:.3f}")
        logger.info(f"  Final {args.metric}: {metric_values[-1]:.3f}")
        logger.info(f"  Change: {metric_values[-1] - metric_values[0]:+.3f}")
        logger.info(f"  Min: {min(metric_values):.3f}")
        logger.info(f"  Max: {max(metric_values):.3f}")
        logger.info(f"  Snapshots: {len(history)}")
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

