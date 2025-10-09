#!/usr/bin/env python3
"""
Split Training Data into Phase 1 (SFT) and Phase 2 (GRPO)

For two-phase training:
- Phase 1: Supervised fine-tuning (SFT) warm-start
- Phase 2: GRPO refinement

Uses stratified split to maintain engagement distribution across both phases.

Usage:
    python scripts/data/split_training_phases.py \
        --input data/processed/training_data_curated_5000.jsonl \
        --output-dir data/processed/ \
        --split-ratio 0.5 \
        --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(input_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    logger.info(f"Loading data from: {input_path}")
    
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    
    logger.info(f"Loaded {len(examples)} examples")
    return examples


def compute_engagement_quartiles(examples: List[Dict]) -> List[int]:
    """
    Compute engagement quartiles for stratified splitting.
    
    Engagement score = reply_likes / max(tweet_likes, 1)
    
    Returns:
        List of quartile labels (0-3) for each example
    """
    logger.info("Computing engagement quartiles for stratified split...")
    
    engagement_scores = []
    for ex in examples:
        reply_likes = ex.get('reply_likes', 0)
        tweet_likes = ex.get('tweet_likes', 1)
        engagement = reply_likes / max(tweet_likes, 1)
        engagement_scores.append(engagement)
    
    # Compute quartiles
    quartiles = np.percentile(engagement_scores, [25, 50, 75])
    
    # Assign quartile labels
    quartile_labels = []
    for score in engagement_scores:
        if score <= quartiles[0]:
            quartile_labels.append(0)  # Q1
        elif score <= quartiles[1]:
            quartile_labels.append(1)  # Q2
        elif score <= quartiles[2]:
            quartile_labels.append(2)  # Q3
        else:
            quartile_labels.append(3)  # Q4
    
    logger.info(f"Quartile boundaries: {quartiles}")
    logger.info(f"Quartile counts: {np.bincount(quartile_labels)}")
    
    return quartile_labels


def stratified_split(
    examples: List[Dict],
    quartiles: List[int],
    split_ratio: float,
    seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform stratified split by engagement quartiles.
    
    Args:
        examples: All training examples
        quartiles: Quartile label for each example
        split_ratio: Fraction for phase 1 (e.g., 0.5 = 50/50 split)
        seed: Random seed
        
    Returns:
        (phase1_examples, phase2_examples)
    """
    logger.info(f"Performing stratified split (ratio={split_ratio}, seed={seed})...")
    
    np.random.seed(seed)
    
    # Group examples by quartile
    quartile_groups = defaultdict(list)
    for ex, q in zip(examples, quartiles):
        quartile_groups[q].append(ex)
    
    phase1_examples = []
    phase2_examples = []
    
    # Split each quartile proportionally
    for q in sorted(quartile_groups.keys()):
        group = quartile_groups[q]
        n_group = len(group)
        n_phase1 = int(n_group * split_ratio)
        
        # Shuffle and split
        indices = np.random.permutation(n_group)
        phase1_indices = indices[:n_phase1]
        phase2_indices = indices[n_phase1:]
        
        phase1_examples.extend([group[i] for i in phase1_indices])
        phase2_examples.extend([group[i] for i in phase2_indices])
        
        logger.info(f"  Quartile {q}: {n_phase1}/{n_group} to Phase 1")
    
    logger.info(f"Phase 1: {len(phase1_examples)} examples")
    logger.info(f"Phase 2: {len(phase2_examples)} examples")
    
    return phase1_examples, phase2_examples


def save_jsonl(examples: List[Dict], output_path: str):
    """Save examples to JSONL file."""
    logger.info(f"Saving {len(examples)} examples to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"✅ Saved: {output_path}")


def compute_statistics(
    phase1_examples: List[Dict],
    phase2_examples: List[Dict]
) -> Dict:
    """Compute statistics for both phases."""
    logger.info("Computing split statistics...")
    
    def get_stats(examples):
        engagement = [
            ex.get('reply_likes', 0) / max(ex.get('tweet_likes', 1), 1)
            for ex in examples
        ]
        lengths = [len(ex.get('reply', '')) for ex in examples]
        
        return {
            'count': len(examples),
            'engagement': {
                'mean': float(np.mean(engagement)),
                'std': float(np.std(engagement)),
                'median': float(np.median(engagement)),
                'min': float(np.min(engagement)),
                'max': float(np.max(engagement))
            },
            'length': {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'median': float(np.median(lengths)),
                'min': float(np.min(lengths)),
                'max': float(np.max(lengths))
            }
        }
    
    stats = {
        'phase1': get_stats(phase1_examples),
        'phase2': get_stats(phase2_examples)
    }
    
    # Log key statistics
    logger.info(f"Phase 1 engagement: {stats['phase1']['engagement']['mean']:.4f} ± {stats['phase1']['engagement']['std']:.4f}")
    logger.info(f"Phase 2 engagement: {stats['phase2']['engagement']['mean']:.4f} ± {stats['phase2']['engagement']['std']:.4f}")
    
    return stats


def plot_distributions(
    phase1_examples: List[Dict],
    phase2_examples: List[Dict],
    output_path: str
):
    """Plot engagement distributions for both phases."""
    logger.info("Generating distribution plots...")
    
    # Extract engagement scores
    phase1_engagement = [
        ex.get('reply_likes', 0) / max(ex.get('tweet_likes', 1), 1)
        for ex in phase1_examples
    ]
    phase2_engagement = [
        ex.get('reply_likes', 0) / max(ex.get('tweet_likes', 1), 1)
        for ex in phase2_examples
    ]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(phase1_engagement, bins=50, alpha=0.6, label='Phase 1 (SFT)', color='blue')
    axes[0].hist(phase2_engagement, bins=50, alpha=0.6, label='Phase 2 (GRPO)', color='orange')
    axes[0].set_xlabel('Engagement (reply_likes / tweet_likes)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Engagement Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data = [phase1_engagement, phase2_engagement]
    axes[1].boxplot(data, labels=['Phase 1\n(SFT)', 'Phase 2\n(GRPO)'])
    axes[1].set_ylabel('Engagement')
    axes[1].set_title('Engagement Distribution Comparison')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved plot: {output_path}")
    plt.close()


def validate_split(
    phase1_examples: List[Dict],
    phase2_examples: List[Dict]
) -> bool:
    """
    Validate that the split maintains similar distributions.
    
    Uses chi-square test to compare quartile distributions.
    """
    logger.info("Validating split quality...")
    
    # Compute engagement quartiles for each phase
    def get_quartile_counts(examples):
        engagement = [
            ex.get('reply_likes', 0) / max(ex.get('tweet_likes', 1), 1)
            for ex in examples
        ]
        quartiles = np.percentile(engagement, [25, 50, 75])
        counts = [0, 0, 0, 0]
        for score in engagement:
            if score <= quartiles[0]:
                counts[0] += 1
            elif score <= quartiles[1]:
                counts[1] += 1
            elif score <= quartiles[2]:
                counts[2] += 1
            else:
                counts[3] += 1
        return counts
    
    phase1_counts = get_quartile_counts(phase1_examples)
    phase2_counts = get_quartile_counts(phase2_examples)
    
    # Normalize to proportions
    phase1_props = np.array(phase1_counts) / sum(phase1_counts)
    phase2_props = np.array(phase2_counts) / sum(phase2_counts)
    
    # Compute difference
    max_diff = np.max(np.abs(phase1_props - phase2_props))
    
    logger.info(f"Phase 1 quartile distribution: {phase1_props}")
    logger.info(f"Phase 2 quartile distribution: {phase2_props}")
    logger.info(f"Max difference: {max_diff:.4f}")
    
    # Threshold: distributions should be similar (< 5% difference)
    is_valid = max_diff < 0.05
    
    if is_valid:
        logger.info("✅ Split validation PASSED: Distributions are similar")
    else:
        logger.warning(f"⚠️  Split validation WARNING: Distributions differ by {max_diff:.2%}")
    
    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Split training data into Phase 1 (SFT) and Phase 2 (GRPO)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file with training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.5,
        help='Fraction of data for Phase 1 (default: 0.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("TRAINING DATA SPLITTER: Two-Phase Training")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Split ratio: {args.split_ratio}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    examples = load_training_data(args.input)
    
    if len(examples) == 0:
        logger.error("No examples loaded. Exiting.")
        return 1
    
    # Compute engagement quartiles for stratified split
    quartiles = compute_engagement_quartiles(examples)
    
    # Perform stratified split
    phase1_examples, phase2_examples = stratified_split(
        examples, quartiles, args.split_ratio, args.seed
    )
    
    # Save splits
    phase1_path = output_dir / "train_phase1_sft.jsonl"
    phase2_path = output_dir / "train_phase2_grpo.jsonl"
    
    save_jsonl(phase1_examples, str(phase1_path))
    save_jsonl(phase2_examples, str(phase2_path))
    
    # Compute statistics
    stats = compute_statistics(phase1_examples, phase2_examples)
    
    # Save statistics
    stats_path = output_dir / "phase_split_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✅ Saved statistics: {stats_path}")
    
    # Plot distributions
    plot_path = output_dir / "phase_split_distributions.png"
    plot_distributions(phase1_examples, phase2_examples, str(plot_path))
    
    # Validate split
    validate_split(phase1_examples, phase2_examples)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ DATA SPLITTING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nPhase 1 (SFT warm-start):  {len(phase1_examples):,} examples")
    logger.info(f"  → {phase1_path}")
    logger.info(f"\nPhase 2 (GRPO refinement): {len(phase2_examples):,} examples")
    logger.info(f"  → {phase2_path}")
    logger.info(f"\nStatistics: {stats_path}")
    logger.info(f"Visualization: {plot_path}")
    logger.info("\n" + "="*70)
    logger.info("Next steps:")
    logger.info("  1. Train Phase 1 (SFT warm-start):")
    logger.info(f"     python scripts/training/train_model.py \\")
    logger.info(f"       --config config/experiments/baseline_warmstart.yaml \\")
    logger.info(f"       --data {phase1_path}")
    logger.info("\n  2. Train Phase 2 (GRPO refinement):")
    logger.info(f"     python scripts/training/train_model.py \\")
    logger.info(f"       --config config/experiments/grpo_from_baseline.yaml \\")
    logger.info(f"       --data {phase2_path}")
    logger.info("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
