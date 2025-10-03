"""
Split Training Data for Two-Phase Training (SFT → GRPO)

This script splits the full training dataset into two non-overlapping subsets
for Phase 1 (SFT warm-start) and Phase 2 (GRPO refinement).

Scientific Rationale:
- Non-overlapping data prevents overfitting in Phase 2
- Stratified sampling maintains engagement distribution in both phases
- Reproducible splits via fixed random seed
- Validates distributions to ensure quality

Usage:
    python scripts/data/split_training_phases.py \
        --input data/processed/training_data_master_*.jsonl \
        --output-dir data/processed/phases \
        --split-ratio 0.5 \
        --seed 42
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(input_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    
    input_file = Path(input_path)
    if not input_file.exists():
        # Try glob pattern
        import glob
        files = glob.glob(input_path)
        if not files:
            raise FileNotFoundError(f"No files found matching: {input_path}")
        input_file = Path(files[0])
    
    logger.info(f"Loading data from: {input_file}")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def create_engagement_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement quartiles for stratification.
    
    Uses reply_likes as primary signal, with fallback to other metrics
    if reply_likes is missing or zero.
    """
    # Primary: reply_likes
    if 'reply_likes' in df.columns:
        engagement_score = df['reply_likes'].copy()
    else:
        logger.warning("'reply_likes' not found, using fallback metrics")
        engagement_score = pd.Series(0, index=df.index)
    
    # Fallback: combine multiple signals
    if engagement_score.sum() == 0:
        logger.warning("All reply_likes are 0, using composite score")
        engagement_score = (
            df.get('reply_retweets', 0) * 2 +  # Retweets worth more
            df.get('reply_likes', 0) +
            np.log1p(df.get('reply_author_followers', 100)) / 10
        )
    
    # Create quartiles
    try:
        df['engagement_quartile'] = pd.qcut(
            engagement_score,
            q=4,
            labels=['Q1_low', 'Q2_medium_low', 'Q3_medium_high', 'Q4_high'],
            duplicates='drop'
        )
    except ValueError as e:
        # If quartiles fail (e.g., too many duplicates), use percentiles
        logger.warning(f"Quartiles failed ({e}), using percentile-based bins")
        df['engagement_quartile'] = pd.cut(
            engagement_score,
            bins=[
                engagement_score.min() - 0.1,
                engagement_score.quantile(0.25),
                engagement_score.quantile(0.5),
                engagement_score.quantile(0.75),
                engagement_score.max() + 0.1
            ],
            labels=['Q1_low', 'Q2_medium_low', 'Q3_medium_high', 'Q4_high'],
            duplicates='drop'
        )
    
    return df


def stratified_split(
    df: pd.DataFrame,
    split_ratio: float = 0.5,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split maintaining engagement distribution.
    
    Args:
        df: DataFrame with engagement_quartile column
        split_ratio: Fraction for Phase 2 (default 0.5 for 50/50 split)
        random_seed: Random seed for reproducibility
        
    Returns:
        (phase1_df, phase2_df): Non-overlapping datasets
    """
    logger.info("Performing stratified split...")
    logger.info(f"  Split ratio: {1-split_ratio:.1%} Phase 1, {split_ratio:.1%} Phase 2")
    logger.info(f"  Random seed: {random_seed}")
    
    # Stratified split
    phase1_df, phase2_df = train_test_split(
        df,
        test_size=split_ratio,
        stratify=df['engagement_quartile'],
        random_state=random_seed
    )
    
    logger.info(f"Split complete:")
    logger.info(f"  Phase 1: {len(phase1_df)} examples")
    logger.info(f"  Phase 2: {len(phase2_df)} examples")
    
    return phase1_df, phase2_df


def validate_split(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """
    Validate that split maintains distributions and has no overlap.
    
    Returns statistics and creates visualization.
    """
    logger.info("Validating split...")
    
    # Check for overlap (should be none)
    if 'tweet_id' in phase1_df.columns and 'reply_id' in phase1_df.columns:
        phase1_ids = set(zip(phase1_df['tweet_id'], phase1_df['reply_id']))
        phase2_ids = set(zip(phase2_df['tweet_id'], phase2_df['reply_id']))
        overlap = phase1_ids & phase2_ids
        
        if overlap:
            logger.error(f"⚠️ Found {len(overlap)} overlapping examples!")
            raise ValueError("Data overlap detected - split is invalid")
        else:
            logger.info("✓ No overlap detected (data is properly partitioned)")
    
    # Compare distributions
    stats = {
        'phase1': {
            'count': len(phase1_df),
            'quartile_dist': phase1_df['engagement_quartile'].value_counts().to_dict(),
        },
        'phase2': {
            'count': len(phase2_df),
            'quartile_dist': phase2_df['engagement_quartile'].value_counts().to_dict(),
        }
    }
    
    # Engagement statistics
    if 'reply_likes' in phase1_df.columns:
        stats['phase1']['engagement_stats'] = {
            'mean': float(phase1_df['reply_likes'].mean()),
            'median': float(phase1_df['reply_likes'].median()),
            'std': float(phase1_df['reply_likes'].std()),
            'min': int(phase1_df['reply_likes'].min()),
            'max': int(phase1_df['reply_likes'].max()),
        }
        stats['phase2']['engagement_stats'] = {
            'mean': float(phase2_df['reply_likes'].mean()),
            'median': float(phase2_df['reply_likes'].median()),
            'std': float(phase2_df['reply_likes'].std()),
            'min': int(phase2_df['reply_likes'].min()),
            'max': int(phase2_df['reply_likes'].max()),
        }
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 (SFT Warm-start)")
    logger.info("="*60)
    logger.info(f"Count: {stats['phase1']['count']}")
    logger.info("Quartile distribution:")
    for q, count in sorted(stats['phase1']['quartile_dist'].items()):
        pct = count / stats['phase1']['count'] * 100
        logger.info(f"  {q}: {count} ({pct:.1f}%)")
    
    if 'engagement_stats' in stats['phase1']:
        logger.info("Engagement statistics:")
        for key, val in stats['phase1']['engagement_stats'].items():
            logger.info(f"  {key}: {val}")
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 (GRPO Refinement)")
    logger.info("="*60)
    logger.info(f"Count: {stats['phase2']['count']}")
    logger.info("Quartile distribution:")
    for q, count in sorted(stats['phase2']['quartile_dist'].items()):
        pct = count / stats['phase2']['count'] * 100
        logger.info(f"  {q}: {count} ({pct:.1f}%)")
    
    if 'engagement_stats' in stats['phase2']:
        logger.info("Engagement statistics:")
        for key, val in stats['phase2']['engagement_stats'].items():
            logger.info(f"  {key}: {val}")
    
    # Create visualization
    create_distribution_plot(phase1_df, phase2_df, output_dir)
    
    # Statistical test (Chi-square for quartile distributions)
    from scipy.stats import chi2_contingency
    
    # Create contingency table
    phase1_counts = [stats['phase1']['quartile_dist'].get(q, 0) 
                     for q in ['Q1_low', 'Q2_medium_low', 'Q3_medium_high', 'Q4_high']]
    phase2_counts = [stats['phase2']['quartile_dist'].get(q, 0) 
                     for q in ['Q1_low', 'Q2_medium_low', 'Q3_medium_high', 'Q4_high']]
    
    contingency = [phase1_counts, phase2_counts]
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    stats['chi_square_test'] = {
        'statistic': float(chi2),
        'p_value': float(p_value),
        'dof': int(dof)
    }
    
    logger.info("\n" + "="*60)
    logger.info("DISTRIBUTION SIMILARITY TEST")
    logger.info("="*60)
    logger.info(f"Chi-square statistic: {chi2:.4f}")
    logger.info(f"p-value: {p_value:.4f}")
    
    if p_value > 0.05:
        logger.info("✓ Distributions are statistically similar (p > 0.05)")
        logger.info("  → Stratified split successfully maintained engagement distribution")
    else:
        logger.warning("⚠️ Distributions differ significantly (p < 0.05)")
        logger.warning("  → May indicate stratification issue")
    
    return stats


def create_distribution_plot(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    output_dir: Path
):
    """Create visualization comparing distributions."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quartile distribution comparison
        ax = axes[0, 0]
        quartile_data = pd.DataFrame({
            'Phase 1 (SFT)': phase1_df['engagement_quartile'].value_counts().sort_index(),
            'Phase 2 (GRPO)': phase2_df['engagement_quartile'].value_counts().sort_index()
        })
        quartile_data.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
        ax.set_title('Engagement Quartile Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quartile')
        ax.set_ylabel('Count')
        ax.legend(title='Phase')
        ax.grid(axis='y', alpha=0.3)
        
        # Engagement distribution (if available)
        if 'reply_likes' in phase1_df.columns:
            ax = axes[0, 1]
            bins = np.linspace(
                min(phase1_df['reply_likes'].min(), phase2_df['reply_likes'].min()),
                max(phase1_df['reply_likes'].max(), phase2_df['reply_likes'].max()),
                30
            )
            ax.hist(phase1_df['reply_likes'], bins=bins, alpha=0.5, label='Phase 1 (SFT)', color='#3498db')
            ax.hist(phase2_df['reply_likes'], bins=bins, alpha=0.5, label='Phase 2 (GRPO)', color='#e74c3c')
            ax.set_title('Reply Likes Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Reply Likes')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Reply length distribution
        if 'reply' in phase1_df.columns:
            ax = axes[1, 0]
            phase1_lengths = phase1_df['reply'].str.len()
            phase2_lengths = phase2_df['reply'].str.len()
            bins = np.linspace(
                min(phase1_lengths.min(), phase2_lengths.min()),
                max(phase1_lengths.max(), phase2_lengths.max()),
                30
            )
            ax.hist(phase1_lengths, bins=bins, alpha=0.5, label='Phase 1 (SFT)', color='#3498db')
            ax.hist(phase2_lengths, bins=bins, alpha=0.5, label='Phase 2 (GRPO)', color='#e74c3c')
            ax.set_title('Reply Length Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Reply Length (characters)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Summary statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_data = []
        summary_data.append(['Metric', 'Phase 1', 'Phase 2'])
        summary_data.append(['─' * 20, '─' * 15, '─' * 15])
        summary_data.append(['Total Examples', f"{len(phase1_df)}", f"{len(phase2_df)}"])
        
        if 'reply_likes' in phase1_df.columns:
            summary_data.append([
                'Mean Likes',
                f"{phase1_df['reply_likes'].mean():.1f}",
                f"{phase2_df['reply_likes'].mean():.1f}"
            ])
            summary_data.append([
                'Median Likes',
                f"{phase1_df['reply_likes'].median():.0f}",
                f"{phase2_df['reply_likes'].median():.0f}"
            ])
        
        if 'reply' in phase1_df.columns:
            summary_data.append([
                'Mean Length',
                f"{phase1_df['reply'].str.len().mean():.0f}",
                f"{phase2_df['reply'].str.len().mean():.0f}"
            ])
        
        table = ax.table(
            cellText=summary_data,
            loc='center',
            cellLoc='left',
            colWidths=[0.4, 0.3, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        output_file = output_dir / 'phase_split_distributions.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved distribution plot: {output_file}")
        
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create distribution plot: {e}")


def save_phase_data(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    output_dir: Path
):
    """Save Phase 1 and Phase 2 data to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove temporary columns
    phase1_clean = phase1_df.drop('engagement_quartile', axis=1, errors='ignore')
    phase2_clean = phase2_df.drop('engagement_quartile', axis=1, errors='ignore')
    
    # Save Phase 1
    phase1_file = output_dir / 'training_data_phase1_sft.jsonl'
    phase1_clean.to_json(phase1_file, orient='records', lines=True)
    logger.info(f"✓ Saved Phase 1 data: {phase1_file}")
    logger.info(f"  {len(phase1_clean)} examples for SFT warm-start")
    
    # Save Phase 2
    phase2_file = output_dir / 'training_data_phase2_grpo.jsonl'
    phase2_clean.to_json(phase2_file, orient='records', lines=True)
    logger.info(f"✓ Saved Phase 2 data: {phase2_file}")
    logger.info(f"  {len(phase2_clean)} examples for GRPO refinement")
    
    return phase1_file, phase2_file


def save_statistics(stats: Dict, output_dir: Path):
    """Save split statistics to JSON."""
    stats_file = output_dir / 'phase_split_statistics.json'
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ Saved statistics: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Split training data for two-phase training (SFT → GRPO)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (50/50 split)
  python scripts/data/split_training_phases.py \\
      --input data/processed/training_data_master_*.jsonl

  # Custom split ratio (60% Phase 1, 40% Phase 2)
  python scripts/data/split_training_phases.py \\
      --input data/processed/training_data_master_*.jsonl \\
      --split-ratio 0.4 \\
      --seed 42

  # Different output directory
  python scripts/data/split_training_phases.py \\
      --input data/processed/training_data_master_*.jsonl \\
      --output-dir data/processed/experiment_splits
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (supports glob patterns)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/phases',
        help='Output directory for split data (default: data/processed/phases)'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.5,
        help='Fraction for Phase 2 (default: 0.5 for 50/50 split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating distribution plots'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Two-Phase Training Data Split")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Split ratio: {1-args.split_ratio:.1%} Phase 1 / {args.split_ratio:.1%} Phase 2")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*60 + "\n")
    
    # Load data
    data = load_training_data(args.input)
    
    if len(data) == 0:
        logger.error("No data loaded!")
        return 1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    logger.info(f"Columns: {list(df.columns)}")
    
    # Create engagement quartiles
    df = create_engagement_quartiles(df)
    
    # Perform stratified split
    phase1_df, phase2_df = stratified_split(
        df,
        split_ratio=args.split_ratio,
        random_seed=args.seed
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Validate split
    stats = validate_split(phase1_df, phase2_df, output_dir)
    
    # Save data
    phase1_file, phase2_file = save_phase_data(phase1_df, phase2_df, output_dir)
    
    # Save statistics
    save_statistics(stats, output_dir)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SPLIT COMPLETE ✓")
    logger.info("="*60)
    logger.info(f"Phase 1 (SFT): {phase1_file}")
    logger.info(f"Phase 2 (GRPO): {phase2_file}")
    logger.info(f"Statistics: {output_dir / 'phase_split_statistics.json'}")
    if not args.no_plots:
        logger.info(f"Visualization: {output_dir / 'phase_split_distributions.png'}")
    logger.info("\nNext steps:")
    logger.info("  1. Review distribution plot to validate split quality")
    logger.info("  2. Update config files to use phase-specific data")
    logger.info("  3. Begin Phase 1 training (SFT warm-start)")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())

