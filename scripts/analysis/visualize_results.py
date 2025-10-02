#!/usr/bin/env python3
"""
Visualization Script for Research Results

Creates publication-quality figures for the paper:
- Diversity comparison (Self-BLEU, Distinct-n, Semantic)
- Pass@k curves
- Training curves from W&B
- Win rate comparisons

Usage:
    python scripts/analysis/visualize_results.py \
        --evaluation-dir output/evaluation/ \
        --output-dir output/figures/
"""

import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


def plot_diversity_comparison(diversity_data: dict, output_path: Path):
    """Create diversity metrics comparison bar chart."""
    logger.info("Creating diversity comparison plot...")
    
    baseline = diversity_data['baseline']
    polychromic = diversity_data['polychromic']
    
    # Prepare data
    metrics = {
        'Self-BLEU\n(lower=better)': [baseline['self_bleu'], polychromic['self_bleu']],
        'Distinct-2\n(higher=better)': [baseline['distinct_2'], polychromic['distinct_2']],
        'Distinct-3\n(higher=better)': [baseline['distinct_3'], polychromic['distinct_3']],
        'Semantic Diversity\n(higher=better)': [baseline['semantic_diversity'], polychromic['semantic_diversity']],
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_values = [v[0] for v in metrics.values()]
    polychromic_values = [v[1] for v in metrics.values()]
    
    rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db')
    rects2 = ax.bar(x + width/2, polychromic_values, width, label='Polychromic', color='#e74c3c')
    
    ax.set_ylabel('Score')
    ax.set_title('Diversity Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'diversity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved to {output_path / 'diversity_comparison.pdf'}")


def plot_passk_curves(passk_data: dict, output_path: Path):
    """Create Pass@k curves."""
    logger.info("Creating Pass@k curves...")
    
    model_a = passk_data['model_a']
    model_b = passk_data['model_b']
    
    k_values = sorted([int(k) for k in model_a.keys()])
    baseline_scores = [model_a[str(k)] for k in k_values]
    polychromic_scores = [model_b[str(k)] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, baseline_scores, 'o-', label='Baseline', linewidth=2, markersize=8, color='#3498db')
    ax.plot(k_values, polychromic_scores, 's-', label='Polychromic', linewidth=2, markersize=8, color='#e74c3c')
    
    ax.set_xlabel('k (number of generations)')
    ax.set_ylabel('Pass@k Score')
    ax.set_title('Pass@k Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # Add improvement annotations
    for i, k in enumerate(k_values):
        improvement = polychromic_scores[i] - baseline_scores[i]
        pct_improvement = (improvement / baseline_scores[i]) * 100 if baseline_scores[i] > 0 else 0
        
        mid_y = (baseline_scores[i] + polychromic_scores[i]) / 2
        ax.annotate(f'+{pct_improvement:.1f}%',
                   xy=(k, mid_y),
                   xytext=(5, 0),
                   textcoords='offset points',
                   fontsize=10,
                   color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(output_path / 'passk_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'passk_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved to {output_path / 'passk_curves.pdf'}")


def plot_llm_judge_results(llm_judge_data: dict, output_path: Path):
    """Create LLM-as-judge win rate visualization."""
    logger.info("Creating LLM judge visualization...")
    
    win_rates = llm_judge_data['win_rates']
    
    categories = ['Baseline Wins', 'Polychromic Wins', 'Ties']
    values = [
        win_rates['model_a_wins'],
        win_rates['model_b_wins'],
        win_rates['ties']
    ]
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('LLM-as-Judge Win Rates')
    
    # Bar chart
    ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Number of Wins')
    ax2.set_title('LLM-as-Judge Results Breakdown')
    
    # Add value labels
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax2.text(i, val + max(values)*0.02, str(val), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'llm_judge_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'llm_judge_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved to {output_path / 'llm_judge_results.pdf'}")


def plot_statistical_significance(stats_data: dict, output_path: Path):
    """Create statistical significance visualization."""
    logger.info("Creating statistical significance plot...")
    
    metrics = []
    p_values = []
    cohens_d = []
    
    for metric_name, data in stats_data.items():
        metrics.append(metric_name)
        p_values.append(data['mann_whitney']['p_value'])
        cohens_d.append(abs(data['cohens_d']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # P-values
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    ax1.barh(metrics, p_values, color=colors, alpha=0.7)
    ax1.axvline(x=0.05, color='black', linestyle='--', label='p=0.05 threshold')
    ax1.set_xlabel('p-value')
    ax1.set_title('Statistical Significance (Mann-Whitney U)')
    ax1.legend()
    ax1.invert_yaxis()
    
    # Effect sizes
    colors_d = ['green' if d > 0.3 else 'orange' if d > 0.2 else 'red' for d in cohens_d]
    ax2.barh(metrics, cohens_d, color=colors_d, alpha=0.7)
    ax2.axvline(x=0.3, color='black', linestyle='--', label='d=0.3 threshold')
    ax2.set_xlabel("Cohen's d (effect size)")
    ax2.set_title('Effect Size Analysis')
    ax2.legend()
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistical_tests.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'statistical_tests.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved to {output_path / 'statistical_tests.pdf'}")


def create_summary_table(evaluation_dir: Path, output_path: Path):
    """Create summary table for paper."""
    logger.info("Creating summary table...")
    
    # Load all results
    diversity = json.load(open(evaluation_dir / 'diversity_metrics.json'))
    quality = json.load(open(evaluation_dir / 'quality_metrics.json'))
    passk = json.load(open(evaluation_dir / 'passk_results.json'))
    
    # Create DataFrame
    data = {
        'Metric': [
            'Self-BLEU ↓',
            'Distinct-2 ↑',
            'Distinct-3 ↑',
            'Semantic Diversity ↑',
            'ROUGE-L',
            'Pass@1',
            'Pass@5',
            'Pass@10',
        ],
        'Baseline': [
            f"{diversity['baseline']['self_bleu']:.3f}",
            f"{diversity['baseline']['distinct_2']:.3f}",
            f"{diversity['baseline']['distinct_3']:.3f}",
            f"{diversity['baseline']['semantic_diversity']:.3f}",
            f"{quality['baseline']['rouge']['rougeL']:.3f}",
            f"{passk['model_a']['1']:.3f}",
            f"{passk['model_a']['5']:.3f}",
            f"{passk['model_a']['10']:.3f}",
        ],
        'Polychromic': [
            f"{diversity['polychromic']['self_bleu']:.3f}",
            f"{diversity['polychromic']['distinct_2']:.3f}",
            f"{diversity['polychromic']['distinct_3']:.3f}",
            f"{diversity['polychromic']['semantic_diversity']:.3f}",
            f"{quality['polychromic']['rouge']['rougeL']:.3f}",
            f"{passk['model_b']['1']:.3f}",
            f"{passk['model_b']['5']:.3f}",
            f"{passk['model_b']['10']:.3f}",
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(output_path / 'summary_table.csv', index=False)
    
    # Save as LaTeX
    latex = df.to_latex(index=False, caption='Evaluation Results Comparison', label='tab:results')
    with open(output_path / 'summary_table.tex', 'w') as f:
        f.write(latex)
    
    logger.info(f"  Saved to {output_path / 'summary_table.csv'}")
    logger.info(f"  Saved to {output_path / 'summary_table.tex'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument('--evaluation-dir', type=str, required=True, help='Evaluation results directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for figures')
    
    args = parser.parse_args()
    
    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("CREATING PUBLICATION-QUALITY FIGURES")
    logger.info("="*60)
    logger.info(f"Evaluation dir: {evaluation_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("="*60 + "\n")
    
    # Load data
    diversity_data = json.load(open(evaluation_dir / 'diversity_metrics.json'))
    passk_data = json.load(open(evaluation_dir / 'passk_results.json'))
    
    # Create plots
    plot_diversity_comparison(diversity_data, output_dir)
    plot_passk_curves(passk_data, output_dir)
    
    # Optional plots (if data exists)
    if (evaluation_dir / 'llm_judge_results.json').exists():
        llm_judge_data = json.load(open(evaluation_dir / 'llm_judge_results.json'))
        plot_llm_judge_results(llm_judge_data, output_dir)
    
    if (evaluation_dir / 'statistical_tests.json').exists():
        stats_data = json.load(open(evaluation_dir / 'statistical_tests.json'))
        plot_statistical_significance(stats_data, output_dir)
    
    # Summary table
    create_summary_table(evaluation_dir, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✓ ALL FIGURES CREATED!")
    logger.info("="*60)
    logger.info(f"\nFigures saved to: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - diversity_comparison.pdf/png")
    logger.info("  - passk_curves.pdf/png")
    logger.info("  - llm_judge_results.pdf/png (if available)")
    logger.info("  - statistical_tests.pdf/png (if available)")
    logger.info("  - summary_table.csv")
    logger.info("  - summary_table.tex (LaTeX format)")
    logger.info("\nUse these for your Arxiv paper!")


if __name__ == "__main__":
    main()

