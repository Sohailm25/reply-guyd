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
    """Create diversity metrics comparison bar chart (handles variable number of models)."""
    logger.info("Creating diversity comparison plot...")
    
    model_names = list(diversity_data.keys())
    n_models = len(model_names)
    
    # Prepare data - each metric gets values from all models
    metrics = {
        'Self-BLEU\n(lower=better)': [diversity_data[m]['self_bleu'] for m in model_names],
        'Distinct-2\n(higher=better)': [diversity_data[m]['distinct_2'] for m in model_names],
        'Distinct-3\n(higher=better)': [diversity_data[m]['distinct_3'] for m in model_names],
        'Semantic Diversity\n(higher=better)': [diversity_data[m]['semantic_diversity'] for m in model_names],
    }
    
    # Create figure (wider for more models)
    fig, ax = plt.subplots(figsize=(max(12, 3*n_models), 6))
    
    x = np.arange(len(metrics))
    width = 0.8 / n_models  # Dynamic width based on number of models
    
    # Colors and patterns for up to 4 models
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, model_name in enumerate(model_names):
        values = [metrics[metric][i] for metric in metrics.keys()]
        offset = width * (i - n_models/2 + 0.5)
        rects = ax.bar(x + offset, values, width, 
                      label=model_name.replace('_', ' ').title(), 
                      color=colors[i % len(colors)])
        
        # Add value labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Diversity Metrics Comparison ({n_models} Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'diversity_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved to {output_path / 'diversity_comparison.pdf'}")


def plot_passk_curves(passk_data: dict, output_path: Path):
    """Create Pass@k curves (handles variable number of models)."""
    logger.info("Creating Pass@k curves...")
    
    model_names = list(passk_data.keys())
    n_models = len(model_names)
    
    # Get k values from first model
    k_values = sorted([int(k) for k in passk_data[model_names[0]].keys()])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    markers = ['o', 's', '^', 'D']
    
    for i, model_name in enumerate(model_names):
        scores = [passk_data[model_name][str(k)] for k in k_values]
        ax.plot(k_values, scores, 
               marker=markers[i % len(markers)], 
               label=model_name.replace('_', ' ').title(), 
               linewidth=2, 
               markersize=8, 
               color=colors[i % len(colors)])
    
    ax.set_xlabel('k (number of generations)')
    ax.set_ylabel('Pass@k Score')
    ax.set_title(f'Pass@k Performance Comparison ({n_models} Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
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
    """Create summary table for paper (handles variable number of models)."""
    logger.info("Creating summary table...")
    
    # Load all results
    diversity = json.load(open(evaluation_dir / 'diversity_metrics.json'))
    quality = json.load(open(evaluation_dir / 'quality_metrics.json'))
    passk = json.load(open(evaluation_dir / 'passk_results.json'))
    
    model_names = list(diversity.keys())
    
    # Create DataFrame with dynamic columns
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
        ]
    }
    
    # Add column for each model
    for model_name in model_names:
        column_name = model_name.replace('_', ' ').title()
        data[column_name] = [
            f"{diversity[model_name]['self_bleu']:.3f}",
            f"{diversity[model_name]['distinct_2']:.3f}",
            f"{diversity[model_name]['distinct_3']:.3f}",
            f"{diversity[model_name]['semantic_diversity']:.3f}",
            f"{quality[model_name]['rouge']['rougeL']:.3f}",
            f"{passk[model_name]['1']:.3f}",
            f"{passk[model_name]['5']:.3f}",
            f"{passk[model_name]['10']:.3f}",
        ]
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    df.to_csv(output_path / 'summary_table.csv', index=False)
    
    # Save as LaTeX
    latex = df.to_latex(
        index=False, 
        caption=f'Evaluation Results Comparison ({len(model_names)} Models)', 
        label='tab:results'
    )
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

