#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple random seeds.

Usage:
    python scripts/analysis/aggregate_seeds.py \
        --results-dir output/evaluation/ \
        --seeds 42 123 456 \
        --output output/evaluation/aggregated_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metrics_for_seed(results_dir: Path, seed: int) -> Dict[str, Any]:
    """Load all metrics for a specific seed."""
    seed_dir = results_dir / f"seed_{seed}"
    
    if not seed_dir.exists():
        logger.warning(f"Results directory for seed {seed} not found: {seed_dir}")
        return None
    
    metrics = {}
    
    # Load diversity metrics
    diversity_file = seed_dir / "diversity_metrics.json"
    if diversity_file.exists():
        with open(diversity_file, 'r') as f:
            metrics['diversity'] = json.load(f)
    
    # Load quality metrics
    quality_file = seed_dir / "quality_metrics.json"
    if quality_file.exists():
        with open(quality_file, 'r') as f:
            metrics['quality'] = json.load(f)
    
    # Load Pass@k metrics
    passk_file = seed_dir / "passk_results.json"
    if passk_file.exists():
        with open(passk_file, 'r') as f:
            metrics['passk'] = json.load(f)
    
    return metrics


def aggregate_metrics(all_metrics: List[Dict], model_name: str) -> Dict[str, Any]:
    """
    Aggregate metrics across seeds for a specific model.
    
    Returns:
        Dictionary with mean, std, and individual seed values for each metric
    """
    aggregated = {}
    
    # Extract values for each metric across all seeds
    metric_values = {}
    
    for seed_metrics in all_metrics:
        # Diversity metrics
        if 'diversity' in seed_metrics and model_name in seed_metrics['diversity']:
            div_metrics = seed_metrics['diversity'][model_name]
            for metric_name, value in div_metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Quality metrics
        if 'quality' in seed_metrics and model_name in seed_metrics['quality']:
            qual_metrics = seed_metrics['quality'][model_name]
            for category, metrics in qual_metrics.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        full_name = f"{category}_{metric_name}"
                        if full_name not in metric_values:
                            metric_values[full_name] = []
                        metric_values[full_name].append(value)
        
        # Pass@k metrics
        if 'passk' in seed_metrics and model_name in seed_metrics['passk']:
            passk_metrics = seed_metrics['passk'][model_name]
            for k, value in passk_metrics.items():
                metric_name = f"pass@{k}"
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
    
    # Compute statistics for each metric
    for metric_name, values in metric_values.items():
        if len(values) > 0:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values,
                'n_seeds': len(values)
            }
    
    return aggregated


def format_for_latex(aggregated_results: Dict[str, Dict]) -> str:
    """
    Format aggregated results as LaTeX table rows.
    
    Returns:
        LaTeX-formatted string
    """
    latex = []
    latex.append("% Aggregated Results (Mean ± Std)")
    latex.append("% Copy this into your LaTeX table\n")
    
    # Get all models
    models = list(aggregated_results.keys())
    
    # Key metrics to include in table
    key_metrics = ['pass@1', 'pass@5', 'pass@10', 'self_bleu', 'rouge_rougeL']
    
    latex.append("\\begin{tabular}{l" + "c" * len(models) + "}")
    latex.append("\\toprule")
    latex.append("Metric & " + " & ".join(models) + " \\\\")
    latex.append("\\midrule")
    
    for metric in key_metrics:
        row = [metric.replace('_', ' ').title()]
        for model in models:
            if metric in aggregated_results[model]:
                m = aggregated_results[model][metric]['mean']
                s = aggregated_results[model][metric]['std']
                row.append(f"{m:.3f} $\\pm$ {s:.3f}")
            else:
                row.append("N/A")
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results across random seeds")
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing results for different seeds')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds to aggregate (default: 42 123 456)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for aggregated results')
    parser.add_argument('--latex-output', type=str,
                       help='Optional: Output LaTeX table')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("AGGREGATING RESULTS ACROSS SEEDS")
    logger.info("="*60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_file}")
    
    # Load metrics for all seeds
    all_seed_metrics = []
    for seed in args.seeds:
        logger.info(f"\nLoading metrics for seed {seed}...")
        metrics = load_metrics_for_seed(results_dir, seed)
        if metrics:
            all_seed_metrics.append(metrics)
            logger.info(f"  ✓ Loaded metrics for seed {seed}")
        else:
            logger.warning(f"  ✗ No metrics found for seed {seed}")
    
    if len(all_seed_metrics) == 0:
        logger.error("No metrics loaded! Check your results directory and seed values.")
        return
    
    logger.info(f"\nLoaded metrics from {len(all_seed_metrics)} seeds")
    
    # Get list of all models
    models = set()
    for seed_metrics in all_seed_metrics:
        if 'diversity' in seed_metrics:
            models.update(seed_metrics['diversity'].keys())
    
    logger.info(f"Found {len(models)} models: {', '.join(models)}")
    
    # Aggregate for each model
    logger.info("\n" + "="*60)
    logger.info("AGGREGATING METRICS")
    logger.info("="*60)
    
    aggregated_results = {}
    for model in models:
        logger.info(f"\n{model}:")
        aggregated_results[model] = aggregate_metrics(all_seed_metrics, model)
        
        # Print key metrics
        for metric_name in ['pass@10', 'self_bleu', 'rouge_rougeL']:
            if metric_name in aggregated_results[model]:
                stats = aggregated_results[model][metric_name]
                logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save aggregated results
    with open(output_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    logger.info(f"\n✅ Saved aggregated results to: {output_file}")
    
    # Generate LaTeX table if requested
    if args.latex_output:
        latex_output = Path(args.latex_output)
        latex_table = format_for_latex(aggregated_results)
        with open(latex_output, 'w') as f:
            f.write(latex_table)
        logger.info(f"✅ Saved LaTeX table to: {latex_output}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Models evaluated: {len(models)}")
    logger.info(f"Seeds aggregated: {len(all_seed_metrics)}")
    logger.info(f"Metrics per model: ~{len(list(aggregated_results.values())[0]) if aggregated_results else 0}")
    
    # Report format for paper
    logger.info("\n" + "="*60)
    logger.info("REPORTING FORMAT FOR PAPER")
    logger.info("="*60)
    logger.info("\nExample reporting:")
    if 'polychromic_lora' in aggregated_results and 'pass@10' in aggregated_results['polychromic_lora']:
        stats = aggregated_results['polychromic_lora']['pass@10']
        logger.info(f"  'Pass@10: {stats['mean']:.2f} ± {stats['std']:.2f} (mean ± std over {stats['n_seeds']} seeds)'")


if __name__ == '__main__':
    main()

