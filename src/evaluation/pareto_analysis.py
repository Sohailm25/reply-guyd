"""
Pareto Frontier Analysis for Multi-Objective Optimization

Standard multi-objective optimization visualization showing quality-diversity trade-offs.
Connects to established MOO theory.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def is_pareto_optimal(costs: np.ndarray, maximize: List[bool]) -> np.ndarray:
    """
    Identify Pareto-optimal points.
    
    Args:
        costs: (N, M) array where N = points, M = objectives
        maximize: List of M booleans indicating if objective should be maximized
        
    Returns:
        Boolean array of length N indicating Pareto-optimal points
    """
    # Convert maximization to minimization
    adjusted_costs = costs.copy()
    for i, should_max in enumerate(maximize):
        if should_max:
            adjusted_costs[:, i] = -adjusted_costs[:, i]
    
    n_points = costs.shape[0]
    is_optimal = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_optimal[i]:
            # Point i is dominated if there exists j where all objectives are better
            is_optimal[is_optimal] = np.any(
                adjusted_costs[is_optimal] > adjusted_costs[i], 
                axis=1
            )
            is_optimal[i] = True  # Restore i
    
    return is_optimal


def compute_pareto_frontier(
    models: Dict[str, Dict[str, float]],
    quality_metric: str = 'rouge_l',
    diversity_metric: str = 'distinct_2'
) -> Dict:
    """
    Compute Pareto frontier for quality vs. diversity trade-off.
    
    Args:
        models: Dict mapping model_name -> {metric_name: value}
        quality_metric: Which metric represents quality
        diversity_metric: Which metric represents diversity
        
    Returns:
        Dict with pareto_models, all_models, and statistics
    """
    if not models:
        logger.warning("No models provided for Pareto analysis")
        return {
            'pareto_models': [],
            'all_models': [],
            'quality_values': [],
            'diversity_values': [],
            'is_pareto_optimal': [],
            'metrics': {'quality': quality_metric, 'diversity': diversity_metric}
        }
    
    # Extract data
    model_names = list(models.keys())
    
    # Check if metrics exist
    quality_values = []
    diversity_values = []
    valid_models = []
    
    for name in model_names:
        if quality_metric in models[name] and diversity_metric in models[name]:
            quality_values.append(models[name][quality_metric])
            diversity_values.append(models[name][diversity_metric])
            valid_models.append(name)
        else:
            logger.warning(f"Model {name} missing metrics {quality_metric} or {diversity_metric}")
    
    if not valid_models:
        logger.error("No valid models with both metrics!")
        return {
            'pareto_models': [],
            'all_models': model_names,
            'quality_values': [],
            'diversity_values': [],
            'is_pareto_optimal': [],
            'metrics': {'quality': quality_metric, 'diversity': diversity_metric}
        }
    
    quality_values = np.array(quality_values)
    diversity_values = np.array(diversity_values)
    
    # Stack into (N, 2) array
    costs = np.column_stack([quality_values, diversity_values])
    
    # Both should be maximized
    is_optimal = is_pareto_optimal(costs, maximize=[True, True])
    
    # Extract Pareto-optimal models
    pareto_models = [valid_models[i] for i in range(len(valid_models)) if is_optimal[i]]
    
    results = {
        'pareto_models': pareto_models,
        'all_models': valid_models,
        'quality_values': quality_values.tolist(),
        'diversity_values': diversity_values.tolist(),
        'is_pareto_optimal': is_optimal.tolist(),
        'metrics': {
            'quality': quality_metric,
            'diversity': diversity_metric
        },
        'statistics': {
            'n_models': len(valid_models),
            'n_pareto_optimal': int(is_optimal.sum()),
            'pareto_percentage': float(is_optimal.sum() / len(valid_models) * 100)
        }
    }
    
    logger.info(f"✅ Computed Pareto frontier:")
    logger.info(f"   Total models: {len(valid_models)}")
    logger.info(f"   Pareto-optimal: {is_optimal.sum()} ({results['statistics']['pareto_percentage']:.1f}%)")
    logger.info(f"   Pareto models: {', '.join(pareto_models)}")
    
    return results


def plot_pareto_frontier(
    models: Dict[str, Dict[str, float]],
    quality_metric: str = 'rouge_l',
    diversity_metric: str = 'distinct_2',
    save_path: Optional[str] = None,
    highlight_model: Optional[str] = None,
    title: str = "Quality-Diversity Trade-off: Pareto Frontier Analysis"
):
    """
    Plot Pareto frontier for quality vs. diversity.
    
    Args:
        models: Dict mapping model_name -> {metric_name: value}
        quality_metric: X-axis metric (quality)
        diversity_metric: Y-axis metric (diversity)
        save_path: Where to save plot
        highlight_model: Model name to highlight (e.g., "Polychromic→GRPO")
        title: Plot title
        
    Returns:
        Pareto analysis results dict
    """
    # Compute Pareto frontier
    results = compute_pareto_frontier(models, quality_metric, diversity_metric)
    
    if not results['all_models']:
        logger.error("Cannot plot - no valid models!")
        return results
    
    model_names = results['all_models']
    quality = np.array(results['quality_values'])
    diversity = np.array(results['diversity_values'])
    is_optimal = np.array(results['is_pareto_optimal'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    for i, name in enumerate(model_names):
        if is_optimal[i]:
            # Pareto-optimal points (larger, colorful)
            if name == highlight_model:
                color = '#d62728'  # Red for highlighted
                marker = '*'
                size = 400
                zorder = 5
                edgewidth = 3
            else:
                color = '#2ca02c'  # Green for Pareto
                marker = 'o'
                size = 200
                zorder = 3
                edgewidth = 2
            
            ax.scatter(quality[i], diversity[i], 
                      color=color, s=size, marker=marker,
                      edgecolors='black', linewidths=edgewidth, zorder=zorder,
                      label=f"{name}" if name == highlight_model else None,
                      alpha=0.9)
        else:
            # Dominated points (smaller, gray)
            ax.scatter(quality[i], diversity[i], 
                      color='lightgray', s=120, marker='o',
                      edgecolors='gray', linewidths=1.5, zorder=2,
                      alpha=0.6)
    
    # Add labels for each point (simplified to avoid annotation issues in headless mode)
    for i, name in enumerate(model_names):
        # Simple offset in data coordinates
        offset_x = 0.002 if quality[i] > quality.mean() else -0.002
        offset_y = 0.005
        ha = 'left' if quality[i] > quality.mean() else 'right'
        
        # Color label based on status
        if is_optimal[i]:
            if name == highlight_model:
                bbox_color = '#ffcccc'  # Light red for highlighted
                fontweight = 'bold'
            else:
                bbox_color = '#ccffcc'  # Light green for Pareto
                fontweight = 'normal'
        else:
            bbox_color = 'white'
            fontweight = 'normal'
        
        ax.text(quality[i] + offset_x, diversity[i] + offset_y, name,
               fontsize=10, ha=ha, va='bottom',
               fontweight=fontweight,
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor=bbox_color,
                        edgecolor='gray', alpha=0.9))
    
    # Draw Pareto frontier line
    if is_optimal.sum() > 1:
        pareto_indices = np.where(is_optimal)[0]
        pareto_quality = quality[pareto_indices]
        pareto_diversity = diversity[pareto_indices]
        
        # Sort by quality for connecting line
        sorted_indices = np.argsort(pareto_quality)
        ax.plot(pareto_quality[sorted_indices], pareto_diversity[sorted_indices],
               'g--', linewidth=2.5, alpha=0.5, label='Pareto Frontier', zorder=1)
    
    # Labels and formatting
    ax.set_xlabel(f'{quality_metric.replace("_", " ").title()} (Quality) →', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{diversity_metric.replace("_", " ").title()} (Diversity) →', 
                 fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    # Manually create legend entries
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='#2ca02c', markersize=12, 
               markeredgecolor='black', markeredgewidth=2,
               label='Pareto Optimal', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='lightgray', markersize=10, 
               markeredgecolor='gray', markeredgewidth=1.5,
               label='Dominated', linestyle='None')
    ]
    
    if highlight_model and highlight_model in model_names:
        legend_elements.insert(0, 
            Line2D([0], [0], marker='*', color='w', 
                   markerfacecolor='#d62728', markersize=15, 
                   markeredgecolor='black', markeredgewidth=2.5,
                   label=f'{highlight_model} (Best)', linestyle='None')
        )
    
    if is_optimal.sum() > 1:
        legend_elements.append(
            Line2D([0], [0], color='#2ca02c', linewidth=2.5, 
                   linestyle='--', alpha=0.5, label='Pareto Frontier')
        )
    
    ax.legend(handles=legend_elements, fontsize=11, loc='best', 
             framealpha=0.95, edgecolor='gray')
    
    # Add annotations
    textstr = f'Pareto-optimal models: {is_optimal.sum()}/{len(model_names)}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.8))
    
    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed: {e}, continuing anyway")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved Pareto frontier plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return results


def save_pareto_analysis(results: Dict, filepath: str):
    """Save Pareto analysis results to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Saved Pareto analysis to {filepath}")


# Example usage
if __name__ == "__main__":
    print("✅ Pareto analysis module loaded successfully")
    print("Standard multi-objective optimization visualization")
    
    # Test with dummy data
    dummy_models = {
        "Baseline (SFT)": {
            "rouge_l": 0.35,
            "distinct_2": 0.62,
        },
        "Polychromic (SFT)": {
            "rouge_l": 0.33,
            "distinct_2": 0.78,
        },
        "Baseline→GRPO": {
            "rouge_l": 0.32,
            "distinct_2": 0.68,
        },
        "Polychromic→GRPO": {
            "rouge_l": 0.34,
            "distinct_2": 0.74,
        }
    }
    
    print("\nTesting with dummy data...")
    results = plot_pareto_frontier(
        dummy_models,
        quality_metric='rouge_l',
        diversity_metric='distinct_2',
        save_path='test_pareto_frontier.pdf',
        highlight_model='Polychromic→GRPO'
    )
    print(f"\nPareto-optimal models: {results['pareto_models']}")
    print(f"Generated test plot: test_pareto_frontier.pdf")

