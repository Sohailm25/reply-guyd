"""
LoRA Parameter Analysis

Analyze LoRA matrices to understand WHERE diversity is encoded in the model.
This is NOVEL - no existing paper analyzes LoRA parameters for diversity.

Key questions:
- Which layers change most in diversity-aware training?
- What is the effective rank of LoRA updates?
- How do baseline vs. polychromic LoRA matrices differ?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


@dataclass
class LayerLoRAStats:
    """Statistics for a single LoRA layer."""
    layer_name: str
    lora_A_norm: float
    lora_B_norm: float
    lora_product_norm: float  # ||B·A||
    effective_rank: float
    top_singular_values: List[float]  # Top 5 singular values
    
    def to_dict(self):
        return asdict(self)


class LoRAAnalyzer:
    """
    Analyze LoRA parameters to understand model changes.
    
    Usage:
        analyzer = LoRAAnalyzer(model_path)
        stats = analyzer.analyze_all_layers()
        analyzer.plot_layer_heatmap(stats, save_path='heatmap.pdf')
    """
    
    def __init__(self, model_path: str):
        """
        Initialize analyzer with path to trained model.
        
        Args:
            model_path: Path to model checkpoint directory
        """
        self.model_path = Path(model_path)
        self.adapter_path = self.model_path / "adapter_model.safetensors"
        
        if not self.adapter_path.exists():
            # Try alternative path
            self.adapter_path = self.model_path / "adapter_model.bin"
        
        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"No adapter model found at {model_path}. "
                f"Expected adapter_model.safetensors or adapter_model.bin"
            )
        
        logger.info(f"Initialized LoRA analyzer for: {model_path}")
        logger.info(f"Adapter file: {self.adapter_path}")
    
    def load_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from checkpoint."""
        logger.info("Loading LoRA weights...")
        
        if str(self.adapter_path).endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                weights = load_file(str(self.adapter_path))
            except ImportError:
                logger.error("safetensors not installed. Install with: pip install safetensors")
                raise
        else:
            weights = torch.load(str(self.adapter_path), map_location='cpu')
        
        logger.info(f"Loaded {len(weights)} weight tensors")
        return weights
    
    def analyze_layer(
        self, 
        lora_A: torch.Tensor, 
        lora_B: torch.Tensor,
        layer_name: str
    ) -> LayerLoRAStats:
        """
        Analyze a single LoRA layer (A and B matrices).
        
        Args:
            lora_A: LoRA A matrix (low-rank input projection)
            lora_B: LoRA B matrix (low-rank output projection)
            layer_name: Name of the layer
            
        Returns:
            LayerLoRAStats with computed metrics
        """
        # Compute norms
        lora_A_norm = torch.norm(lora_A).item()
        lora_B_norm = torch.norm(lora_B).item()
        
        # Compute product B·A
        lora_product = torch.mm(lora_B, lora_A)
        lora_product_norm = torch.norm(lora_product).item()
        
        # Compute SVD for effective rank
        try:
            U, S, V = torch.svd(lora_product)
            singular_values = S.cpu().numpy()
            
            # Effective rank (participation ratio)
            S_normalized = singular_values / singular_values.sum()
            effective_rank = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-10)))
            
            # Top 5 singular values
            top_singular_values = singular_values[:5].tolist()
        except Exception as e:
            logger.warning(f"SVD failed for {layer_name}: {e}")
            effective_rank = 0.0
            top_singular_values = []
        
        return LayerLoRAStats(
            layer_name=layer_name,
            lora_A_norm=lora_A_norm,
            lora_B_norm=lora_B_norm,
            lora_product_norm=lora_product_norm,
            effective_rank=effective_rank,
            top_singular_values=top_singular_values
        )
    
    def analyze_all_layers(self) -> List[LayerLoRAStats]:
        """
        Analyze all LoRA layers in the model.
        
        Returns:
            List of LayerLoRAStats for each layer
        """
        weights = self.load_lora_weights()
        
        # Group weights by layer
        layers = {}
        for key, value in weights.items():
            if 'lora_A' in key:
                base_name = key.replace('.lora_A.weight', '').replace('.lora_A.default.weight', '')
                if base_name not in layers:
                    layers[base_name] = {}
                layers[base_name]['A'] = value
            elif 'lora_B' in key:
                base_name = key.replace('.lora_B.weight', '').replace('.lora_B.default.weight', '')
                if base_name not in layers:
                    layers[base_name] = {}
                layers[base_name]['B'] = value
        
        # Analyze each layer
        all_stats = []
        logger.info(f"Analyzing {len(layers)} LoRA layers...")
        
        for layer_name, matrices in sorted(layers.items()):
            if 'A' in matrices and 'B' in matrices:
                try:
                    stats = self.analyze_layer(
                        matrices['A'], 
                        matrices['B'],
                        layer_name
                    )
                    all_stats.append(stats)
                    
                    if len(all_stats) % 10 == 0:
                        logger.info(f"  Analyzed {len(all_stats)} layers...")
                except Exception as e:
                    logger.warning(f"Failed to analyze {layer_name}: {e}")
        
        logger.info(f"✅ Analyzed {len(all_stats)} LoRA layers")
        return all_stats
    
    def save_analysis(self, stats: List[LayerLoRAStats], filepath: str):
        """Save analysis results to JSON."""
        data = {
            'model_path': str(self.model_path),
            'n_layers': len(stats),
            'layers': [s.to_dict() for s in stats]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Saved LoRA analysis to {filepath}")
    
    def plot_layer_heatmap(
        self, 
        stats: List[LayerLoRAStats],
        save_path: str,
        title: str = "LoRA Layer Analysis"
    ):
        """
        Plot heatmap showing which layers have largest updates.
        
        Args:
            stats: List of layer statistics
            save_path: Where to save plot
            title: Plot title
        """
        # Extract data
        layer_names = [s.layer_name for s in stats]
        product_norms = [s.lora_product_norm for s in stats]
        effective_ranks = [s.effective_rank for s in stats]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, max(8, len(layer_names) * 0.3)))
        
        # Plot 1: Product norms (magnitude of update)
        ax = axes[0]
        y_pos = np.arange(len(layer_names))
        ax.barh(y_pos, product_norms, color='#2ca02c', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
        ax.set_xlabel('||B·A|| (Update Magnitude)', fontweight='bold')
        ax.set_title('Layer Update Magnitudes', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Effective ranks
        ax = axes[1]
        ax.barh(y_pos, effective_ranks, color='#1f77b4', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
        ax.set_xlabel('Effective Rank', fontweight='bold')
        ax.set_title('Layer Effective Ranks', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        try:
            plt.tight_layout()
        except:
            pass
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved layer heatmap to {save_path}")
        plt.close()


def compare_lora_models(
    baseline_path: str,
    polychromic_path: str,
    output_dir: str
):
    """
    Compare LoRA parameters between baseline and polychromic models.
    
    Args:
        baseline_path: Path to baseline model
        polychromic_path: Path to polychromic model
        output_dir: Where to save comparison results
    """
    logger.info("="*60)
    logger.info("Comparing LoRA Models")
    logger.info("="*60)
    
    # Analyze both models
    logger.info("\n1. Analyzing baseline model...")
    baseline_analyzer = LoRAAnalyzer(baseline_path)
    baseline_stats = baseline_analyzer.analyze_all_layers()
    
    logger.info("\n2. Analyzing polychromic model...")
    polychromic_analyzer = LoRAAnalyzer(polychromic_path)
    polychromic_stats = polychromic_analyzer.analyze_all_layers()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual analyses
    baseline_analyzer.save_analysis(
        baseline_stats, 
        str(output_path / 'baseline_lora_analysis.json')
    )
    polychromic_analyzer.save_analysis(
        polychromic_stats,
        str(output_path / 'polychromic_lora_analysis.json')
    )
    
    # Plot individual heatmaps
    baseline_analyzer.plot_layer_heatmap(
        baseline_stats,
        str(output_path / 'baseline_lora_heatmap.pdf'),
        title='Baseline LoRA Layer Analysis'
    )
    polychromic_analyzer.plot_layer_heatmap(
        polychromic_stats,
        str(output_path / 'polychromic_lora_heatmap.pdf'),
        title='Polychromic LoRA Layer Analysis'
    )
    
    # Create comparison plot
    logger.info("\n3. Creating comparison plot...")
    plot_comparison(baseline_stats, polychromic_stats, output_path)
    
    # Compute and save summary statistics
    logger.info("\n4. Computing summary statistics...")
    summary = compute_comparison_summary(baseline_stats, polychromic_stats)
    
    with open(output_path / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("✅ LoRA Comparison Complete!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  • baseline_lora_analysis.json")
    logger.info(f"  • polychromic_lora_analysis.json")
    logger.info(f"  • baseline_lora_heatmap.pdf")
    logger.info(f"  • polychromic_lora_heatmap.pdf")
    logger.info(f"  • lora_comparison.pdf")
    logger.info(f"  • comparison_summary.json")
    logger.info("="*60)
    
    return summary


def plot_comparison(
    baseline_stats: List[LayerLoRAStats],
    polychromic_stats: List[LayerLoRAStats],
    output_dir: Path
):
    """Create comparison plot between baseline and polychromic."""
    # Match layers
    baseline_dict = {s.layer_name: s for s in baseline_stats}
    polychromic_dict = {s.layer_name: s for s in polychromic_stats}
    
    common_layers = sorted(set(baseline_dict.keys()) & set(polychromic_dict.keys()))
    
    if not common_layers:
        logger.warning("No common layers found for comparison")
        return
    
    # Extract data for common layers
    layer_names = common_layers
    baseline_norms = [baseline_dict[l].lora_product_norm for l in common_layers]
    polychromic_norms = [polychromic_dict[l].lora_product_norm for l in common_layers]
    
    # Compute differences
    differences = [p - b for p, b in zip(polychromic_norms, baseline_norms)]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(layer_names) * 0.3)))
    
    # Plot 1: Side-by-side comparison
    ax = axes[0]
    y_pos = np.arange(len(layer_names))
    width = 0.35
    
    ax.barh(y_pos - width/2, baseline_norms, width, 
           label='Baseline', color='#1f77b4', alpha=0.7)
    ax.barh(y_pos + width/2, polychromic_norms, width,
           label='Polychromic', color='#ff7f0e', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
    ax.set_xlabel('||B·A|| (Update Magnitude)', fontweight='bold')
    ax.set_title('Layer-wise Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Differences
    ax = axes[1]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in differences]
    ax.barh(y_pos, differences, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
    ax.set_xlabel('Difference (Polychromic - Baseline)', fontweight='bold')
    ax.set_title('Update Magnitude Differences', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('LoRA Parameter Comparison: Baseline vs. Polychromic', 
                fontsize=16, fontweight='bold')
    
    try:
        plt.tight_layout()
    except:
        pass
    
    save_path = output_dir / 'lora_comparison.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved comparison plot to {save_path}")
    plt.close()


def compute_comparison_summary(
    baseline_stats: List[LayerLoRAStats],
    polychromic_stats: List[LayerLoRAStats]
) -> Dict:
    """Compute summary statistics for comparison."""
    baseline_dict = {s.layer_name: s for s in baseline_stats}
    polychromic_dict = {s.layer_name: s for s in polychromic_stats}
    
    common_layers = sorted(set(baseline_dict.keys()) & set(polychromic_dict.keys()))
    
    # Compute aggregate statistics
    baseline_norms = [baseline_dict[l].lora_product_norm for l in common_layers]
    polychromic_norms = [polychromic_dict[l].lora_product_norm for l in common_layers]
    
    summary = {
        'n_common_layers': len(common_layers),
        'baseline': {
            'mean_norm': float(np.mean(baseline_norms)),
            'std_norm': float(np.std(baseline_norms)),
            'max_norm': float(np.max(baseline_norms)),
            'max_norm_layer': common_layers[np.argmax(baseline_norms)]
        },
        'polychromic': {
            'mean_norm': float(np.mean(polychromic_norms)),
            'std_norm': float(np.std(polychromic_norms)),
            'max_norm': float(np.max(polychromic_norms)),
            'max_norm_layer': common_layers[np.argmax(polychromic_norms)]
        },
        'differences': {
            'mean_difference': float(np.mean([p - b for p, b in zip(polychromic_norms, baseline_norms)])),
            'layers_with_higher_polychromic': sum(1 for p, b in zip(polychromic_norms, baseline_norms) if p > b),
            'layers_with_higher_baseline': sum(1 for p, b in zip(polychromic_norms, baseline_norms) if b > p)
        }
    }
    
    logger.info("\n📊 Summary Statistics:")
    logger.info(f"  Common layers: {summary['n_common_layers']}")
    logger.info(f"  Baseline mean norm: {summary['baseline']['mean_norm']:.4f}")
    logger.info(f"  Polychromic mean norm: {summary['polychromic']['mean_norm']:.4f}")
    logger.info(f"  Mean difference: {summary['differences']['mean_difference']:.4f}")
    logger.info(f"  Layers where polychromic > baseline: {summary['differences']['layers_with_higher_polychromic']}")
    
    return summary


# Test
if __name__ == "__main__":
    print("✅ LoRA analysis module loaded successfully")
    print("This module provides NOVEL analysis - no existing paper does this!")
    print("")
    print("Usage:")
    print("  analyzer = LoRAAnalyzer('path/to/model')")
    print("  stats = analyzer.analyze_all_layers()")
    print("  analyzer.plot_layer_heatmap(stats, 'heatmap.pdf')")

