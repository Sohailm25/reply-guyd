#!/usr/bin/env python3
"""
Analyze LoRA Parameters

Understand WHERE diversity is encoded in the model by analyzing LoRA matrices.
This is NOVEL analysis - no existing paper does this.

Usage:
    # Analyze single model:
    python scripts/analysis/analyze_lora_parameters.py \
        --model output/experiments/polychromic_0.3/seed_42 \
        --output output/analysis/lora_polychromic
    
    # Compare two models:
    python scripts/analysis/analyze_lora_parameters.py \
        --baseline output/experiments/baseline/seed_42 \
        --polychromic output/experiments/polychromic_0.3/seed_42 \
        --output output/analysis/lora_comparison
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.lora_analysis import (
    LoRAAnalyzer,
    compare_lora_models
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LoRA parameters to understand model changes"
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to single model to analyze'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        help='Path to baseline model (for comparison)'
    )
    parser.add_argument(
        '--polychromic',
        type=str,
        help='Path to polychromic model (for comparison)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    
    args = parser.parse_args()
    
    # Check arguments
    if args.model and (args.baseline or args.polychromic):
        logger.error("Use either --model (single) or --baseline + --polychromic (comparison)")
        sys.exit(1)
    
    if not args.model and not (args.baseline and args.polychromic):
        logger.error("Must provide either --model or both --baseline and --polychromic")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("LoRA Parameter Analysis")
    logger.info("="*60)
    
    if args.model:
        # Single model analysis
        logger.info(f"\nAnalyzing single model: {args.model}")
        
        analyzer = LoRAAnalyzer(args.model)
        stats = analyzer.analyze_all_layers()
        
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        analyzer.save_analysis(stats, str(output_path / 'lora_analysis.json'))
        
        # Plot heatmap
        model_name = Path(args.model).parent.name
        analyzer.plot_layer_heatmap(
            stats,
            str(output_path / 'lora_heatmap.pdf'),
            title=f'LoRA Analysis: {model_name}'
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ Analysis Complete!")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {args.output}")
        logger.info(f"  • lora_analysis.json")
        logger.info(f"  • lora_heatmap.pdf")
        logger.info("="*60)
        
    else:
        # Comparison analysis
        logger.info(f"\nComparing models:")
        logger.info(f"  Baseline:    {args.baseline}")
        logger.info(f"  Polychromic: {args.polychromic}")
        
        summary = compare_lora_models(
            args.baseline,
            args.polychromic,
            args.output
        )
        
        # Print key findings
        logger.info("\n" + "="*60)
        logger.info("🔍 KEY FINDINGS")
        logger.info("="*60)
        
        mean_diff = summary['differences']['mean_difference']
        if mean_diff > 0:
            logger.info(f"✅ Polychromic has HIGHER average update magnitude (+{mean_diff:.4f})")
            logger.info(f"   This suggests more substantial parameter changes")
        else:
            logger.info(f"⚠️  Baseline has HIGHER average update magnitude ({mean_diff:.4f})")
        
        n_higher_poly = summary['differences']['layers_with_higher_polychromic']
        n_higher_base = summary['differences']['layers_with_higher_baseline']
        total = summary['n_common_layers']
        
        logger.info(f"\n📊 Layer-wise comparison:")
        logger.info(f"   Polychromic > Baseline: {n_higher_poly}/{total} layers ({n_higher_poly/total*100:.1f}%)")
        logger.info(f"   Baseline > Polychromic: {n_higher_base}/{total} layers ({n_higher_base/total*100:.1f}%)")
        
        logger.info(f"\n🎯 Largest updates:")
        logger.info(f"   Baseline:    {summary['baseline']['max_norm_layer'].split('.')[-1]}")
        logger.info(f"                (norm = {summary['baseline']['max_norm']:.4f})")
        logger.info(f"   Polychromic: {summary['polychromic']['max_norm_layer'].split('.')[-1]}")
        logger.info(f"                (norm = {summary['polychromic']['max_norm']:.4f})")
        
        logger.info("="*60)


if __name__ == "__main__":
    main()

