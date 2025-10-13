#!/usr/bin/env python3
"""
Clean Evaluation Script Using Refactored Architecture

Demonstrates the new, cleaner way to evaluate models:
- Environment abstraction
- Unified benchmark runner
- Centralized model loading

Much simpler than the original evaluate_comprehensive.py!
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# New organized imports
from src.models import load_model_with_lora
from src.environments import TwitterReplyEnvironment
from src.evaluation.benchmark import EvaluationBenchmark, compare_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_data_path: str):
    """Load test data from JSONL file."""
    test_data = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate models with clean architecture")
    parser.add_argument(
        '--baseline-lora',
        type=str,
        help='Path to baseline LoRA checkpoint'
    )
    parser.add_argument(
        '--polychromic-lora',
        type=str,
        help='Path to polychromic LoRA checkpoint'
    )
    parser.add_argument(
        '--grpo-lora',
        type=str,
        help='Path to GRPO LoRA checkpoint'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data (JSONL)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='.',
        help='Path to base model (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-generations',
        type=int,
        default=10,
        help='Number of generations per example'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum number of examples to evaluate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.9,
        help='Generation temperature'
    )
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"📊 Loading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data)
    logger.info(f"  Loaded {len(test_data)} test examples")
    
    # Collect models to evaluate
    models_to_evaluate = {}
    
    # Load each specified model
    if args.baseline_lora:
        logger.info(f"\n📦 Loading baseline model from: {args.baseline_lora}")
        model, tokenizer = load_model_with_lora(
            base_model_path=args.base_model,
            lora_path=args.baseline_lora
        )
        env = TwitterReplyEnvironment(model, tokenizer)
        models_to_evaluate['baseline'] = env
    
    if args.polychromic_lora:
        logger.info(f"\n📦 Loading polychromic model from: {args.polychromic_lora}")
        model, tokenizer = load_model_with_lora(
            base_model_path=args.base_model,
            lora_path=args.polychromic_lora
        )
        env = TwitterReplyEnvironment(model, tokenizer)
        models_to_evaluate['polychromic'] = env
    
    if args.grpo_lora:
        logger.info(f"\n📦 Loading GRPO model from: {args.grpo_lora}")
        model, tokenizer = load_model_with_lora(
            base_model_path=args.base_model,
            lora_path=args.grpo_lora
        )
        env = TwitterReplyEnvironment(model, tokenizer)
        models_to_evaluate['grpo'] = env
    
    if not models_to_evaluate:
        logger.error("❌ No models specified! Use --baseline-lora, --polychromic-lora, or --grpo-lora")
        sys.exit(1)
    
    logger.info(f"\n✅ Loaded {len(models_to_evaluate)} model(s): {', '.join(models_to_evaluate.keys())}")
    
    # Run evaluation
    if len(models_to_evaluate) == 1:
        # Single model evaluation
        name = list(models_to_evaluate.keys())[0]
        env = list(models_to_evaluate.values())[0]
        
        logger.info(f"\n🔬 Evaluating {name} model...")
        benchmark = EvaluationBenchmark(env, test_data)
        results = benchmark.run(
            n_generations=args.n_generations,
            max_examples=args.max_examples,
            temperature=args.temperature
        )
        
        # Print summary
        benchmark.print_summary(results)
        
        # Save results
        output_path = Path(args.output) / f"{name}_results.json"
        benchmark.save_results(results, str(output_path))
        
    else:
        # Multi-model comparison
        logger.info(f"\n🔬 Comparing {len(models_to_evaluate)} models...")
        results = compare_models(
            models_to_evaluate,
            test_data,
            n_generations=args.n_generations,
            max_examples=args.max_examples
        )
        
        # Print summaries for each model
        for name, model_results in results["models"].items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Results for: {name}")
            logger.info(f"{'='*60}")
            benchmark = EvaluationBenchmark(models_to_evaluate[name], test_data)
            benchmark.print_summary(model_results)
        
        # Save combined results
        output_path = Path(args.output) / "comparison_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Results saved to: {output_path}")
    
    logger.info("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

