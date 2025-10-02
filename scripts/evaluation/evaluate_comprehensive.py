#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

Evaluates trained models on all metrics:
- Diversity metrics (Self-BLEU, Distinct-n, semantic diversity)
- Quality metrics (ROUGE, BERTScore)
- LLM-as-judge evaluation
- Pass@k evaluation
- Statistical significance tests

Usage:
    python scripts/evaluation/evaluate_comprehensive.py \
        --baseline output/experiments/baseline/seed_42 \
        --polychromic output/experiments/polychromic_0.3/seed_42 \
        --test-data data/processed/test_data.jsonl \
        --output output/evaluation/
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import (
    compute_all_diversity_metrics,
    compute_rouge_scores,
    comprehensive_statistical_comparison,
)
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.passk_evaluation import compare_passk, HeuristicQualityChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model_path: str = "./"):
    """Load a trained model with LoRA adapters."""
    logger.info(f"Loading model from: {model_path}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_replies(
    model,
    tokenizer,
    prompts: List[str],
    n_per_prompt: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 100
) -> List[List[str]]:
    """Generate multiple replies for each prompt."""
    logger.info(f"Generating {n_per_prompt} replies for {len(prompts)} prompts...")
    
    all_replies = []
    
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(prompts)}")
        
        # Format as chat
        messages = [{
            "role": "user",
            "content": f"Generate an engaging Twitter reply to this tweet:\n\n{prompt}\n\nReply:"
        }]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate multiple replies
        replies = []
        for _ in range(n_per_prompt):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            reply = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            ).strip()
            
            replies.append(reply)
        
        all_replies.append(replies)
    
    return all_replies


def load_test_data(test_data_path: str, max_examples: int = 100) -> List[Dict]:
    """Load test data."""
    logger.info(f"Loading test data from: {test_data_path}")
    
    examples = []
    with open(test_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line)
            examples.append({
                'tweet_id': data['tweet_id'],
                'tweet': data['tweet'],
                'reference_reply': data['reply'],
                'reply_likes': data.get('reply_likes', 0)
            })
    
    logger.info(f"Loaded {len(examples)} test examples")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline model')
    parser.add_argument('--polychromic', type=str, required=True, help='Path to polychromic model')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--base-model', type=str, default='./', help='Base model path')
    parser.add_argument('--max-examples', type=int, default=100, help='Max test examples')
    parser.add_argument('--n-generations', type=int, default=10, help='Generations per prompt')
    parser.add_argument('--skip-llm-judge', action='store_true', help='Skip LLM-as-judge (expensive)')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key for LLM judge')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Baseline: {args.baseline}")
    logger.info(f"Polychromic: {args.polychromic}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output: {args.output}")
    logger.info("="*60 + "\n")
    
    # Load test data
    test_examples = load_test_data(args.test_data, args.max_examples)
    prompts = [ex['tweet'] for ex in test_examples]
    references = [ex['reference_reply'] for ex in test_examples]
    
    # Load models
    logger.info("\n" + "="*60)
    logger.info("LOADING MODELS")
    logger.info("="*60)
    
    baseline_model, baseline_tokenizer = load_model(args.baseline, args.base_model)
    polychromic_model, polychromic_tokenizer = load_model(args.polychromic, args.base_model)
    
    # Generate replies
    logger.info("\n" + "="*60)
    logger.info("GENERATING REPLIES")
    logger.info("="*60)
    
    logger.info("\nBaseline model:")
    baseline_replies = generate_replies(
        baseline_model,
        baseline_tokenizer,
        prompts,
        n_per_prompt=args.n_generations
    )
    
    logger.info("\nPolychromic model:")
    polychromic_replies = generate_replies(
        polychromic_model,
        polychromic_tokenizer,
        prompts,
        n_per_prompt=args.n_generations
    )
    
    # Flatten for diversity metrics
    baseline_flat = [r for replies in baseline_replies for r in replies]
    polychromic_flat = [r for replies in polychromic_replies for r in replies]
    
    # ===== DIVERSITY METRICS =====
    logger.info("\n" + "="*60)
    logger.info("DIVERSITY METRICS")
    logger.info("="*60)
    
    logger.info("\nBaseline model:")
    baseline_diversity = compute_all_diversity_metrics(baseline_flat)
    
    logger.info("\nPolychromic model:")
    polychromic_diversity = compute_all_diversity_metrics(polychromic_flat)
    
    # Save diversity metrics
    diversity_results = {
        'baseline': baseline_diversity,
        'polychromic': polychromic_diversity,
    }
    
    with open(output_dir / 'diversity_metrics.json', 'w') as f:
        json.dump(diversity_results, f, indent=2)
    
    # ===== QUALITY METRICS =====
    logger.info("\n" + "="*60)
    logger.info("QUALITY METRICS")
    logger.info("="*60)
    
    # Use first generation for quality comparison
    baseline_first = [replies[0] for replies in baseline_replies]
    polychromic_first = [replies[0] for replies in polychromic_replies]
    
    logger.info("\nComputing ROUGE scores...")
    baseline_rouge = compute_rouge_scores(baseline_first, references)
    polychromic_rouge = compute_rouge_scores(polychromic_first, references)
    
    quality_results = {
        'baseline': {'rouge': baseline_rouge},
        'polychromic': {'rouge': polychromic_rouge},
    }
    
    with open(output_dir / 'quality_metrics.json', 'w') as f:
        json.dump(quality_results, f, indent=2)
    
    # ===== PASS@K EVALUATION =====
    logger.info("\n" + "="*60)
    logger.info("PASS@K EVALUATION")
    logger.info("="*60)
    
    quality_checker = HeuristicQualityChecker(min_length=10, max_length=300)
    
    # Define generation functions
    def baseline_gen(prompt, n):
        idx = prompts.index(prompt)
        return baseline_replies[idx][:n]
    
    def polychromic_gen(prompt, n):
        idx = prompts.index(prompt)
        return polychromic_replies[idx][:n]
    
    passk_results = compare_passk(
        prompts,
        baseline_gen,
        polychromic_gen,
        quality_checker,
        k_values=[1, 5, 10],
        n_total=args.n_generations
    )
    
    with open(output_dir / 'passk_results.json', 'w') as f:
        json.dump(passk_results, f, indent=2)
    
    # ===== LLM-AS-JUDGE =====
    if not args.skip_llm_judge and args.anthropic_key:
        logger.info("\n" + "="*60)
        logger.info("LLM-AS-JUDGE EVALUATION")
        logger.info("="*60)
        
        judge = LLMJudge(api_key=args.anthropic_key)
        
        # Prepare pairs
        judge_examples = [
            {
                'tweet': prompts[i],
                'reply_a': baseline_first[i],
                'reply_b': polychromic_first[i]
            }
            for i in range(len(prompts))
        ]
        
        # Evaluate
        judge_results = judge.batch_evaluate(judge_examples, verbose=True)
        win_rates = judge.compute_win_rates(judge_results)
        
        logger.info("\n" + "="*60)
        logger.info("LLM Judge Results:")
        logger.info(f"  Baseline wins: {win_rates['model_a_wins']} ({win_rates['model_a_win_rate']*100:.1f}%)")
        logger.info(f"  Polychromic wins: {win_rates['model_b_wins']} ({win_rates['model_b_win_rate']*100:.1f}%)")
        logger.info(f"  Ties: {win_rates['ties']}")
        logger.info(f"  Total cost: ${judge.total_cost:.2f}")
        logger.info("="*60)
        
        with open(output_dir / 'llm_judge_results.json', 'w') as f:
            json.dump({
                'results': judge_results,
                'win_rates': win_rates,
                'total_cost': judge.total_cost
            }, f, indent=2)
    
    # ===== STATISTICAL TESTS =====
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("="*60)
    
    # Test on diversity scores
    stats_results = {}
    
    # Self-BLEU (lower is better for diversity)
    stats_results['self_bleu'] = comprehensive_statistical_comparison(
        [baseline_diversity['self_bleu']],  # Single value, but wrap in list
        [polychromic_diversity['self_bleu']],
        "Self-BLEU"
    )
    
    # Distinct-2 (higher is better)
    stats_results['distinct_2'] = comprehensive_statistical_comparison(
        [baseline_diversity['distinct_2']],
        [polychromic_diversity['distinct_2']],
        "Distinct-2"
    )
    
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # ===== SUMMARY REPORT =====
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    summary = {
        'diversity': {
            'baseline_self_bleu': baseline_diversity['self_bleu'],
            'polychromic_self_bleu': polychromic_diversity['self_bleu'],
            'baseline_distinct_2': baseline_diversity['distinct_2'],
            'polychromic_distinct_2': polychromic_diversity['distinct_2'],
            'baseline_semantic_div': baseline_diversity['semantic_diversity'],
            'polychromic_semantic_div': polychromic_diversity['semantic_diversity'],
        },
        'quality': {
            'baseline_rouge_l': baseline_rouge['rougeL'],
            'polychromic_rouge_l': polychromic_rouge['rougeL'],
        },
        'passk': passk_results,
    }
    
    logger.info("\nDiversity Metrics:")
    logger.info(f"  Self-BLEU (lower=better):")
    logger.info(f"    Baseline: {baseline_diversity['self_bleu']:.4f}")
    logger.info(f"    Polychromic: {polychromic_diversity['self_bleu']:.4f}")
    logger.info(f"  Distinct-2 (higher=better):")
    logger.info(f"    Baseline: {baseline_diversity['distinct_2']:.4f}")
    logger.info(f"    Polychromic: {polychromic_diversity['distinct_2']:.4f}")
    logger.info(f"  Semantic Diversity (higher=better):")
    logger.info(f"    Baseline: {baseline_diversity['semantic_diversity']:.4f}")
    logger.info(f"    Polychromic: {polychromic_diversity['semantic_diversity']:.4f}")
    
    logger.info("\nQuality Metrics:")
    logger.info(f"  ROUGE-L:")
    logger.info(f"    Baseline: {baseline_rouge['rougeL']:.4f}")
    logger.info(f"    Polychromic: {polychromic_rouge['rougeL']:.4f}")
    
    logger.info("="*60)
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("="*60)
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

