#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

Evaluates multiple model variants:
- Zero-shot: Base Qwen3 with simple prompt
- Prompt-engineered: Base Qwen3 with carefully crafted prompt
- Baseline LoRA: Fine-tuned with standard supervised learning
- Polychromic LoRA: Fine-tuned with diversity-aware objective

Metrics computed:
- Diversity metrics (Self-BLEU, Distinct-n, semantic diversity)
- Quality metrics (ROUGE, BERTScore)
- LLM-as-judge evaluation
- Pass@k evaluation
- Statistical significance tests

Usage:
    # All four models
    python scripts/evaluation/evaluate_comprehensive.py \
        --include-zero-shot \
        --include-prompt-engineered \
        --baseline-lora output/experiments/baseline/seed_42 \
        --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
        --test-data data/processed/test_data.jsonl \
        --output output/evaluation/
    
    # Just LoRA models
    python scripts/evaluation/evaluate_comprehensive.py \
        --baseline-lora output/experiments/baseline/seed_42 \
        --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
        --test-data data/processed/test_data.jsonl \
        --output output/evaluation/
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import List, Dict, Callable
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
from src.evaluation.passk_evaluation import compute_passk, HeuristicQualityChecker
from src.evaluation.prompt_templates import PROMPT_VARIANTS, get_prompt_description

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model_path: str = "./"):
    """Load a trained model with LoRA adapters."""
    logger.info(f"Loading LoRA model from: {model_path}")
    
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


def load_base_model_only(base_model_path: str = "./"):
    """Load base model without LoRA adapters for zero-shot/prompt-engineered baselines."""
    logger.info(f"Loading base model (no LoRA) from: {base_model_path}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
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


def generate_with_prompt_template(
    model,
    tokenizer,
    prompts: List[str],
    prompt_template_fn: Callable[[str], str],
    n_per_prompt: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 100
) -> List[List[str]]:
    """
    Generate replies using base model with custom prompt template.
    
    Args:
        model: Base model (no LoRA)
        tokenizer: Tokenizer
        prompts: List of tweets to reply to
        prompt_template_fn: Function that takes tweet and returns formatted prompt
        n_per_prompt: Number of generations per prompt
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of lists of generated replies
    """
    logger.info(f"Generating {n_per_prompt} replies for {len(prompts)} prompts with custom prompt template...")
    
    all_replies = []
    
    for i, tweet in enumerate(prompts):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(prompts)}")
        
        # Get formatted prompt from template
        prompt_text = prompt_template_fn(tweet)
        
        # Format as Qwen3 chat
        messages = [{
            "role": "user",
            "content": prompt_text
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
    
    # Model paths (all optional - select which ones to evaluate)
    parser.add_argument('--baseline-lora', type=str, help='Path to baseline LoRA model')
    parser.add_argument('--polychromic-lora', type=str, help='Path to polychromic LoRA model')
    parser.add_argument('--base-model', type=str, default='./', help='Base model path (Qwen3-8B)')
    
    # Prompt-based baselines (no training needed)
    parser.add_argument('--include-zero-shot', action='store_true',
                       help='Include zero-shot baseline (simple prompt)')
    parser.add_argument('--include-prompt-engineered', action='store_true',
                       help='Include prompt-engineered baseline (optimized prompt)')
    parser.add_argument('--prompt-variant', type=str, default='with_examples',
                       choices=list(PROMPT_VARIANTS.keys()),
                       help='Which prompt template to use for prompt-engineered baseline')
    
    # Data and evaluation settings
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--max-examples', type=int, default=100, help='Max test examples')
    parser.add_argument('--n-generations', type=int, default=10, help='Generations per prompt')
    parser.add_argument('--skip-llm-judge', action='store_true', help='Skip LLM-as-judge (expensive)')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key for LLM judge')
    
    args = parser.parse_args()
    
    # Validate that at least one model is specified
    if not any([args.baseline_lora, args.polychromic_lora, args.include_zero_shot, args.include_prompt_engineered]):
        parser.error("Must specify at least one model to evaluate")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Models to evaluate:")
    if args.include_zero_shot:
        logger.info(f"  ✓ Zero-shot (base Qwen3, simple prompt)")
    if args.include_prompt_engineered:
        logger.info(f"  ✓ Prompt-engineered ({args.prompt_variant})")
        logger.info(f"    Description: {get_prompt_description(args.prompt_variant)}")
    if args.baseline_lora:
        logger.info(f"  ✓ Baseline LoRA ({args.baseline_lora})")
    if args.polychromic_lora:
        logger.info(f"  ✓ Polychromic LoRA ({args.polychromic_lora})")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output: {args.output}")
    logger.info("="*60 + "\n")
    
    # Load test data
    test_examples = load_test_data(args.test_data, args.max_examples)
    prompts = [ex['tweet'] for ex in test_examples]
    references = [ex['reference_reply'] for ex in test_examples]
    
    # Dictionary to store all model replies
    all_model_replies = {}
    
    # Load base model if needed (for zero-shot or prompt-engineered)
    base_model = None
    base_tokenizer = None
    
    if args.include_zero_shot or args.include_prompt_engineered:
        logger.info("\n" + "="*60)
        logger.info("LOADING BASE MODEL (for prompt-based baselines)")
        logger.info("="*60)
        base_model, base_tokenizer = load_base_model_only(args.base_model)
    
    # Generate replies for each selected model
    logger.info("\n" + "="*60)
    logger.info("GENERATING REPLIES")
    logger.info("="*60)
    
    # 1. Zero-shot baseline
    if args.include_zero_shot:
        logger.info("\n[1/N] Zero-Shot Baseline:")
        all_model_replies['zero_shot'] = generate_with_prompt_template(
            base_model,
            base_tokenizer,
            prompts,
            PROMPT_VARIANTS['zero_shot'],
            n_per_prompt=args.n_generations
        )
    
    # 2. Prompt-engineered baseline
    if args.include_prompt_engineered:
        logger.info(f"\n[2/N] Prompt-Engineered Baseline (variant: {args.prompt_variant}):")
        all_model_replies['prompt_engineered'] = generate_with_prompt_template(
            base_model,
            base_tokenizer,
            prompts,
            PROMPT_VARIANTS[args.prompt_variant],
            n_per_prompt=args.n_generations
        )
    
    # 3. Baseline LoRA
    if args.baseline_lora:
        logger.info("\n" + "="*60)
        logger.info("LOADING BASELINE LORA")
        logger.info("="*60)
        baseline_lora_model, baseline_lora_tokenizer = load_model(args.baseline_lora, args.base_model)
        
        logger.info("\n[3/N] Baseline LoRA:")
        all_model_replies['baseline_lora'] = generate_replies(
            baseline_lora_model,
            baseline_lora_tokenizer,
            prompts,
            n_per_prompt=args.n_generations
        )
    
    # 4. Polychromic LoRA
    if args.polychromic_lora:
        logger.info("\n" + "="*60)
        logger.info("LOADING POLYCHROMIC LORA")
        logger.info("="*60)
        polychromic_lora_model, polychromic_lora_tokenizer = load_model(args.polychromic_lora, args.base_model)
        
        logger.info("\n[4/N] Polychromic LoRA:")
        all_model_replies['polychromic_lora'] = generate_replies(
            polychromic_lora_model,
            polychromic_lora_tokenizer,
            prompts,
            n_per_prompt=args.n_generations
        )
    
    # ===== COMPUTE METRICS FOR ALL MODELS =====
    logger.info("\n" + "="*60)
    logger.info("COMPUTING METRICS FOR ALL MODELS")
    logger.info("="*60)
    
    diversity_results = {}
    quality_results = {}
    
    for model_name, replies in all_model_replies.items():
        logger.info(f"\n{model_name}:")
        
        # Diversity metrics (use all generations)
        flat_replies = [r for reply_list in replies for r in reply_list]
        logger.info("  Computing diversity metrics...")
        diversity_results[model_name] = compute_all_diversity_metrics(flat_replies)
        
        # Quality metrics (use first generation only)
        first_replies = [reply_list[0] for reply_list in replies]
        logger.info("  Computing quality metrics...")
        quality_results[model_name] = {
            'rouge': compute_rouge_scores(first_replies, references)
        }
    
    # Save metrics
    with open(output_dir / 'diversity_metrics.json', 'w') as f:
        json.dump(diversity_results, f, indent=2)
    
    with open(output_dir / 'quality_metrics.json', 'w') as f:
        json.dump(quality_results, f, indent=2)
    
    # ===== PASS@K EVALUATION =====
    logger.info("\n" + "="*60)
    logger.info("PASS@K EVALUATION")
    logger.info("="*60)
    
    quality_checker = HeuristicQualityChecker(min_length=10, max_length=300)
    
    passk_results = {}
    for model_name, replies in all_model_replies.items():
        logger.info(f"\n{model_name}:")
        
        # Create generation function for this model
        def model_gen(prompt, n, replies=replies):
            idx = prompts.index(prompt)
            return replies[idx][:n]
        
        # Compute Pass@k for this model
        model_passk = compute_passk(
            prompts,
            model_gen,
            quality_checker,
            k_values=[1, 5, 10],
            n_total=args.n_generations,
            verbose=False
        )
        
        passk_results[model_name] = model_passk
    
    with open(output_dir / 'passk_results.json', 'w') as f:
        json.dump(passk_results, f, indent=2)
    
    # ===== LLM-AS-JUDGE =====
    if not args.skip_llm_judge and args.anthropic_key:
        logger.info("\n" + "="*60)
        logger.info("LLM-AS-JUDGE EVALUATION")
        logger.info("="*60)
        
        judge = LLMJudge(api_key=args.anthropic_key)
        
        model_names = list(all_model_replies.keys())
        all_judge_results = {}
        
        # Pairwise comparisons between all models
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                logger.info(f"\nComparing {model_a} vs {model_b}:")
                
                first_replies_a = [reply_list[0] for reply_list in all_model_replies[model_a]]
                first_replies_b = [reply_list[0] for reply_list in all_model_replies[model_b]]
                
                # Prepare pairs
                judge_examples = [
                    {
                        'tweet': prompts[j],
                        'reply_a': first_replies_a[j],
                        'reply_b': first_replies_b[j]
                    }
                    for j in range(len(prompts))
                ]
                
                # Evaluate
                results = judge.batch_evaluate(judge_examples, verbose=True)
                win_rates = judge.compute_win_rates(results)
                
                comparison_key = f"{model_a}_vs_{model_b}"
                all_judge_results[comparison_key] = {
                    'model_a': model_a,
                    'model_b': model_b,
                    'results': results,
                    'win_rates': win_rates
                }
                
                logger.info(f"  {model_a} wins: {win_rates['model_a_wins']} ({win_rates['model_a_win_rate']*100:.1f}%)")
                logger.info(f"  {model_b} wins: {win_rates['model_b_wins']} ({win_rates['model_b_win_rate']*100:.1f}%)")
                logger.info(f"  Ties: {win_rates['ties']}")
        
        logger.info(f"\nTotal LLM-as-judge cost: ${judge.total_cost:.2f}")
        
        with open(output_dir / 'llm_judge_results.json', 'w') as f:
            json.dump({
                'comparisons': all_judge_results,
                'total_cost': judge.total_cost
            }, f, indent=2)
    
    # ===== STATISTICAL TESTS =====
    # Only compute if we have LoRA models to compare
    if args.baseline_lora and args.polychromic_lora:
        logger.info("\n" + "="*60)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("="*60)
        logger.info("(Comparing Baseline LoRA vs Polychromic LoRA)")
        
        stats_results = {}
        
        # Self-BLEU (lower is better for diversity)
        stats_results['self_bleu'] = comprehensive_statistical_comparison(
            [diversity_results['baseline_lora']['self_bleu']],
            [diversity_results['polychromic_lora']['self_bleu']],
            "Self-BLEU"
        )
        
        # Distinct-2 (higher is better)
        stats_results['distinct_2'] = comprehensive_statistical_comparison(
            [diversity_results['baseline_lora']['distinct_2']],
            [diversity_results['polychromic_lora']['distinct_2']],
            "Distinct-2"
        )
        
        with open(output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
    else:
        logger.info("\n(Skipping statistical tests - need both baseline and polychromic LoRA models)")
    
    # ===== SUMMARY REPORT =====
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    summary = {
        'models_evaluated': list(all_model_replies.keys()),
        'diversity': diversity_results,
        'quality': quality_results,
        'passk': passk_results,
    }
    
    # Print summary for each model
    for model_name in all_model_replies.keys():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Diversity:")
        logger.info(f"    Self-BLEU (↓): {diversity_results[model_name]['self_bleu']:.4f}")
        logger.info(f"    Distinct-2 (↑): {diversity_results[model_name]['distinct_2']:.4f}")
        logger.info(f"    Semantic Div (↑): {diversity_results[model_name]['semantic_diversity']:.4f}")
        logger.info(f"  Quality:")
        logger.info(f"    ROUGE-L: {quality_results[model_name]['rouge']['rougeL']:.4f}")
        logger.info(f"  Pass@k:")
        for k in [1, 5, 10]:
            if str(k) in passk_results[model_name]:
                logger.info(f"    Pass@{k}: {passk_results[model_name][str(k)]:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*60)
    logger.info("\nGenerated files:")
    logger.info("  - diversity_metrics.json")
    logger.info("  - quality_metrics.json")
    logger.info("  - passk_results.json")
    if not args.skip_llm_judge and args.anthropic_key:
        logger.info("  - llm_judge_results.json")
    if args.baseline_lora and args.polychromic_lora:
        logger.info("  - statistical_tests.json")
    logger.info("  - summary.json")
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

