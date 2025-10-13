#!/usr/bin/env python3
"""
Save Model Generations to Disk

This script ONLY generates and saves model outputs to disk.
Use evaluate_from_saved.py to analyze the saved generations.

This separation allows:
1. Generation crashes → Just re-run generation
2. Evaluation crashes → Just re-run evaluation on saved generations  
3. Try different evaluation metrics without re-generating

Usage:
    python scripts/evaluation/save_generations.py \
        --baseline-lora output/experiments/baseline/seed_42 \
        --polychromic-lora output/experiments/polychromic_0.3/seed_42 \
        --test-data data/processed/test_data.jsonl \
        --output output/generations/ \
        --n-generations 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model_path: str = "./"):
    """Load a trained model with LoRA adapters."""
    logger.info(f"Loading LoRA model from: {model_path}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA detected - using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        logger.warning("No CUDA - loading full precision model on CPU")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_and_save(
    model,
    tokenizer,
    prompts: List[str],
    output_file: str,
    n_per_prompt: int = 10,
    max_new_tokens: int = 100
):
    """Generate replies and save to JSON file."""
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        replies = []
        
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
        
        # Generate n replies
        for _ in range(n_per_prompt):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
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
        
        results.append({
            'prompt': prompt,
            'replies': replies
        })
        
        # Save incrementally every 50 examples
        if (i + 1) % 50 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  Saved checkpoint at {i+1}/{len(prompts)}")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Saved {len(results)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-lora', type=str)
    parser.add_argument('--polychromic-lora', type=str)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n-generations', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--base-model', type=str, default="./")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    if args.max_examples:
        test_data = test_data[:args.max_examples]
    
    prompts = [item['tweet'] for item in test_data]
    logger.info(f"Loaded {len(prompts)} test examples")
    
    # Save prompts for reference
    with open(output_dir / 'prompts.json', 'w') as f:
        json.dump({'prompts': prompts}, f, indent=2)
    
    # Generate for baseline
    if args.baseline_lora:
        logger.info("\n" + "="*60)
        logger.info("BASELINE LORA")
        logger.info("="*60)
        
        model, tokenizer = load_model(args.baseline_lora, args.base_model)
        generate_and_save(
            model, 
            tokenizer, 
            prompts,
            output_dir / 'baseline_lora_generations.json',
            n_per_prompt=args.n_generations
        )
        
        # Free memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generate for polychromic
    if args.polychromic_lora:
        logger.info("\n" + "="*60)
        logger.info("POLYCHROMIC LORA")
        logger.info("="*60)
        
        model, tokenizer = load_model(args.polychromic_lora, args.base_model)
        generate_and_save(
            model,
            tokenizer,
            prompts,
            output_dir / 'polychromic_lora_generations.json',
            n_per_prompt=args.n_generations
        )
        
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info("\n" + "="*60)
    logger.info("✓ ALL GENERATIONS SAVED!")
    logger.info("="*60)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nNext step:")
    logger.info(f"  python scripts/evaluation/evaluate_from_saved.py --generations {output_dir}")


if __name__ == '__main__':
    main()


