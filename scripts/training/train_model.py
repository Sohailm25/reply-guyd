#!/usr/bin/env python3
"""
Main Training Script for RunPod

Supports both baseline and polychromic training.
Can be run with different configurations for ablation studies.

Usage:
    python scripts/training/train_model.py --config config/experiments/baseline.yaml
    python scripts/training/train_model.py --config config/experiments/polychromic_0.3.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training import (
    BaseLoRATrainer,
    PolychromicTrainer,
    PolychromicConfig,
    TwitterReplyDataModule
)
from src.training.base_trainer import setup_lora_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """
    Setup model and tokenizer with quantization.
    
    Returns:
        (model, tokenizer)
    """
    logger.info("="*60)
    logger.info("Loading Model and Tokenizer")
    logger.info("="*60)
    
    model_path = config['model']['path']
    quant_config = config['model']['quantization']
    
    # Setup quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
    )
    
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Quantization: {quant_config['bnb_4bit_quant_type']}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quant_config['bnb_4bit_compute_dtype'] == 'bfloat16' else torch.float16,
        trust_remote_code=False
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model size: {model.num_parameters():,} parameters")
    logger.info("="*60 + "\n")
    
    return model, tokenizer


def setup_training_args(config: dict) -> TrainingArguments:
    """Setup HuggingFace training arguments."""
    train_config = config['training']
    output_config = config['output']
    wandb_config = config.get('wandb', {})
    
    # Setup W&B
    if wandb_config.get('enabled', True):
        wandb.init(
            project=wandb_config.get('project', 'qwen3-training'),
            name=wandb_config.get('name', 'experiment'),
            tags=wandb_config.get('tags', []),
            config=config
        )
    
    args = TrainingArguments(
        output_dir=output_config['output_dir'],
        logging_dir=output_config.get('logging_dir', f"{output_config['output_dir']}/logs"),
        
        # Training hyperparameters
        num_train_epochs=train_config['num_epochs'],
        learning_rate=train_config['learning_rate'],
        warmup_ratio=train_config.get('warmup_ratio', 0.03),
        weight_decay=train_config.get('weight_decay', 0.01),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        
        # Batch sizes
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
        
        # Optimizer
        optim=train_config.get('optim', 'adamw_torch'),
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'cosine'),
        
        # Precision
        bf16=train_config.get('bf16', True),
        fp16=train_config.get('fp16', False),
        
        # Gradient checkpointing
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        
        # Evaluation
        evaluation_strategy=train_config.get('eval_strategy', 'steps'),
        eval_steps=train_config.get('eval_steps', 50),
        logging_steps=train_config.get('logging_steps', 10),
        save_steps=train_config.get('save_steps', 50),
        save_total_limit=train_config.get('save_total_limit', 3),
        
        # Best model
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_config.get('greater_is_better', False),
        
        # Reporting
        report_to=["wandb"] if wandb_config.get('enabled', True) else [],
        run_name=wandb_config.get('name', 'experiment'),
        
        # Misc
        seed=config['experiment'].get('random_seed', 42),
        data_seed=config['experiment'].get('random_seed', 42),
    )
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-8B with LoRA")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data (JSONL file or directory)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if exists'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set random seed
    seed = config['experiment'].get('random_seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = setup_lora_model(model, config['lora'])
    
    # Setup data
    logger.info("Loading data...")
    data_module = TwitterReplyDataModule(
        data_path=args.data,
        tokenizer=tokenizer,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        max_length=config['data']['max_length'],
        enable_thinking=config['data'].get('enable_thinking', False),
        random_seed=seed
    )
    
    train_dataset = data_module.get_train_dataset()
    eval_dataset = data_module.get_val_dataset()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Choose trainer
    experiment_type = config['experiment'].get('type', 'baseline')
    
    if experiment_type == 'polychromic':
        logger.info("="*60)
        logger.info("Using Polychromic Trainer (Diversity-Aware)")
        logger.info("="*60)
        
        polychromic_config = PolychromicConfig(**config['polychromic'])
        
        trainer = PolychromicTrainer(
            polychromic_config=polychromic_config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        logger.info("="*60)
        logger.info("Using Base Trainer (Standard LoRA)")
        logger.info("="*60)
        
        trainer = BaseLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            log_samples=True,
            num_eval_samples=5
        )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Type: {experiment_type}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info("="*60 + "\n")
    
    try:
        if args.resume and os.path.exists(training_args.output_dir):
            logger.info("Resuming from checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        
        # Save final model
        logger.info("\nSaving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save config
        config_save_path = os.path.join(training_args.output_dir, 'experiment_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Model saved to: {training_args.output_dir}")
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Finish W&B
        if config.get('wandb', {}).get('enabled', True):
            wandb.finish()


if __name__ == "__main__":
    main()

