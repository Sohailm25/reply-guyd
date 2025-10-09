#!/usr/bin/env python3
"""
Clean Training Script Using Refactored Architecture

This demonstrates the new, cleaner way to use the refactored codebase:
- Environment abstraction for reward computation
- Centralized model loading
- Organized trainer imports

The original train_model.py still works (backward compatibility maintained),
but this shows the recommended approach for new code.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
import wandb
from transformers import TrainingArguments, DataCollatorForSeq2Seq, set_seed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# New organized imports
from src.models import load_base_model, apply_lora_to_model
from src.environments import TwitterReplyEnvironment
from src.rewards import HeuristicReward, CompositeReward
from src.training.trainers import BaseLoRATrainer, PolychromicTrainer, GRPOTrainer, PolychromicConfig, GRPOConfig
from src.training import TwitterReplyDataModule

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
        num_train_epochs=train_config['num_epochs'],
        learning_rate=train_config['learning_rate'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
        eval_strategy="steps",
        eval_steps=train_config.get('eval_steps', 50),
        logging_steps=train_config.get('logging_steps', 10),
        save_steps=train_config.get('save_steps', 50),
        save_total_limit=train_config.get('save_total_limit', 3),
        bf16=train_config.get('bf16', True),
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        report_to=["wandb"] if wandb_config.get('enabled', True) else [],
        seed=config['experiment'].get('random_seed', 42),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-8B with clean architecture")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for warm-start')
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set random seed
    seed = config['experiment'].get('random_seed', 42)
    set_seed(seed)
    
    # 1. Load model using centralized loader
    logger.info("\n📦 Loading model...")
    model, tokenizer = load_base_model(
        model_path=config['model']['path'],
        quantization_config=config['model']['quantization']
    )
    
    # 2. Apply LoRA
    logger.info("🔧 Applying LoRA adapters...")
    from src.models.loading import apply_lora_to_model
    model = apply_lora_to_model(model, config['lora'])
    
    # 3. Load checkpoint if specified (for two-phase training)
    checkpoint_path = args.checkpoint or config.get('training', {}).get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"\n♻️  Loading checkpoint from: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        logger.info("✅ Checkpoint loaded successfully")
    
    # 4. Setup data
    logger.info("\n📊 Loading data...")
    data_module = TwitterReplyDataModule(
        data_path=args.data,
        tokenizer=tokenizer,
        train_split=0.9,
        val_split=0.1,
        max_length=config['data']['max_length'],
        random_seed=seed
    )
    
    train_dataset = data_module.get_train_dataset()
    eval_dataset = data_module.get_val_dataset()
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 5. Setup training arguments
    training_args = setup_training_args(config)
    
    # 6. Choose trainer based on experiment type
    experiment_type = config['experiment'].get('type', 'baseline')
    use_grpo = config.get('training', {}).get('use_grpo', False) or experiment_type == 'grpo'
    
    if use_grpo:
        logger.info("\n🤖 Setting up GRPO Trainer")
        
        # Create reward function
        reward_fn = HeuristicReward()
        
        # Create environment (clean abstraction!)
        env = TwitterReplyEnvironment(
            model=model,
            tokenizer=tokenizer,
            reward_fn=reward_fn
        )
        
        # GRPO config
        grpo_config = GRPOConfig(**config.get('grpo', {}))
        
        # Create trainer with environment
        trainer = GRPOTrainer(
            environment=env,  # Clean interface!
            grpo_config=grpo_config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
    elif experiment_type == 'polychromic':
        logger.info("\n🌈 Setting up Polychromic Trainer")
        
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
        logger.info("\n📝 Setting up Base Trainer")
        
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
    
    # 7. Train
    logger.info("\n🚀 Starting Training")
    logger.info("="*60)
    logger.info(f"  Experiment: {config['experiment']['name']}")
    logger.info(f"  Type: {experiment_type}")
    logger.info(f"  Output: {training_args.output_dir}")
    logger.info("="*60 + "\n")
    
    try:
        trainer.train()
        
        # Save final model
        logger.info("\n💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save config
        config_save_path = os.path.join(training_args.output_dir, 'experiment_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"✅ Training complete! Model saved to: {training_args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if config.get('wandb', {}).get('enabled', True):
            wandb.finish()


if __name__ == "__main__":
    main()

