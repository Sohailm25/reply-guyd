#!/usr/bin/env python3
"""
Main Training Script for RunPod

Supports baseline, polychromic, and GRPO training.
Can be run with different configurations for ablation studies.

Usage:
    python scripts/training/train_model.py --config config/experiments/baseline.yaml
    python scripts/training/train_model.py --config config/experiments/polychromic_0.3.yaml
    python scripts/training/train_model.py --config config/experiments/grpo_from_baseline.yaml
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
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training import (
    BaseLoRATrainer,
    PolychromicTrainer,
    PolychromicConfig,
    GRPOTrainer,
    GRPOConfig,
    HeuristicRewardFunction,
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


def load_model_and_tokenizer(config: dict):
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
    unsloth_config = config.get('unsloth', {})
    use_unsloth = unsloth_config.get('enabled', False)
    max_seq_length = config.get('data', {}).get('max_length', 2048)
    
    if use_unsloth:
        logger.info("Using Unsloth FastLanguageModel backend")
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError(
                "Unsloth integration requested but the 'unsloth' package is not installed."
            ) from exc
        except NotImplementedError as exc:
            raise RuntimeError(
                "Unsloth detected no available GPU/accelerator. "
                "Run on a CUDA-capable machine or disable `unsloth.enabled`."
            ) from exc
        
        dtype_value = unsloth_config.get('dtype')
        if isinstance(dtype_value, str):
            if not hasattr(torch, dtype_value):
                raise ValueError(f"Unsupported dtype '{dtype_value}' for Unsloth integration.")
            dtype_value = getattr(torch, dtype_value)
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Quantization: {'4bit' if quant_config['load_in_4bit'] else 'full-precision'}")
        
        unsloth_kwargs = {
            "max_seq_length": max_seq_length,
            "dtype": dtype_value,
            "load_in_4bit": quant_config['load_in_4bit'],
            "device_map": unsloth_config.get('device_map', 'auto'),
            "rope_scaling": unsloth_config.get('rope_scaling', config['model'].get('rope_scaling')),
            "tokenizer_name": unsloth_config.get('tokenizer_name'),
            "trust_remote_code": config['model'].get('trust_remote_code', False),
        }
        # Remove None values to avoid overriding defaults
        unsloth_kwargs = {k: v for k, v in unsloth_kwargs.items() if v is not None}
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            **unsloth_kwargs,
        )
        backend = "unsloth"
    else:
        # Setup quantization for standard HF path
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
        )
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Quantization: {quant_config['bnb_4bit_quant_type']}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if quant_config['bnb_4bit_compute_dtype'] == 'bfloat16' else torch.float16,
            trust_remote_code=config['model'].get('trust_remote_code', False)
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        backend = "hf"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model size: {model.num_parameters():,} parameters")
    logger.info("="*60 + "\n")
    
    return model, tokenizer, backend


def apply_lora_adapters(model, config: dict, backend: str):
    """Attach LoRA adapters using either standard PEFT or Unsloth helper."""
    logger.info("Applying LoRA adapters...")
    lora_cfg = config['lora']
    
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        grad_ckpt_enabled = config['training'].get('gradient_checkpointing', True)
        return FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get('rank', 16),
            target_modules=lora_cfg.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.0),
            bias=lora_cfg.get('bias', 'none'),
            use_gradient_checkpointing="unsloth" if grad_ckpt_enabled else False,
        )
    
    return setup_lora_model(model, lora_cfg)


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
        eval_strategy=train_config.get('eval_strategy', 'steps'),
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
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to Phase 1 checkpoint for two-phase training (overrides config)'
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
    model, tokenizer, model_backend = load_model_and_tokenizer(config)
    
    # Apply LoRA
    model = apply_lora_adapters(model, config, model_backend)
    
    # Load checkpoint if specified (for two-phase training)
    # Priority: command-line arg > config file
    checkpoint_path = args.checkpoint or config.get('training', {}).get('checkpoint_path', None)
    
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            logger.info("\n" + "="*60)
            logger.info("LOADING PHASE 1 CHECKPOINT (Two-Phase Training)")
            logger.info("="*60)
            logger.info(f"  Checkpoint: {checkpoint_path}")
            logger.info("  Purpose: Warm-start for Phase 2 (GRPO/continued training)")
            logger.info("="*60)
            
            try:
                # Load LoRA adapters from Phase 1 checkpoint
                # This replaces the fresh LoRA adapters with trained ones
                model = PeftModel.from_pretrained(
                    model,
                    checkpoint_path,
                    is_trainable=True  # Keep trainable for Phase 2
                )
                
                logger.info("✅ Checkpoint loaded successfully!")
                logger.info("  Model now initialized with Phase 1 weights")
                logger.info("  Ready to continue training with Phase 2 data")
                logger.info("="*60 + "\n")
                
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint: {e}")
                logger.error("Falling back to fresh LoRA initialization")
                logger.error("Training will start from base model\n")
        else:
            logger.warning("\n" + "="*60)
            logger.warning("⚠️  CHECKPOINT PATH SPECIFIED BUT NOT FOUND")
            logger.warning("="*60)
            logger.warning(f"  Path: {checkpoint_path}")
            logger.warning("  Starting from base model + fresh LoRA instead")
            logger.warning("  This is NOT two-phase training!")
            logger.warning("="*60 + "\n")
    
    # Setup data
    logger.info("Loading data...")
    data_module = TwitterReplyDataModule(
        data_path=args.data,
        tokenizer=tokenizer,
        train_split=0.9,
        val_split=0.1,
        test_split=0.0,
        max_length=config['data']['max_length'],
        enable_thinking=config['data'].get('enable_thinking', False),
        random_seed=seed
    )
    
    train_dataset = data_module.get_train_dataset()
    eval_dataset = data_module.get_val_dataset()
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Choose trainer based on experiment type or training config
    experiment_type = config['experiment'].get('type', 'baseline')
    use_grpo = config.get('training', {}).get('use_grpo', False) or experiment_type == 'grpo'
    
    if use_grpo:
        logger.info("="*60)
        logger.info("Using GRPO Trainer (Reinforcement Learning)")
        logger.info("="*60)
        
        # Initialize reward function
        reward_type = config.get('training', {}).get('reward_type', 'heuristic')
        
        if reward_type == 'heuristic':
            logger.info("  Reward model: Heuristic ensemble")
            reward_function = HeuristicRewardFunction()
        else:
            # TODO: Support learned reward model
            logger.warning(f"  Reward type '{reward_type}' not yet supported, using heuristic")
            reward_function = HeuristicRewardFunction()
        
        # GRPO configuration
        grpo_config_dict = config.get('grpo', {})
        grpo_config = GRPOConfig(**grpo_config_dict)
        
        # Create GRPO trainer
        # Note: reference_model=None means it will create a frozen copy of the loaded model
        # If checkpoint was loaded, reference = frozen copy of Phase 1 checkpoint ✅
        trainer = GRPOTrainer(
            reward_function=reward_function,
            reference_model=None,  # Will create frozen copy of current model state
            grpo_config=grpo_config,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
    elif experiment_type == 'polychromic':
        logger.info("="*60)
        logger.info("Using Polychromic Trainer (Diversity-Aware)")
        logger.info("="*60)
        
        polychromic_params = {k: v for k, v in config['polychromic'].items() if k != 'enabled'}
        polychromic_config = PolychromicConfig(**polychromic_params)
        
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
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Phase 1 Checkpoint: {checkpoint_path} ✅")
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
