"""
Base LoRA Trainer for standard supervised fine-tuning.

This serves as the baseline for comparison with polychromic training.
Uses standard cross-entropy loss without diversity objectives.
"""

import torch
import logging
from typing import Dict, Optional
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

logger = logging.getLogger(__name__)


class BaseLoRATrainer(Trainer):
    """
    Standard LoRA trainer with enhanced logging for research.
    
    Differences from vanilla Trainer:
    - Detailed W&B logging
    - Gradient statistics
    - Sample generation during evaluation
    - Multiple random seed support
    """
    
    def __init__(
        self,
        *args,
        log_samples: bool = True,
        num_eval_samples: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.log_samples = log_samples
        self.num_eval_samples = num_eval_samples
        
        logger.info("Initialized BaseLoRATrainer")
        logger.info(f"  Log samples: {log_samples}")
        logger.info(f"  Num eval samples: {num_eval_samples}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Standard cross-entropy loss with enhanced logging.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Log to W&B
        if self.state.global_step % self.args.logging_steps == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/epoch': self.state.epoch,
            }, step=self.state.global_step)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval"
    ):
        """
        Override evaluation to include sample generation.
        """
        # Standard evaluation
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Generate samples during evaluation
        if self.log_samples and self.state.global_step > 0:
            self._log_sample_generations(dataloader, metric_key_prefix)
        
        # Log evaluation metrics to W&B
        wandb.log({
            f'{metric_key_prefix}/loss': output.metrics.get(f'{metric_key_prefix}_loss', 0),
        }, step=self.state.global_step)
        
        return output
    
    def _log_sample_generations(self, dataloader, metric_key_prefix="eval"):
        """
        Generate and log sample replies during evaluation.
        """
        self.model.eval()
        samples = []
        
        # Get a few samples from dataloader
        for i, batch in enumerate(dataloader):
            if i >= self.num_eval_samples:
                break
            
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            # Generate reply for first item in batch
            input_ids = batch['input_ids'][0:1]
            
            # Find prompt length (where labels start being valid)
            labels = batch.get('labels', None)
            if labels is not None:
                # Labels are -100 for prompt tokens
                valid_label_mask = labels[0] != -100
                if valid_label_mask.any():
                    prompt_length = valid_label_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    prompt_length = len(input_ids[0]) // 2
            else:
                prompt_length = len(input_ids[0]) // 2
            
            prompt_ids = input_ids[:, :prompt_length]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            generated_text = self.tokenizer.decode(
                outputs[0][prompt_length:], 
                skip_special_tokens=True
            )
            
            # Ground truth if available
            if labels is not None:
                ground_truth = self.tokenizer.decode(
                    labels[0][labels[0] != -100],
                    skip_special_tokens=True
                )
            else:
                ground_truth = "N/A"
            
            samples.append({
                'prompt': prompt_text,
                'generated': generated_text,
                'ground_truth': ground_truth
            })
        
        # Log to W&B as table
        if samples:
            table = wandb.Table(
                columns=['Prompt', 'Generated', 'Ground Truth'],
                data=[[s['prompt'][:200], s['generated'], s['ground_truth']] for s in samples]
            )
            wandb.log({f'{metric_key_prefix}/sample_generations': table}, step=self.state.global_step)
        
        self.model.train()


def setup_lora_model(model, lora_config_dict: Dict):
    """
    Setup LoRA adapters on a model.
    
    Args:
        model: Base model (potentially quantized)
        lora_config_dict: LoRA configuration parameters
        
    Returns:
        model: Model with LoRA adapters attached
    """
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_config_dict.get('rank', 16),
        lora_alpha=lora_config_dict.get('alpha', 16),
        lora_dropout=lora_config_dict.get('dropout', 0.1),
        target_modules=lora_config_dict.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config_dict.get('bias', 'none'),
        task_type=lora_config_dict.get('task_type', 'CAUSAL_LM'),
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params
    
    logger.info(f"LoRA Model Setup Complete:")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"  All params: {all_params:,}")
    logger.info(f"  LoRA rank: {lora_config.r}")
    logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    return model

