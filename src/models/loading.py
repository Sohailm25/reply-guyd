"""
Centralized Model Loading Utilities

Provides consistent model loading across training and evaluation.
"""

import torch
import logging
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def load_base_model(
    model_path: str,
    quantization_config: Optional[Dict[str, Any]] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = True
):
    """
    Load base model with optional quantization.
    
    Args:
        model_path: Path to model (local or HuggingFace)
        quantization_config: Quantization settings (None = no quantization)
        device_map: Device placement strategy
        torch_dtype: Torch data type (None = auto)
        use_flash_attention: Whether to use flash attention 2
        
    Returns:
        (model, tokenizer) tuple
    """
    logger.info("="*60)
    logger.info("Loading Base Model")
    logger.info("="*60)
    logger.info(f"  Model: {model_path}")
    
    # Setup quantization if requested
    bnb_config = None
    if quantization_config is not None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True),
        )
        logger.info(f"  Quantization: {quantization_config.get('bnb_4bit_quant_type', 'nf4')}")
    
    # Determine torch dtype
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    
    logger.info(f"  Dtype: {torch_dtype}")
    logger.info(f"  Device map: {device_map}")
    
    # Load model
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": False
    }
    
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    
    if use_flash_attention and torch.cuda.is_available():
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("  Using Flash Attention 2")
        except:
            logger.warning("  Flash Attention 2 not available, using default")
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("  Set pad_token = eos_token")
    
    logger.info(f"  Model loaded: {model.config.model_type}")
    logger.info(f"  Parameters: {model.num_parameters():,}")
    logger.info("="*60 + "\n")
    
    return model, tokenizer


def load_model_with_lora(
    base_model_path: str,
    lora_path: str,
    quantization_config: Optional[Dict[str, Any]] = None,
    device_map: str = "auto",
    merge_weights: bool = False
):
    """
    Load model with LoRA adapters.
    
    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA checkpoint
        quantization_config: Quantization settings
        device_map: Device placement
        merge_weights: Whether to merge LoRA weights into base model
        
    Returns:
        (model, tokenizer) tuple
    """
    logger.info(f"Loading model with LoRA adapters from: {lora_path}")
    
    # Load base model
    base_model, tokenizer = load_base_model(
        base_model_path,
        quantization_config=quantization_config,
        device_map=device_map
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=False
    )
    
    if merge_weights:
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
    
    model.eval()
    
    logger.info("✅ Model with LoRA loaded successfully")
    
    return model, tokenizer


def setup_lora_config(lora_config: Dict[str, Any]) -> LoraConfig:
    """
    Create LoRA configuration from dict.
    
    Args:
        lora_config: Dict with LoRA parameters
        
    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        bias=lora_config.get('bias', "none"),
        task_type=lora_config.get('task_type', "CAUSAL_LM"),
    )


def apply_lora_to_model(model, lora_config: Dict[str, Any]):
    """
    Apply LoRA adapters to model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration dict
        
    Returns:
        Model with LoRA adapters
    """
    lora_cfg = setup_lora_config(lora_config)
    model = get_peft_model(model, lora_cfg)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info("="*60)
    logger.info("LoRA Applied")
    logger.info("="*60)
    logger.info(f"  Rank: {lora_config.get('r', 16)}")
    logger.info(f"  Alpha: {lora_config.get('lora_alpha', 32)}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info("="*60 + "\n")
    
    return model

