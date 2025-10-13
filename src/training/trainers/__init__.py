"""
Trainer Modules

Organized collection of all trainers:
- BaseLoRATrainer: Standard supervised fine-tuning with LoRA
- PolychromicTrainer: Diversity-aware supervised fine-tuning
- GRPOTrainer: Group Relative Policy Optimization (RL)
"""

from .base import BaseLoRATrainer, setup_lora_model
from .polychromic import PolychromicTrainer, PolychromicConfig
from .grpo import GRPOTrainer, GRPOConfig

__all__ = [
    'BaseLoRATrainer',
    'setup_lora_model',
    'PolychromicTrainer',
    'PolychromicConfig',
    'GRPOTrainer',
    'GRPOConfig',
]

