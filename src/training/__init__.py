"""
Training modules for polychromic LoRA fine-tuning.

This package contains:
- BaseTrainer: Standard LoRA fine-tuning
- PolychromicTrainer: Diversity-aware LoRA fine-tuning
- DataModule: Data loading and preprocessing
"""

from .base_trainer import BaseLoRATrainer
from .polychromic_trainer import PolychromicTrainer, PolychromicConfig
from .data_module import TwitterReplyDataModule

__all__ = [
    'BaseLoRATrainer',
    'PolychromicTrainer',
    'PolychromicConfig',
    'TwitterReplyDataModule',
]

