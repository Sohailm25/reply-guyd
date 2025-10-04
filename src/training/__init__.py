"""
Training modules for polychromic LoRA fine-tuning.

This package contains:
- BaseTrainer: Standard LoRA fine-tuning
- PolychromicTrainer: Diversity-aware LoRA fine-tuning
- GRPOTrainer: Group Relative Policy Optimization (reinforcement learning)
- DataModule: Data loading and preprocessing
- HeuristicRewardFunction: Reward function for GRPO
"""

from .base_trainer import BaseLoRATrainer
from .polychromic_trainer import PolychromicTrainer, PolychromicConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig
from .data_module import TwitterReplyDataModule
from .heuristic_reward import HeuristicRewardFunction

__all__ = [
    'BaseLoRATrainer',
    'PolychromicTrainer',
    'PolychromicConfig',
    'GRPOTrainer',
    'GRPOConfig',
    'HeuristicRewardFunction',
    'TwitterReplyDataModule',
]

