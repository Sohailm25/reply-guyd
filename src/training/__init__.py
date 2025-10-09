"""
Training modules for polychromic LoRA fine-tuning.

This package contains:
- BaseTrainer: Standard LoRA fine-tuning
- PolychromicTrainer: Diversity-aware LoRA fine-tuning
- GRPOTrainer: Group Relative Policy Optimization (reinforcement learning)
- DataModule: Data loading and preprocessing

Note: Trainers are being reorganized into src.training.trainers submodule.
      Rewards have moved to src.rewards module.
      
Recommended imports for new code:
    from src.training.trainers import BaseLoRATrainer, PolychromicTrainer, GRPOTrainer
    from src.rewards import HeuristicReward, CompositeReward
    from src.environments import TwitterReplyEnvironment
"""

# New location (recommended)
from .trainers import (
    BaseLoRATrainer,
    PolychromicTrainer,
    PolychromicConfig,
    GRPOTrainer,
    GRPOConfig,
    setup_lora_model
)

# Legacy imports (for backward compatibility with existing scripts)
from .base_trainer import BaseLoRATrainer as _BaseLoRATrainer
from .polychromic_trainer import PolychromicTrainer as _PolychromicTrainer, PolychromicConfig as _PolychromicConfig
from .grpo_trainer import GRPOTrainer as _GRPOTrainer, GRPOConfig as _GRPOConfig

from .data_module import TwitterReplyDataModule

# Backward compatibility: Re-export HeuristicRewardFunction from new location
from src.rewards import HeuristicReward as HeuristicRewardFunction

__all__ = [
    'BaseLoRATrainer',
    'setup_lora_model',
    'PolychromicTrainer',
    'PolychromicConfig',
    'GRPOTrainer',
    'GRPOConfig',
    'HeuristicRewardFunction',  # Deprecated: use src.rewards.HeuristicReward
    'TwitterReplyDataModule',
]

