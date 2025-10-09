"""
Reward Functions for RL Training

This module contains all reward functions used for GRPO and other RL-based
training methods. Separating rewards from training code allows for:
- Easy composition of multiple rewards
- Reuse across different environments
- Clear testing and validation
"""

from .heuristic import HeuristicRewardFunction as HeuristicReward
from .composite import CompositeReward

__all__ = [
    'HeuristicReward',
    'CompositeReward',
]

