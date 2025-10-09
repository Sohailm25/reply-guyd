"""
Base Environment for RL-style evaluation and training.

Inspired by Prime Intellect's verifiers but adapted for our use case.
Provides a clean abstraction for:
- Generation with diversity
- Reward computation
- Evaluation metrics
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch


class BaseEnvironment(ABC):
    """
    Base class for environments (e.g., TwitterReply, CodeGen, etc.)
    
    This abstraction allows easy extension to new domains while maintaining
    consistent interfaces for training and evaluation.
    """
    
    @abstractmethod
    def reset(self, prompt: str) -> Dict[str, Any]:
        """
        Reset environment with new prompt.
        
        Args:
            prompt: Input prompt (e.g., tweet to reply to)
            
        Returns:
            Dict with environment state
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> Tuple[float, Dict[str, Any]]:
        """
        Take action (generate text), return reward and info.
        
        Args:
            action: Generated response
            
        Returns:
            (reward, info_dict)
        """
        pass
    
    @abstractmethod
    def generate_candidates(self, n: int, **kwargs) -> List[str]:
        """
        Generate n diverse candidates.
        
        Args:
            n: Number of candidates to generate
            **kwargs: Generation parameters (temperature, top_p, etc.)
            
        Returns:
            List of n generated responses
        """
        pass
    
    @abstractmethod
    def compute_reward(self, prompt: str, response: str) -> float:
        """
        Compute reward for response.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Reward score (higher is better)
        """
        pass
    
    def compute_rewards_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Compute rewards for batch of responses.
        
        Default implementation calls compute_reward for each pair.
        Override for more efficient batch processing.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of reward scores
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Mismatch: {len(prompts)} prompts, {len(responses)} responses")
        
        return [self.compute_reward(p, r) for p, r in zip(prompts, responses)]

