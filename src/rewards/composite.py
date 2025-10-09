"""
Composite Reward Function

Combines multiple reward functions with configurable weights.
Useful for balancing different objectives (e.g., relevance + diversity + safety).
"""

from typing import Dict, Callable, List
import logging

logger = logging.getLogger(__name__)


class CompositeReward:
    """
    Combine multiple reward functions with weights.
    
    Example:
        >>> from src.rewards import HeuristicReward, CompositeReward
        >>> 
        >>> heuristic = HeuristicReward()
        >>> learned = LearnedReward(model_path="...")
        >>> 
        >>> composite = CompositeReward({
        ...     "heuristic": (heuristic, 0.4),
        ...     "learned": (learned, 0.6)
        ... })
        >>> 
        >>> score = composite("tweet text", "reply text")
    """
    
    def __init__(self, rewards: Dict[str, tuple[Callable, float]]):
        """
        Initialize composite reward.
        
        Args:
            rewards: Dict mapping {name: (reward_fn, weight)}
                    Weights should sum to 1.0 (but not enforced)
        """
        self.rewards = rewards
        
        # Validate weights
        total_weight = sum(weight for _, weight in rewards.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Reward weights sum to {total_weight:.3f}, not 1.0. "
                "Consider normalizing for interpretability."
            )
        
        logger.info(f"CompositeReward initialized with {len(rewards)} components:")
        for name, (_, weight) in rewards.items():
            logger.info(f"  - {name}: {weight:.3f}")
    
    def __call__(self, prompt: str, response: str) -> float:
        """
        Compute composite reward.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Weighted combination of all reward scores
        """
        return self.compute_reward(prompt, response)
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """
        Compute composite reward.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Weighted combination of all reward scores
        """
        total = 0.0
        
        for name, (reward_fn, weight) in self.rewards.items():
            try:
                score = reward_fn(prompt, response)
                total += weight * score
            except Exception as e:
                logger.error(f"Error computing reward '{name}': {e}")
                # Continue with other rewards
        
        return total
    
    def compute_rewards_batch(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> List[float]:
        """
        Compute composite rewards for batch.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of composite reward scores
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Mismatch: {len(prompts)} prompts, {len(responses)} responses")
        
        # Try batch computation for each reward function
        component_scores = {}
        
        for name, (reward_fn, weight) in self.rewards.items():
            try:
                # Check if reward function supports batch
                if hasattr(reward_fn, 'compute_rewards_batch'):
                    scores = reward_fn.compute_rewards_batch(prompts, responses)
                else:
                    # Fall back to individual computation
                    scores = [reward_fn(p, r) for p, r in zip(prompts, responses)]
                
                component_scores[name] = (scores, weight)
                
            except Exception as e:
                logger.error(f"Error computing batch reward '{name}': {e}")
                # Use zeros for this component
                component_scores[name] = ([0.0] * len(prompts), weight)
        
        # Combine weighted scores
        total_scores = []
        for i in range(len(prompts)):
            total = sum(
                scores[i] * weight
                for scores, weight in component_scores.values()
            )
            total_scores.append(total)
        
        return total_scores
    
    def get_breakdown(self, prompt: str, response: str) -> Dict[str, float]:
        """
        Get per-component scores (for debugging/analysis).
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            Dict mapping component name to score
        """
        breakdown = {}
        
        for name, (reward_fn, weight) in self.rewards.items():
            try:
                score = reward_fn(prompt, response)
                breakdown[name] = {
                    'raw_score': score,
                    'weight': weight,
                    'weighted_score': score * weight
                }
            except Exception as e:
                logger.error(f"Error computing breakdown for '{name}': {e}")
                breakdown[name] = {
                    'raw_score': 0.0,
                    'weight': weight,
                    'weighted_score': 0.0,
                    'error': str(e)
                }
        
        # Add total
        breakdown['total'] = sum(
            comp['weighted_score'] 
            for comp in breakdown.values()
            if isinstance(comp, dict) and 'weighted_score' in comp
        )
        
        return breakdown
    
    def add_reward(self, name: str, reward_fn: Callable, weight: float):
        """
        Add a new reward component.
        
        Args:
            name: Name for this reward
            reward_fn: Reward function
            weight: Weight for this reward
        """
        if name in self.rewards:
            logger.warning(f"Replacing existing reward '{name}'")
        
        self.rewards[name] = (reward_fn, weight)
        logger.info(f"Added reward '{name}' with weight {weight:.3f}")
    
    def remove_reward(self, name: str):
        """
        Remove a reward component.
        
        Args:
            name: Name of reward to remove
        """
        if name in self.rewards:
            del self.rewards[name]
            logger.info(f"Removed reward '{name}'")
        else:
            logger.warning(f"Reward '{name}' not found")
    
    def update_weight(self, name: str, new_weight: float):
        """
        Update weight for a reward component.
        
        Args:
            name: Name of reward
            new_weight: New weight value
        """
        if name not in self.rewards:
            raise ValueError(f"Reward '{name}' not found")
        
        reward_fn, old_weight = self.rewards[name]
        self.rewards[name] = (reward_fn, new_weight)
        
        logger.info(f"Updated '{name}' weight: {old_weight:.3f} → {new_weight:.3f}")


# Convenience function for creating balanced composites
def create_balanced_composite(reward_fns: List[Callable], names: List[str] = None) -> CompositeReward:
    """
    Create composite reward with equal weights.
    
    Args:
        reward_fns: List of reward functions
        names: Optional list of names (defaults to reward_1, reward_2, ...)
        
    Returns:
        CompositeReward with equal weights
    """
    n = len(reward_fns)
    weight = 1.0 / n
    
    if names is None:
        names = [f"reward_{i+1}" for i in range(n)]
    
    if len(names) != n:
        raise ValueError(f"Number of names ({len(names)}) must match number of functions ({n})")
    
    rewards = {
        name: (fn, weight)
        for name, fn in zip(names, reward_fns)
    }
    
    return CompositeReward(rewards)

