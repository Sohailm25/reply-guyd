"""
Twitter Reply Environment

Encapsulates:
- Reward computation (heuristic + learned)
- Generation with diversity
- Evaluation metrics
- Prompt formatting

This environment provides a clean interface for Twitter reply generation,
separating concerns from training code.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import torch
from .base import BaseEnvironment
import logging

logger = logging.getLogger(__name__)


class TwitterReplyEnvironment(BaseEnvironment):
    """
    Environment for Twitter reply generation and evaluation.
    
    Handles:
    - Prompt formatting for Twitter replies
    - Diverse candidate generation
    - Reward computation
    - Batch evaluation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        reward_fn: Optional[Callable[[str, str], float]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Twitter Reply Environment.
        
        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            reward_fn: Reward function (defaults to heuristic if None)
            device: Device to run on (None = auto-detect)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.current_prompt = None
        self.current_tweet = None
        
        # Setup device
        if device is None:
            self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
        else:
            self.device = device
        
        # Setup reward function
        if reward_fn is None:
            # Import here to avoid circular dependency
            from src.rewards import HeuristicReward
            self.reward_fn = HeuristicReward()
        else:
            self.reward_fn = reward_fn
        
        logger.info(f"TwitterReplyEnvironment initialized on {self.device}")
    
    def reset(self, tweet: str) -> Dict[str, Any]:
        """
        Reset environment with new tweet.
        
        Args:
            tweet: Tweet to generate replies for
            
        Returns:
            Dict with tweet and formatted prompt
        """
        self.current_tweet = tweet
        self.current_prompt = self._format_prompt(tweet)
        
        return {
            "tweet": tweet,
            "prompt": self.current_prompt
        }
    
    def step(self, reply: str) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward for reply.
        
        Args:
            reply: Generated reply
            
        Returns:
            (reward, info_dict)
        """
        if self.current_tweet is None:
            raise ValueError("Must call reset() before step()")
        
        reward = self.reward_fn(self.current_tweet, reply)
        
        info = {
            "tweet": self.current_tweet,
            "reply": reply,
            "reward": reward,
            "reply_length": len(reply)
        }
        
        return reward, info
    
    def generate_candidates(
        self,
        n: int,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_new_tokens: int = 100,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate n diverse reply candidates.
        
        Args:
            n: Number of candidates to generate
            temperature: Sampling temperature for diversity
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (True) or greedy (False)
            
        Returns:
            List of n generated replies
        """
        if self.current_prompt is None:
            raise ValueError("Must call reset() before generate_candidates()")
        
        # Tokenize prompt
        inputs = self.tokenizer(
            self.current_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)
        
        replies = []
        
        # Generate n diverse replies
        for _ in range(n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract reply (everything after prompt)
            reply = full_text[len(self.current_prompt):].strip()
            
            # Handle empty replies
            if not reply or len(reply) < 3:
                reply = "I appreciate your perspective on this."  # Fallback
            
            replies.append(reply)
        
        return replies
    
    def compute_reward(self, tweet: str, reply: str) -> float:
        """
        Compute reward for tweet/reply pair.
        
        Args:
            tweet: Original tweet
            reply: Generated reply
            
        Returns:
            Reward score (higher is better)
        """
        return self.reward_fn(tweet, reply)
    
    def compute_rewards_batch(self, tweets: List[str], replies: List[str]) -> List[float]:
        """
        Compute rewards for batch of tweet/reply pairs.
        
        More efficient than calling compute_reward repeatedly if reward
        function supports batch processing.
        
        Args:
            tweets: List of tweets
            replies: List of replies
            
        Returns:
            List of reward scores
        """
        if len(tweets) != len(replies):
            raise ValueError(f"Mismatch: {len(tweets)} tweets, {len(replies)} replies")
        
        # Check if reward function has batch method
        if hasattr(self.reward_fn, 'compute_rewards_batch'):
            return self.reward_fn.compute_rewards_batch(tweets, replies)
        else:
            # Fall back to individual computation
            return [self.reward_fn(t, r) for t, r in zip(tweets, replies)]
    
    def _format_prompt(self, tweet: str) -> str:
        """
        Format tweet into prompt for model.
        
        Args:
            tweet: Original tweet
            
        Returns:
            Formatted prompt string
        """
        return f"Generate an engaging Twitter reply to this tweet:\n\n{tweet}\n\nReply:"
    
    def evaluate_batch(
        self,
        tweets: List[str],
        replies: List[str],
        compute_diversity: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate batch of replies with multiple metrics.
        
        Args:
            tweets: List of tweets
            replies: List of replies
            compute_diversity: Whether to compute diversity metrics
            
        Returns:
            Dict with evaluation results
        """
        results = {}
        
        # Compute rewards
        rewards = self.compute_rewards_batch(tweets, replies)
        results['rewards'] = rewards
        results['avg_reward'] = sum(rewards) / len(rewards) if rewards else 0.0
        
        # Length statistics
        lengths = [len(r) for r in replies]
        results['avg_length'] = sum(lengths) / len(lengths) if lengths else 0.0
        results['min_length'] = min(lengths) if lengths else 0
        results['max_length'] = max(lengths) if lengths else 0
        
        # Diversity metrics (if requested)
        if compute_diversity:
            try:
                from src.evaluation.metrics import compute_all_diversity_metrics
                diversity = compute_all_diversity_metrics(replies)
                results['diversity'] = diversity
            except ImportError:
                logger.warning("Could not import diversity metrics")
        
        return results
    
    def set_reward_function(self, reward_fn: Callable[[str, str], float]):
        """
        Update the reward function.
        
        Useful for switching between heuristic and learned rewards,
        or combining multiple rewards.
        
        Args:
            reward_fn: New reward function
        """
        self.reward_fn = reward_fn
        logger.info("Reward function updated")

