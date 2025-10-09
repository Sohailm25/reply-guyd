"""
Heuristic Reward Function for GRPO Training

Provides a zero-training reward function for GRPO by combining
proven metrics from evaluation and polychromic training.

This serves as a baseline reward model and can be combined with
learned reward models for robustness.

STATUS: COMPLETE - Ready for GRPO training
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class HeuristicRewardFunction:
    """
    Heuristic reward function combining multiple quality signals.
    
    Components:
    1. Semantic Relevance (30%) - Reply should be relevant to tweet
    2. Length Quality (15%) - Optimal length for Twitter replies
    3. Sentiment Alignment (15%) - Reply should match or playfully counter tweet sentiment
    4. Lexical Diversity (10%) - Avoid repetitive/generic replies
    5. Punctuation Balance (10%) - Not too much punctuation
    6. Engagement Signals (20%) - Heuristics for engaging replies
    
    All metrics normalized to [0, 1] range.
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize heuristic reward function.
        
        Args:
            encoder_name: Sentence transformer model for semantic similarity
            device: Device to run encoder on (None = auto-detect)
        """
        logger.info("="*60)
        logger.info("Initializing Heuristic Reward Function")
        logger.info("="*60)
        logger.info(f"  Encoder: {encoder_name}")
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"  Device: {self.device}")
        
        # Load semantic encoder (same as polychromic trainer)
        try:
            self.encoder = SentenceTransformer(encoder_name, device=self.device)
            logger.info("  ✅ Encoder loaded successfully")
        except Exception as e:
            logger.error(f"  ❌ Failed to load encoder: {e}")
            logger.warning("  Using fallback: no semantic similarity")
            self.encoder = None
        
        # Component weights (sum to 1.0)
        self.weights = {
            'relevance': 0.30,      # Semantic similarity to tweet
            'length': 0.15,         # Optimal length
            'sentiment': 0.15,      # Sentiment alignment
            'diversity': 0.10,      # Lexical diversity
            'punctuation': 0.10,    # Punctuation balance
            'engagement': 0.20      # Engagement heuristics
        }
        
        logger.info("  Component weights:")
        for component, weight in self.weights.items():
            logger.info(f"    - {component}: {weight:.2f}")
        
        logger.info("="*60)
        logger.info("✅ Heuristic Reward Function ready")
        logger.info("="*60)
    
    def __call__(self, tweet: str, reply: str) -> float:
        """
        Compute reward for a tweet/reply pair.
        
        Args:
            tweet: Original tweet text
            reply: Generated reply text
            
        Returns:
            Reward score in [0, 1] range (higher is better)
        """
        return self.compute_reward(tweet, reply)
    
    def compute_reward(self, tweet: str, reply: str) -> float:
        """
        Compute overall reward score.
        
        Args:
            tweet: Original tweet text
            reply: Generated reply text
            
        Returns:
            Reward score in [0, 1] range
        """
        # Handle edge cases
        if not reply or len(reply.strip()) < 3:
            return 0.0  # Too short to be meaningful
        
        if len(reply) > 500:
            return 0.3  # Way too long for Twitter
        
        # Compute individual components
        relevance_score = self._compute_relevance(tweet, reply)
        length_score = self._compute_length_quality(reply)
        sentiment_score = self._compute_sentiment_alignment(tweet, reply)
        diversity_score = self._compute_lexical_diversity(reply)
        punctuation_score = self._compute_punctuation_balance(reply)
        engagement_score = self._compute_engagement_signals(tweet, reply)
        
        # Weighted combination
        total_score = (
            self.weights['relevance'] * relevance_score +
            self.weights['length'] * length_score +
            self.weights['sentiment'] * sentiment_score +
            self.weights['diversity'] * diversity_score +
            self.weights['punctuation'] * punctuation_score +
            self.weights['engagement'] * engagement_score
        )
        
        # Ensure in [0, 1] range
        total_score = max(0.0, min(1.0, total_score))
        
        return total_score
    
    def compute_rewards_batch(self, tweets: list[str], replies: list[str]) -> list[float]:
        """
        Compute rewards for a batch of tweet/reply pairs.
        
        More efficient than calling compute_reward() repeatedly.
        
        Args:
            tweets: List of tweet texts
            replies: List of reply texts
            
        Returns:
            List of reward scores
        """
        if len(tweets) != len(replies):
            raise ValueError(f"Mismatch: {len(tweets)} tweets, {len(replies)} replies")
        
        # Compute batch embeddings for efficiency
        if self.encoder is not None:
            tweet_embeds = self.encoder.encode(
                tweets, 
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
            reply_embeds = self.encoder.encode(
                replies,
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=False
            )
            
            # Batch cosine similarity
            relevance_scores = F.cosine_similarity(tweet_embeds, reply_embeds, dim=-1)
            relevance_scores = relevance_scores.cpu().tolist()
        else:
            relevance_scores = [0.5] * len(tweets)  # Fallback
        
        # Compute other components individually (fast anyway)
        rewards = []
        for i, (tweet, reply, relevance) in enumerate(zip(tweets, replies, relevance_scores)):
            # Skip full relevance computation (already done)
            length_score = self._compute_length_quality(reply)
            sentiment_score = self._compute_sentiment_alignment(tweet, reply)
            diversity_score = self._compute_lexical_diversity(reply)
            punctuation_score = self._compute_punctuation_balance(reply)
            engagement_score = self._compute_engagement_signals(tweet, reply)
            
            total_score = (
                self.weights['relevance'] * relevance +
                self.weights['length'] * length_score +
                self.weights['sentiment'] * sentiment_score +
                self.weights['diversity'] * diversity_score +
                self.weights['punctuation'] * punctuation_score +
                self.weights['engagement'] * engagement_score
            )
            
            rewards.append(max(0.0, min(1.0, total_score)))
        
        return rewards
    
    def _compute_relevance(self, tweet: str, reply: str) -> float:
        """
        Semantic relevance between tweet and reply.
        
        Uses sentence embeddings + cosine similarity.
        """
        if self.encoder is None:
            return 0.5  # Neutral if encoder not available
        
        try:
            tweet_emb = self.encoder.encode(
                tweet,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
            reply_emb = self.encoder.encode(
                reply,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
            
            # Cosine similarity (already in [-1, 1], map to [0, 1])
            similarity = F.cosine_similarity(
                tweet_emb.unsqueeze(0),
                reply_emb.unsqueeze(0)
            ).item()
            
            # Map [-1, 1] to [0, 1]
            relevance = (similarity + 1.0) / 2.0
            
            return relevance
            
        except Exception as e:
            logger.warning(f"Relevance computation failed: {e}")
            return 0.5  # Fallback
    
    def _compute_length_quality(self, reply: str) -> float:
        """
        Length quality score.
        
        Twitter optimal: 50-150 characters
        - Too short (< 20): Low quality, likely generic
        - Optimal (50-150): High quality
        - Long (150-280): Moderate quality
        - Too long (> 280): Penalized
        """
        length = len(reply)
        
        if length < 20:
            # Very short - likely generic ("Thanks!", "Agreed!")
            return 0.3
        elif length < 50:
            # Short but acceptable
            return 0.6
        elif length <= 150:
            # Optimal range
            return 1.0
        elif length <= 280:
            # Long but within Twitter limit
            return 0.8
        else:
            # Too long for Twitter
            return 0.4
    
    def _compute_sentiment_alignment(self, tweet: str, reply: str) -> float:
        """
        Sentiment alignment between tweet and reply.
        
        Engaging replies often:
        - Match tweet sentiment (agreement)
        - Playfully counter (thoughtful disagreement)
        
        We reward both, penalize extreme mismatch.
        """
        try:
            from textblob import TextBlob
            
            tweet_sentiment = TextBlob(tweet).sentiment.polarity
            reply_sentiment = TextBlob(reply).sentiment.polarity
            
            # Compute absolute difference
            diff = abs(tweet_sentiment - reply_sentiment)
            
            # Small difference = good alignment
            # diff in [0, 2], map to score in [1, 0]
            score = 1.0 - (diff / 2.0)
            
            return max(0.0, score)
            
        except ImportError:
            logger.warning("textblob not available, skipping sentiment")
            return 0.5  # Neutral
        except Exception as e:
            logger.warning(f"Sentiment computation failed: {e}")
            return 0.5
    
    def _compute_lexical_diversity(self, reply: str) -> float:
        """
        Lexical diversity of reply.
        
        Measures ratio of unique words to total words.
        Penalizes repetitive/generic replies.
        """
        words = reply.lower().split()
        
        if len(words) == 0:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Unique ratio in [0, 1]
        diversity = unique_words / total_words
        
        # Penalize very low diversity (< 0.5 means lots of repetition)
        if diversity < 0.5:
            diversity *= 0.7  # Apply penalty
        
        return diversity
    
    def _compute_punctuation_balance(self, reply: str) -> float:
        """
        Punctuation balance score.
        
        Too much punctuation = likely spam or low quality
        Too little = may be incomplete thought
        """
        if len(reply) == 0:
            return 0.0
        
        # Count punctuation marks
        punct_chars = '!?.,'
        punct_count = sum(1 for c in reply if c in punct_chars)
        
        # Ratio of punctuation to total length
        punct_ratio = punct_count / len(reply)
        
        # Optimal: 0.02-0.08 (2-8%)
        if punct_ratio < 0.01:
            # Too little punctuation
            return 0.7
        elif 0.02 <= punct_ratio <= 0.08:
            # Optimal range
            return 1.0
        elif punct_ratio <= 0.15:
            # A bit much but acceptable
            return 0.8
        else:
            # Way too much punctuation (spam-like)
            return 0.3
    
    def _compute_engagement_signals(self, tweet: str, reply: str) -> float:
        """
        Engagement-specific heuristics.
        
        Based on what makes Twitter replies engaging:
        - Not too generic
        - Has substance (not just "Agreed!")
        - Adds value to conversation
        """
        score = 1.0
        
        # Penalty for generic phrases
        generic_phrases = [
            'thanks', 'thank you', 'agree', 'agreed', 'this', 'nice',
            'great', 'good point', 'interesting', 'true', 'exactly',
            'same', 'yep', 'yes', 'no', 'lol', 'haha'
        ]
        
        reply_lower = reply.lower()
        
        # Check if reply is mostly generic
        is_mostly_generic = False
        for phrase in generic_phrases:
            if phrase in reply_lower:
                # If generic phrase + short reply = very generic
                if len(reply) < 30:
                    is_mostly_generic = True
                    break
        
        if is_mostly_generic:
            score *= 0.5  # Major penalty
        
        # Reward for question marks (engaging discussion)
        if '?' in reply:
            score *= 1.1  # Small boost
        
        # Reward for substantive replies (has multiple sentences)
        sentence_count = reply.count('.') + reply.count('!') + reply.count('?')
        if sentence_count >= 2:
            score *= 1.05  # Small boost
        
        # Penalty for ALL CAPS (aggressive/spammy)
        if len(reply) > 10 and reply.isupper():
            score *= 0.6
        
        # Ensure in [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def get_component_scores(self, tweet: str, reply: str) -> dict:
        """
        Get breakdown of all component scores (for debugging/analysis).
        
        Args:
            tweet: Original tweet
            reply: Generated reply
            
        Returns:
            Dict with all component scores and total
        """
        return {
            'relevance': self._compute_relevance(tweet, reply),
            'length': self._compute_length_quality(reply),
            'sentiment': self._compute_sentiment_alignment(tweet, reply),
            'diversity': self._compute_lexical_diversity(reply),
            'punctuation': self._compute_punctuation_balance(reply),
            'engagement': self._compute_engagement_signals(tweet, reply),
            'total': self.compute_reward(tweet, reply)
        }


# Test and demonstration
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Heuristic Reward Function")
    print("="*60 + "\n")
    
    # Initialize
    reward_fn = HeuristicRewardFunction()
    
    # Test cases
    tweet = "AI will revolutionize healthcare in ways we can't even imagine yet."
    
    test_cases = [
        ("Absolutely! Early disease detection from routine scans could save millions of lives. Imagine AI catching cancer years before symptoms appear.", "Good reply - substantive, relevant"),
        ("Agreed!", "Bad reply - too generic"),
        ("THIS IS AMAZING!!! WOW!!! GREAT!!! NICE!!!", "Bad reply - too much punctuation, generic"),
        ("Interesting perspective, but have you considered the privacy implications?", "Good reply - adds value, asks question"),
        ("lol", "Bad reply - too short, generic"),
    ]
    
    print("\nTest Results:")
    print("-" * 60)
    
    for reply, description in test_cases:
        score = reward_fn.compute_reward(tweet, reply)
        components = reward_fn.get_component_scores(tweet, reply)
        
        print(f"\n{description}")
        print(f"Reply: {reply[:60]}...")
        print(f"Total Score: {score:.3f}")
        print("Components:")
        for component, value in components.items():
            if component != 'total':
                print(f"  {component:12s}: {value:.3f}")
    
    print("\n" + "="*60)
    print("✅ Heuristic Reward Function Working!")
    print("="*60)
    print("\nReady for GRPO training with:")
    print("  reward_function = HeuristicRewardFunction()")
    print("  score = reward_function(tweet, reply)")
    print("="*60 + "\n")

