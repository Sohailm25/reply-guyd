"""
Data Cleaner
Clean and preprocess collected tweet-reply pairs
Implements deduplication and normalization strategies
"""

import re
import logging
from typing import List, Dict, Set
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and preprocess data
    
    From current_research.md:
    - Semantic deduplication (similarity > 0.9)
    - Text normalization (elongated chars, slang, hashtags)
    - Emoji handling
    """
    
    def __init__(self, config: Dict = None):
        """Initialize cleaner"""
        
        self.config = config or {}
        
        # Semantic similarity threshold for deduplication
        self.similarity_threshold = self.config.get("semantic_similarity_threshold", 0.9)
        
        # Load sentence transformer for deduplication
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence encoder loaded for semantic deduplication")
        except Exception as e:
            logger.warning(f"Could not load sentence encoder: {e}")
            self.encoder = None
    
    def clean_batch(self, pairs: List[Dict]) -> List[Dict]:
        """
        Clean a batch of tweet-reply pairs
        
        Steps:
        1. Normalize text (elongations, etc.)
        2. Remove exact duplicates
        3. Remove semantic duplicates
        4. Sort by quality
        
        Returns:
            Cleaned list of pairs
        """
        logger.info(f"Cleaning {len(pairs)} pairs...")
        
        # Step 1: Normalize text
        for pair in pairs:
            pair["tweet"] = self.normalize_text(pair["tweet"])
            pair["reply"] = self.normalize_text(pair["reply"])
        
        # Step 2: Remove exact duplicates
        pairs = self._remove_exact_duplicates(pairs)
        logger.info(f"After exact deduplication: {len(pairs)} pairs")
        
        # Step 3: Remove semantic duplicates
        if self.encoder:
            pairs = self._remove_semantic_duplicates(pairs)
            logger.info(f"After semantic deduplication: {len(pairs)} pairs")
        
        # Step 4: Sort by quality (engagement)
        pairs = self._sort_by_quality(pairs)
        
        logger.info(f"Cleaning complete: {len(pairs)} final pairs")
        
        return pairs
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize social media text
        
        From current_research.md preprocessing:
        - Normalize elongated characters ("soooo" -> "so")
        - Process hashtags (#TextMining -> "Text Mining")
        - Handle emojis (preserve or convert to text)
        - Remove extra whitespace
        """
        # Normalize elongated characters (3+ repetitions -> 2)
        # "soooooo" -> "soo", "hahahaha" -> "haha"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Process hashtags: #MachineLearning -> Machine Learning
        def process_hashtag(match):
            tag = match.group(1)
            # Split camelCase: MachineLearning -> Machine Learning
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
            return spaced
        
        text = re.sub(r'#(\w+)', process_hashtag, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Note: We preserve emojis as they carry engagement signals
        # Could convert to text descriptions if needed:
        # import emoji
        # text = emoji.demojize(text)  # 👍 -> :thumbs_up:
        
        return text
    
    def _remove_exact_duplicates(self, pairs: List[Dict]) -> List[Dict]:
        """Remove pairs with identical reply text"""
        seen_replies: Set[str] = set()
        unique_pairs = []
        
        for pair in pairs:
            reply = pair["reply"]
            
            if reply not in seen_replies:
                seen_replies.add(reply)
                unique_pairs.append(pair)
        
        removed = len(pairs) - len(unique_pairs)
        if removed > 0:
            logger.info(f"Removed {removed} exact duplicates")
        
        return unique_pairs
    
    def _remove_semantic_duplicates(self, pairs: List[Dict]) -> List[Dict]:
        """
        Remove semantically similar replies using embeddings
        
        From current_research.md: similarity threshold 0.9
        Keep the reply with higher engagement when duplicates found
        """
        if self.encoder is None:
            logger.warning("Encoder not available, skipping semantic deduplication")
            return pairs
        
        # Extract replies
        replies = [pair["reply"] for pair in pairs]
        
        # Generate embeddings
        logger.info("Generating embeddings for semantic deduplication...")
        embeddings = self.encoder.encode(replies, show_progress_bar=True)
        
        # Find duplicates
        keep_indices = []
        duplicate_groups = defaultdict(list)
        
        for i in range(len(embeddings)):
            # Check if already marked as duplicate
            is_duplicate = False
            
            for kept_idx in keep_indices:
                similarity = cosine_similarity(
                    [embeddings[i]], 
                    [embeddings[kept_idx]]
                )[0][0]
                
                if similarity > self.similarity_threshold:
                    # This is a duplicate
                    is_duplicate = True
                    duplicate_groups[kept_idx].append(i)
                    
                    # Keep the one with higher engagement
                    if pairs[i]["reply_likes"] > pairs[kept_idx]["reply_likes"]:
                        # Replace kept index with this one
                        keep_indices.remove(kept_idx)
                        keep_indices.append(i)
                        duplicate_groups[i] = duplicate_groups[kept_idx] + [kept_idx]
                        del duplicate_groups[kept_idx]
                    
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # Filter pairs
        unique_pairs = [pairs[i] for i in sorted(keep_indices)]
        
        removed = len(pairs) - len(unique_pairs)
        if removed > 0:
            logger.info(f"Removed {removed} semantic duplicates")
        
        return unique_pairs
    
    def _sort_by_quality(self, pairs: List[Dict]) -> List[Dict]:
        """
        Sort pairs by quality score
        
        Quality = weighted combination of:
        - Reply engagement (likes + retweets)
        - Author credibility (followers)
        - Timing (not too fast, not too slow)
        """
        def quality_score(pair: Dict) -> float:
            # Engagement score (normalized)
            likes = pair.get("reply_likes", 0)
            retweets = pair.get("reply_retweets", 0)
            engagement = likes + (retweets * 2)  # Retweets worth more
            
            # Author credibility
            followers = pair.get("reply_author_followers", 0)
            credibility = min(followers / 10000, 1.0)  # Cap at 10K
            
            # Timing score (prefer 5 min - 2 hours)
            time_diff = pair.get("reply_time_diff_seconds", 0)
            if 300 <= time_diff <= 7200:  # 5 min to 2 hours
                timing = 1.0
            elif time_diff < 300:
                timing = 0.5  # Too fast
            else:
                timing = 0.7  # Later is okay
            
            # Weighted combination
            score = (
                engagement * 0.6 +
                credibility * 100 * 0.2 +
                timing * 50 * 0.2
            )
            
            return score
        
        sorted_pairs = sorted(pairs, key=quality_score, reverse=True)
        return sorted_pairs
    
    def analyze_dataset(self, pairs: List[Dict]) -> Dict:
        """
        Analyze dataset statistics
        
        Returns summary statistics for manual review
        """
        stats = {
            "total_pairs": len(pairs),
            "reply_length": {
                "min": min(len(p["reply"]) for p in pairs),
                "max": max(len(p["reply"]) for p in pairs),
                "mean": np.mean([len(p["reply"]) for p in pairs]),
                "median": np.median([len(p["reply"]) for p in pairs]),
            },
            "engagement": {
                "min_likes": min(p["reply_likes"] for p in pairs),
                "max_likes": max(p["reply_likes"] for p in pairs),
                "mean_likes": np.mean([p["reply_likes"] for p in pairs]),
                "median_likes": np.median([p["reply_likes"] for p in pairs]),
            },
            "authors": {
                "unique_count": len(set(p["reply_author"] for p in pairs)),
                "avg_followers": np.mean([p.get("reply_author_followers", 0) for p in pairs]),
            },
            "topics": {
                "unique_tweets": len(set(p["tweet_id"] for p in pairs)),
                "avg_replies_per_tweet": len(pairs) / len(set(p["tweet_id"] for p in pairs)),
            }
        }
        
        return stats


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    cleaner = DataCleaner()
    
    # Test with sample data
    test_pairs = [
        {
            "tweet_id": "1",
            "tweet": "Check out my new blog post!!! #AI #MachineLearning",
            "reply": "This is sooooo cool! Great work!",
            "reply_author": "user1",
            "reply_likes": 10,
            "reply_retweets": 2,
            "reply_author_followers": 1000,
            "reply_time_diff_seconds": 600,
        },
        {
            "tweet_id": "1",
            "tweet": "Check out my new blog post!!! #AI #MachineLearning",
            "reply": "This is so cool! Great work!",  # Similar to above
            "reply_author": "user2",
            "reply_likes": 5,
            "reply_retweets": 0,
            "reply_author_followers": 500,
            "reply_time_diff_seconds": 1200,
        },
        {
            "tweet_id": "2",
            "tweet": "What's your favorite Python library?",
            "reply": "NumPy for sure! Can't live without it.",
            "reply_author": "user3",
            "reply_likes": 15,
            "reply_retweets": 3,
            "reply_author_followers": 5000,
            "reply_time_diff_seconds": 900,
        },
    ]
    
    # Clean
    cleaned = cleaner.clean_batch(test_pairs)
    
    print(f"\n=== Cleaned Data ===")
    print(f"Original: {len(test_pairs)} pairs")
    print(f"Cleaned: {len(cleaned)} pairs")
    
    # Analyze
    stats = cleaner.analyze_dataset(cleaned)
    print(f"\n=== Dataset Statistics ===")
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Reply length: {stats['reply_length']['mean']:.1f} chars (avg)")
    print(f"Reply engagement: {stats['engagement']['mean_likes']:.1f} likes (avg)")
    print(f"Unique authors: {stats['authors']['unique_count']}")


if __name__ == "__main__":
    main()

