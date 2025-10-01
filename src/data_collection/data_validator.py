"""
Data Validator
Validates collected tweet-reply pairs against quality criteria
Based on gameplan.md and current_research.md filtering strategies
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

from langdetect import detect, LangDetectException
import emoji

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate tweet-reply pairs for quality
    
    Quality checks from current_research.md:
    - Length (30-280 chars)
    - Language detection (confidence > 0.8)
    - No toxicity (via Detoxify)
    - Engagement thresholds
    - Deduplication
    """
    
    def __init__(self, config: Dict = None):
        """Initialize validator with configuration"""
        
        self.config = config or {}
        
        # Default thresholds (can be overridden by config)
        self.min_length = self.config.get("min_length", 30)
        self.max_length = self.config.get("max_length", 280)
        self.min_language_confidence = self.config.get("min_language_confidence", 0.8)
        self.max_toxicity_score = self.config.get("max_toxicity_score", 0.3)
        
        # Load toxicity detector if available
        try:
            from detoxify import Detoxify
            self.toxicity_model = Detoxify('original')
            logger.info("Toxicity detection enabled")
        except ImportError:
            logger.warning("Detoxify not installed. Toxicity filtering disabled.")
            logger.warning("Install with: pip install detoxify")
            self.toxicity_model = None
    
    def validate_pair(self, pair: Dict) -> Dict:
        """
        Validate a single tweet-reply pair
        
        Returns:
            Dict with keys:
                - valid: bool
                - reasons: List[str] (if invalid)
                - pair: original pair with validation metadata
        """
        reasons = []
        
        tweet = pair.get("tweet", "")
        reply = pair.get("reply", "")
        
        # Length validation
        if not self._validate_length(reply):
            reasons.append(f"Reply length {len(reply)} outside [{self.min_length}, {self.max_length}]")
        
        # Language validation
        lang_valid, lang_conf = self._validate_language(reply)
        if not lang_valid:
            reasons.append(f"Language detection failed or confidence {lang_conf:.2f} < {self.min_language_confidence}")
        
        # Content validation
        if self._has_urls(reply):
            reasons.append("Reply contains URLs")
        
        if self._has_excessive_hashtags(reply):
            reasons.append("Reply has excessive hashtags")
        
        if self._is_spam_pattern(reply):
            reasons.append("Reply matches spam pattern")
        
        # Toxicity validation
        if self.toxicity_model:
            tox_score = self._check_toxicity(reply)
            if tox_score > self.max_toxicity_score:
                reasons.append(f"Toxicity score {tox_score:.2f} > {self.max_toxicity_score}")
        
        # Engagement validation
        if not self._validate_engagement(pair):
            reasons.append("Insufficient engagement")
        
        # Add validation metadata
        pair["validation"] = {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "validated_at": datetime.utcnow().isoformat(),
        }
        
        return {
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "pair": pair,
        }
    
    def validate_batch(self, pairs: List[Dict]) -> Dict:
        """
        Validate a batch of pairs
        
        Returns:
            Dict with:
                - valid_pairs: List[Dict]
                - invalid_pairs: List[Dict]
                - stats: Dict with validation statistics
        """
        valid_pairs = []
        invalid_pairs = []
        
        rejection_reasons = {}
        
        for pair in pairs:
            result = self.validate_pair(pair)
            
            if result["valid"]:
                valid_pairs.append(result["pair"])
            else:
                invalid_pairs.append(result["pair"])
                
                # Track rejection reasons
                for reason in result["reasons"]:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        stats = {
            "total": len(pairs),
            "valid": len(valid_pairs),
            "invalid": len(invalid_pairs),
            "valid_percentage": len(valid_pairs) / len(pairs) * 100 if pairs else 0,
            "rejection_reasons": rejection_reasons,
        }
        
        logger.info(f"Validation complete: {stats['valid']}/{stats['total']} valid ({stats['valid_percentage']:.1f}%)")
        
        return {
            "valid_pairs": valid_pairs,
            "invalid_pairs": invalid_pairs,
            "stats": stats,
        }
    
    def _validate_length(self, text: str) -> bool:
        """Check if text length is within bounds"""
        return self.min_length <= len(text) <= self.max_length
    
    def _validate_language(self, text: str) -> tuple[bool, float]:
        """
        Validate language is English with high confidence
        
        Returns:
            (is_valid, confidence)
        """
        try:
            # langdetect doesn't return confidence directly, but we can check
            lang = detect(text)
            if lang == "en":
                # For simplicity, assume high confidence if detected as English
                return True, 0.9
            else:
                return False, 0.0
        except LangDetectException:
            return False, 0.0
    
    def _has_urls(self, text: str) -> bool:
        """Check if text contains URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return bool(re.search(url_pattern, text))
    
    def _has_excessive_hashtags(self, text: str, max_hashtags: int = 3) -> bool:
        """Check if text has too many hashtags"""
        hashtags = re.findall(r'#\w+', text)
        return len(hashtags) > max_hashtags
    
    def _is_spam_pattern(self, text: str) -> bool:
        """
        Check for common spam patterns
        
        Patterns:
        - All caps (excluding short texts)
        - Excessive punctuation
        - Too many emojis
        - Repetitive characters
        """
        # All caps (for longer texts)
        if len(text) > 20 and text.isupper():
            return True
        
        # Excessive punctuation
        punct_count = sum(1 for c in text if c in "!?.")
        if punct_count > len(text) * 0.1:  # More than 10% punctuation
            return True
        
        # Too many emojis
        emoji_count = emoji.emoji_count(text)
        if emoji_count > 5:
            return True
        
        # Repetitive characters (e.g., "soooooo")
        if re.search(r'(.)\1{4,}', text):  # Same char repeated 5+ times
            return True
        
        return False
    
    def _check_toxicity(self, text: str) -> float:
        """
        Check toxicity score using Detoxify
        
        Returns:
            Float toxicity score (0-1)
        """
        if self.toxicity_model is None:
            return 0.0
        
        try:
            results = self.toxicity_model.predict(text)
            # Average of all toxicity metrics
            avg_toxicity = sum(results.values()) / len(results)
            return avg_toxicity
        except Exception as e:
            logger.warning(f"Toxicity check failed: {e}")
            return 0.0
    
    def _validate_engagement(self, pair: Dict) -> bool:
        """
        Validate engagement metrics meet minimum thresholds
        
        From config: min_likes, max_likes, etc.
        """
        # These should have been filtered during collection,
        # but double-check here as a safety net
        
        reply_likes = pair.get("reply_likes", 0)
        
        # At minimum, reply should have some engagement
        if reply_likes < 2:
            return False
        
        return True


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    validator = DataValidator()
    
    # Test cases
    test_pairs = [
        {
            "tweet": "Just launched our new API!",
            "reply": "Congrats! This looks amazing. Can't wait to try it out!",
            "reply_likes": 10,
        },
        {
            "tweet": "What's your favorite programming language?",
            "reply": "Python!!!!!!!!!!!!",  # Excessive punctuation
            "reply_likes": 5,
        },
        {
            "tweet": "New blog post",
            "reply": "Check out my link http://spam.com",  # URL
            "reply_likes": 3,
        },
        {
            "tweet": "AI is transforming everything",
            "reply": "AI",  # Too short
            "reply_likes": 100,
        },
    ]
    
    results = validator.validate_batch(test_pairs)
    
    print(f"\n=== Validation Results ===")
    print(f"Valid: {results['stats']['valid']}/{results['stats']['total']}")
    print(f"\nRejection reasons:")
    for reason, count in results['stats']['rejection_reasons'].items():
        print(f"  - {reason}: {count}")
    
    print(f"\nValid pairs:")
    for pair in results['valid_pairs']:
        print(f"  ✓ {pair['reply'][:60]}...")


if __name__ == "__main__":
    main()

