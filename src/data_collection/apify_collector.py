"""
Apify-based Twitter Data Collector
Based on current_research.md: $0.25 per 1,000 tweets, much better than official API
"""

import os
import time
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv
import yaml

load_dotenv()

logger = logging.getLogger(__name__)


class ApifyCollector:
    """
    Collect Twitter data using Apify actors
    
    Advantages over official API:
    - ~$0.25 per 1,000 tweets vs $200 for 10,000 posts
    - Can filter by engagement at collection time
    - Access to historical data (not just last 7 days)
    """
    
    def __init__(self, config_path: str = "config/data_collection_config.yaml"):
        """Initialize Apify client with configuration"""
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Apify client
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables")
        
        self.client = ApifyClient(api_token)
        self.actor_id = self.config["apify"]["actor_id"]
        
        # Setup storage
        self.raw_data_dir = Path(self.config["storage"]["raw_data_dir"])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint management
        self.checkpoint_file = self.raw_data_dir / "collection_checkpoint.json"
        self.checkpoint_enabled = self.config["storage"]["checkpoint_enabled"]
        
        logger.info(f"Apify collector initialized with actor: {self.actor_id}")
    
    def collect_tweets_and_replies(
        self,
        search_query: str,
        max_tweets: int = 100,
        max_replies_per_tweet: int = 50
    ) -> List[Dict]:
        """
        Collect tweets matching query and their high-engagement replies
        
        Args:
            search_query: Twitter search query (e.g., "AI lang:en min_faves:200")
            max_tweets: Maximum tweets to collect
            max_replies_per_tweet: Maximum replies per tweet
        
        Returns:
            List of tweet-reply pairs with metadata
        """
        logger.info(f"Starting collection for query: {search_query}")
        logger.info(f"Target: {max_tweets} tweets, up to {max_replies_per_tweet} replies each")
        
        # Step 1: Collect main tweets
        tweets = self._collect_main_tweets(search_query, max_tweets)
        logger.info(f"Collected {len(tweets)} main tweets")
        
        # Step 2: Collect replies for each tweet
        tweet_reply_pairs = []
        
        for idx, tweet in enumerate(tweets):
            logger.info(f"Processing tweet {idx+1}/{len(tweets)}: {tweet['id']}")
            
            # Get replies
            replies = self._collect_tweet_replies(
                tweet['id'],
                max_replies=max_replies_per_tweet
            )
            
            # Filter high-engagement replies
            filtered_replies = self._filter_replies(replies, tweet)
            
            # Create pairs
            for reply in filtered_replies:
                pair = self._create_training_pair(tweet, reply)
                tweet_reply_pairs.append(pair)
            
            logger.info(f"Found {len(filtered_replies)} high-quality replies")
            
            # Checkpoint
            if self.checkpoint_enabled and (idx + 1) % 10 == 0:
                self._save_checkpoint(tweet_reply_pairs)
            
            # Rate limiting (be respectful to Apify)
            time.sleep(2)
        
        logger.info(f"Collection complete: {len(tweet_reply_pairs)} training pairs")
        return tweet_reply_pairs
    
    def _collect_main_tweets(self, search_query: str, max_tweets: int) -> List[Dict]:
        """Collect main tweets using Apify actor"""
        
        run_input = {
            "searchTerms": [search_query],
            "maxItems": max_tweets,
            "sort": "Latest",  # Or "Top" for most engaging
            "tweetLanguage": "en",
            "customMapFunction": "(object) => { return {...object} }",
            "addUserInfo": True,
        }
        
        try:
            # Run the actor
            logger.info("Starting Apify actor run...")
            run = self.client.actor(self.actor_id).call(
                run_input=run_input,
                timeout_secs=self.config["apify"]["timeout_seconds"]
            )
            
            # Fetch results
            tweets = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                tweets.append(self._normalize_tweet(item))
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error collecting tweets via Apify: {e}")
            raise
    
    def _collect_tweet_replies(self, tweet_id: str, max_replies: int = 50) -> List[Dict]:
        """
        Collect replies to a specific tweet
        
        Note: This uses conversation search, which may require a different actor
        or custom implementation. Apify's tweet-scraper might not directly support this.
        """
        
        # Build conversation query
        # conversation_id in Twitter search shows all tweets in a thread
        search_query = f"conversation_id:{tweet_id}"
        
        run_input = {
            "searchTerms": [search_query],
            "maxItems": max_replies,
            "sort": "Latest",
            "addUserInfo": True,
        }
        
        try:
            run = self.client.actor(self.actor_id).call(
                run_input=run_input,
                timeout_secs=self.config["apify"]["timeout_seconds"]
            )
            
            replies = []
            for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
                # Filter out the original tweet itself
                if item.get("id") != tweet_id:
                    replies.append(self._normalize_tweet(item))
            
            return replies
            
        except Exception as e:
            logger.error(f"Error collecting replies for tweet {tweet_id}: {e}")
            return []
    
    def _filter_replies(self, replies: List[Dict], original_tweet: Dict) -> List[Dict]:
        """
        Filter replies based on quality criteria from gameplan.md
        
        Filters:
        - Engagement threshold (min likes)
        - Author filters (avoid celebrities and bots)
        - Timing filter (avoid first-reply advantage)
        - Content filters (length, media, URLs)
        """
        filters = self.config["collection"]["reply_filters"]
        filtered = []
        
        original_time = datetime.fromisoformat(original_tweet["created_at"].replace("Z", "+00:00"))
        
        for reply in replies:
            # Engagement filter
            likes = reply.get("favorite_count", 0)
            if not (filters["min_likes"] <= likes <= filters["max_likes"]):
                continue
            
            retweets = reply.get("retweet_count", 0)
            if not (filters["min_retweets"] <= retweets <= filters["max_retweets"]):
                continue
            
            # Author filters (avoid celebrity effect)
            followers = reply.get("author", {}).get("followers_count", 0)
            if followers < filters["min_follower_count"]:
                continue  # Likely bot/spam
            if followers > filters["max_follower_count"]:
                continue  # Celebrity advantage
            
            # Timing filter (from gameplan.md: avoid first 5 min advantage)
            reply_time = datetime.fromisoformat(reply["created_at"].replace("Z", "+00:00"))
            time_diff = (reply_time - original_time).total_seconds()
            
            if time_diff < filters["min_time_delay_seconds"]:
                continue  # Too fast, timing advantage
            if time_diff > filters["max_time_delay_seconds"]:
                continue  # Too late, might be off-topic
            
            # Content filters
            text = reply.get("full_text", "")
            
            if not (filters["min_length"] <= len(text) <= filters["max_length"]):
                continue
            
            if filters["has_media"] is False and reply.get("has_media", False):
                continue  # Can't replicate media-based engagement
            
            if filters["has_urls"] is False and ("http" in text.lower()):
                continue  # URLs drive engagement we can't replicate
            
            # Passed all filters
            reply["time_diff_seconds"] = time_diff
            filtered.append(reply)
        
        return filtered
    
    def _create_training_pair(self, tweet: Dict, reply: Dict) -> Dict:
        """Create a training pair with metadata"""
        return {
            "tweet_id": tweet["id"],
            "tweet": tweet["full_text"],
            "tweet_author": tweet.get("author", {}).get("screen_name", "unknown"),
            "tweet_likes": tweet.get("favorite_count", 0),
            "tweet_retweets": tweet.get("retweet_count", 0),
            "tweet_created_at": tweet["created_at"],
            
            "reply_id": reply["id"],
            "reply": reply["full_text"],
            "reply_author": reply.get("author", {}).get("screen_name", "unknown"),
            "reply_author_followers": reply.get("author", {}).get("followers_count", 0),
            "reply_likes": reply.get("favorite_count", 0),
            "reply_retweets": reply.get("retweet_count", 0),
            "reply_created_at": reply["created_at"],
            "reply_time_diff_seconds": reply.get("time_diff_seconds", 0),
            
            "collected_at": datetime.utcnow().isoformat(),
        }
    
    def _normalize_tweet(self, raw_tweet: Dict) -> Dict:
        """
        Normalize Apify tweet format to consistent structure
        
        Different Apify actors may return different schemas
        """
        return {
            "id": raw_tweet.get("id") or raw_tweet.get("id_str"),
            "full_text": raw_tweet.get("full_text") or raw_tweet.get("text", ""),
            "created_at": raw_tweet.get("created_at", ""),
            "favorite_count": raw_tweet.get("favorite_count", 0),
            "retweet_count": raw_tweet.get("retweet_count", 0),
            "reply_count": raw_tweet.get("reply_count", 0),
            "has_media": bool(raw_tweet.get("entities", {}).get("media")),
            "author": {
                "screen_name": raw_tweet.get("user", {}).get("screen_name", ""),
                "followers_count": raw_tweet.get("user", {}).get("followers_count", 0),
                "verified": raw_tweet.get("user", {}).get("verified", False),
            },
            "raw": raw_tweet,  # Keep original for debugging
        }
    
    def _save_checkpoint(self, data: List[Dict]):
        """Save collection checkpoint"""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "pairs_collected": len(data),
        }
        
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(data)} pairs")
    
    def save_to_file(self, data: List[Dict], filename: str = None):
        """Save collected data to file"""
        if filename is None:
            filename = f"apify_collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath = self.raw_data_dir / filename
        
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Saved {len(data)} pairs to {filepath}")
        return filepath


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    collector = ApifyCollector()
    
    # Collect data for one query
    search_query = "AI OR artificial intelligence lang:en min_faves:200"
    pairs = collector.collect_tweets_and_replies(
        search_query=search_query,
        max_tweets=10,  # Start small for testing
        max_replies_per_tweet=30
    )
    
    # Save results
    collector.save_to_file(pairs)
    
    print(f"\nCollected {len(pairs)} tweet-reply training pairs")
    
    # Show sample
    if pairs:
        print("\nSample pair:")
        sample = pairs[0]
        print(f"Tweet: {sample['tweet'][:100]}...")
        print(f"Reply: {sample['reply'][:100]}...")
        print(f"Reply engagement: {sample['reply_likes']} likes")


if __name__ == "__main__":
    main()

