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
from email.utils import parsedate_to_datetime

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
        
        # Global author diversity tracking (persists across all queries)
        self.author_counts = {}
        
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
        raw_tweets = self._collect_main_tweets(search_query, max_tweets)
        logger.info(f"Collected {len(raw_tweets)} raw tweets from Apify")
        
        # Step 2: Filter tweets based on quality criteria (including verification)
        tweets = self._filter_main_tweets(raw_tweets)
        logger.info(f"After filtering: {len(tweets)} tweets passed quality filters (verified, engagement, etc.)")
        
        if len(tweets) == 0:
            logger.warning("No tweets passed filters. Consider adjusting tweet_filters in config.")
            return []
        
        # Step 3: Collect replies for each tweet with GLOBAL author diversity tracking
        tweet_reply_pairs = []
        max_per_author = self.config["collection"]["reply_filters"].get("max_replies_per_author", 10)
        
        # Log current author diversity state
        logger.info(f"Author diversity tracking: {len(self.author_counts)} unique authors already tracked, {sum(self.author_counts.values())} total replies")
        
        for idx, tweet in enumerate(tweets):
            logger.info(f"Processing tweet {idx+1}/{len(tweets)}: {tweet['id']}")
            
            # Get replies
            replies = self._collect_tweet_replies(
                tweet['id'],
                max_replies=max_replies_per_tweet
            )
            
            # Filter high-engagement replies
            filtered_replies = self._filter_replies(replies, tweet)
            
            # Create pairs with GLOBAL author diversity enforcement
            added_for_tweet = 0
            for reply in filtered_replies:
                # Extract author identifier - prefer ID over screen_name
                author_id_field = reply.get("author", {}).get("id", "")
                author_screen_name = reply.get("author", {}).get("screen_name", "")
                
                # Use author ID if available, else screen_name, else "unknown"
                if author_id_field:
                    author_id = f"id_{author_id_field}"
                elif author_screen_name:
                    author_id = author_screen_name
                else:
                    author_id = "unknown"
                
                # Check GLOBAL author limit (tracked across all queries)
                if self.author_counts.get(author_id, 0) >= max_per_author:
                    logger.debug(f"Skipping reply from {author_id} (already have {self.author_counts[author_id]} replies globally)")
                    continue
                
                pair = self._create_training_pair(tweet, reply)
                tweet_reply_pairs.append(pair)
                self.author_counts[author_id] = self.author_counts.get(author_id, 0) + 1
                added_for_tweet += 1
            
            logger.info(f"Found {len(filtered_replies)} high-quality replies ({added_for_tweet} added, {len(filtered_replies) - added_for_tweet} skipped for author diversity)")
            
            # Checkpoint
            if self.checkpoint_enabled and (idx + 1) % 10 == 0:
                self._save_checkpoint(tweet_reply_pairs)
            
            # Rate limiting (be respectful to Apify)
            time.sleep(2)
        
        logger.info(f"Collection complete: {len(tweet_reply_pairs)} training pairs from {len(self.author_counts)} unique authors (GLOBAL)")
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
    
    def _filter_main_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """Filter main tweets based on quality criteria (verified, engagement, etc.)"""
        filters = self.config["collection"]["tweet_filters"]
        filtered = []
        
        rejection_stats = {
            "total": len(tweets),
            "not_verified": 0,
            "engagement_low": 0,
            "engagement_high": 0,
            "is_retweet": 0,
            "is_quote": 0,
            "no_replies": 0,
            "passed": 0,
        }
        
        for tweet in tweets:
            # Verification filter (if required)
            if filters.get("verified") == True:
                is_verified = tweet.get("author", {}).get("verified", False)
                if not is_verified:
                    rejection_stats["not_verified"] += 1
                    continue
            
            # Engagement filters
            likes = tweet.get("favorite_count", 0)
            retweets = tweet.get("retweet_count", 0)
            
            if likes < filters.get("min_likes", 0):
                rejection_stats["engagement_low"] += 1
                continue
            if likes > filters.get("max_likes", float('inf')):
                rejection_stats["engagement_high"] += 1
                continue
            
            if retweets < filters.get("min_retweets", 0):
                rejection_stats["engagement_low"] += 1
                continue
            if retweets > filters.get("max_retweets", float('inf')):
                rejection_stats["engagement_high"] += 1
                continue
            
            # Content type filters
            if filters.get("is_retweet") == False and tweet.get("is_retweet", False):
                rejection_stats["is_retweet"] += 1
                continue
            
            if filters.get("is_quote") == False and tweet.get("is_quote", False):
                rejection_stats["is_quote"] += 1
                continue
            
            # Passed all filters
            rejection_stats["passed"] += 1
            filtered.append(tweet)
        
        # Log statistics
        if rejection_stats["total"] > 0:
            logger.info(f"\n📊 Tweet Filter Statistics ({rejection_stats['total']} total tweets):")
            logger.info(f"  ✓ Passed all filters: {rejection_stats['passed']}")
            if rejection_stats["not_verified"] > 0:
                logger.info(f"  ✗ Not verified: {rejection_stats['not_verified']}")
            if rejection_stats["engagement_low"] > 0:
                logger.info(f"  ✗ Engagement too low: {rejection_stats['engagement_low']}")
            if rejection_stats["engagement_high"] > 0:
                logger.info(f"  ✗ Engagement too high: {rejection_stats['engagement_high']}")
            if rejection_stats["is_retweet"] > 0:
                logger.info(f"  ✗ Is retweet: {rejection_stats['is_retweet']}")
            if rejection_stats["is_quote"] > 0:
                logger.info(f"  ✗ Is quote: {rejection_stats['is_quote']}")
        
        return filtered
    
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
        
        # Track rejection reasons for debugging
        rejection_stats = {
            "total": len(replies),
            "no_date": 0,
            "date_parse_error": 0,
            "engagement_low": 0,
            "engagement_high": 0,
            "followers_low": 0,
            "followers_high": 0,
            "timing_too_fast": 0,
            "timing_too_slow": 0,
            "length_invalid": 0,
            "has_media": 0,
            "has_urls": 0,
            "passed": 0,
        }
        
        # Parse original tweet timestamp (skip if invalid)
        try:
            if not original_tweet.get("created_at"):
                logger.warning(f"Original tweet {original_tweet.get('id')} missing created_at, skipping replies")
                return []
            original_time = self._parse_twitter_date(original_tweet["created_at"])
            if original_time is None:
                logger.warning(f"Could not parse date for tweet {original_tweet.get('id')}")
                return []
        except Exception as e:
            logger.warning(f"Error parsing date for tweet {original_tweet.get('id')}: {e}")
            return []
        
        # Log original tweet details
        logger.info("=" * 80)
        logger.info("📝 ORIGINAL TWEET:")
        logger.info(f"   Posted: {original_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"   Likes: {original_tweet.get('favorite_count', 0)} | RTs: {original_tweet.get('retweet_count', 0)}")
        tweet_text = original_tweet.get('full_text', '')[:150]
        logger.info(f"   Text: {tweet_text}{'...' if len(original_tweet.get('full_text', '')) > 150 else ''}")
        logger.info(f"\n📬 Processing {len(replies)} replies...")
        logger.info("=" * 80)
        
        for idx, reply in enumerate(replies, 1):
            # Get basic reply info for logging
            full_text = reply.get("full_text", "")
            reply_text = full_text[:100]
            likes = reply.get("favorite_count", 0)
            retweets = reply.get("retweet_count", 0)
            followers = reply.get("author", {}).get("followers_count", 0)
            
            # Track rejection reason
            rejection_reason = None
            
            # Engagement filter
            if likes < filters["min_likes"]:
                rejection_stats["engagement_low"] += 1
                rejection_reason = f"❌ Likes too low ({likes} < {filters['min_likes']})"
            elif likes > filters["max_likes"]:
                rejection_stats["engagement_high"] += 1
                rejection_reason = f"❌ Likes too high ({likes} > {filters['max_likes']})"
            elif not (filters["min_retweets"] <= retweets <= filters["max_retweets"]):
                rejection_stats["engagement_low"] += 1
                rejection_reason = f"❌ RTs out of range ({retweets} not in [{filters['min_retweets']}, {filters['max_retweets']}])"
            
            # Author filters (avoid celebrity effect)
            elif followers < filters["min_follower_count"]:
                rejection_stats["followers_low"] += 1
                rejection_reason = f"❌ Followers too low ({followers} < {filters['min_follower_count']})"
            elif followers > filters["max_follower_count"]:
                rejection_stats["followers_high"] += 1
                rejection_reason = f"❌ Followers too high ({followers} > {filters['max_follower_count']})"
            
            # Timing filter (from gameplan.md: avoid first 5 min advantage)
            else:
                try:
                    if not reply.get("created_at"):
                        rejection_stats["no_date"] += 1
                        rejection_reason = "❌ Missing timestamp"
                    else:
                        reply_time = self._parse_twitter_date(reply["created_at"])
                        if reply_time is None:
                            rejection_stats["date_parse_error"] += 1
                            rejection_reason = "❌ Date parse error"
                        else:
                            time_diff = (reply_time - original_time).total_seconds()
                            time_diff_hours = time_diff / 3600
                            
                            if time_diff < filters["min_time_delay_seconds"]:
                                rejection_stats["timing_too_fast"] += 1
                                rejection_reason = f"❌ Replied too fast ({time_diff/60:.1f}m < {filters['min_time_delay_seconds']/60}m)"
                            elif time_diff > filters["max_time_delay_seconds"]:
                                rejection_stats["timing_too_slow"] += 1
                                rejection_reason = f"❌ Replied too slow ({time_diff_hours:.1f}h > {filters['max_time_delay_seconds']/3600}h)"
                            else:
                                # Content filters
                                if not (filters["min_length"] <= len(full_text) <= filters["max_length"]):
                                    rejection_stats["length_invalid"] += 1
                                    rejection_reason = f"❌ Length invalid ({len(full_text)} not in [{filters['min_length']}, {filters['max_length']}])"
                                elif filters["has_media"] is False and reply.get("has_media", False):
                                    rejection_stats["has_media"] += 1
                                    rejection_reason = "❌ Has media"
                                elif filters["has_urls"] is False and ("http" in full_text.lower()):
                                    rejection_stats["has_urls"] += 1
                                    rejection_reason = "❌ Has URLs"
                                else:
                                    # Passed all filters!
                                    rejection_stats["passed"] += 1
                                    reply["time_diff_seconds"] = time_diff
                                    filtered.append(reply)
                                    
                                    logger.info(f"\n  ✅ Reply #{idx} ACCEPTED:")
                                    logger.info(f"     Posted: {reply_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({time_diff_hours:.1f}h after)")
                                    logger.info(f"     Engagement: {likes} likes, {retweets} RTs | Author: {followers:,} followers")
                                    logger.info(f"     Text: {reply_text}{'...' if len(reply.get('full_text', '')) > 100 else ''}")
                                    continue
                                    
                except Exception as e:
                    rejection_stats["date_parse_error"] += 1
                    rejection_reason = f"❌ Date parsing error: {e}"
            
            # Log rejection (only show first 10 rejections to avoid spam)
            if rejection_reason and idx <= 10:
                logger.info(f"\n  Reply #{idx}: {rejection_reason}")
                logger.info(f"     Likes: {likes}, RTs: {retweets}, Followers: {followers:,}")
                logger.info(f"     Text: {reply_text}{'...' if len(reply.get('full_text', '')) > 100 else ''}")
        
        # Log rejection statistics
        if rejection_stats["total"] > 0:
            logger.info(f"Filter statistics for {rejection_stats['total']} replies:")
            logger.info(f"  ✓ Passed all filters: {rejection_stats['passed']}")
            if rejection_stats["engagement_low"] > 0:
                logger.info(f"  ✗ Engagement too low: {rejection_stats['engagement_low']}")
            if rejection_stats["engagement_high"] > 0:
                logger.info(f"  ✗ Engagement too high: {rejection_stats['engagement_high']}")
            if rejection_stats["followers_low"] > 0:
                logger.info(f"  ✗ Followers too low (<{filters['min_follower_count']}): {rejection_stats['followers_low']}")
            if rejection_stats["followers_high"] > 0:
                logger.info(f"  ✗ Followers too high (>{filters['max_follower_count']}): {rejection_stats['followers_high']}")
            if rejection_stats["timing_too_fast"] > 0:
                logger.info(f"  ✗ Replied too fast (<{filters['min_time_delay_seconds']}s): {rejection_stats['timing_too_fast']}")
            if rejection_stats["timing_too_slow"] > 0:
                logger.info(f"  ✗ Replied too slow (>{filters['max_time_delay_seconds']}s): {rejection_stats['timing_too_slow']}")
            if rejection_stats["length_invalid"] > 0:
                logger.info(f"  ✗ Length invalid (not {filters['min_length']}-{filters['max_length']} chars): {rejection_stats['length_invalid']}")
            if rejection_stats["has_media"] > 0:
                logger.info(f"  ✗ Has media: {rejection_stats['has_media']}")
            if rejection_stats["has_urls"] > 0:
                logger.info(f"  ✗ Has URLs: {rejection_stats['has_urls']}")
            if rejection_stats["no_date"] > 0:
                logger.info(f"  ✗ Missing date: {rejection_stats['no_date']}")
            if rejection_stats["date_parse_error"] > 0:
                logger.info(f"  ✗ Date parse error: {rejection_stats['date_parse_error']}")
        
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
    
    def _parse_twitter_date(self, date_string: str) -> Optional[datetime]:
        """
        Parse Twitter date in multiple formats
        
        Twitter uses: 'Thu Oct 02 07:11:55 +0000 2025' (RFC 2822)
        ISO format: '2025-10-02T07:11:55+00:00'
        """
        if not date_string:
            return None
        
        try:
            # Try Twitter's native format first (RFC 2822)
            # Example: 'Thu Oct 02 07:11:55 +0000 2025'
            return parsedate_to_datetime(date_string)
        except (ValueError, TypeError):
            pass
        
        try:
            # Try ISO format
            return datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
        
        # Could not parse
        return None
    
    def _normalize_tweet(self, raw_tweet: Dict) -> Dict:
        """
        Normalize Apify tweet format to consistent structure
        
        Different Apify actors may return different schemas
        """
        # Handle various date field names Apify might return
        created_at = (
            raw_tweet.get("created_at") or 
            raw_tweet.get("createdAt") or 
            raw_tweet.get("timestamp") or
            ""
        )
        
        # Extract author info - try multiple field structures
        user_obj = raw_tweet.get("user") or raw_tweet.get("author") or {}
        followers_count = (
            user_obj.get("followers") or        # Apify field (CORRECT!)
            user_obj.get("followers_count") or  # Twitter API format
            user_obj.get("followersCount") or   # Alternative camelCase
            raw_tweet.get("followersCount") or  # Top-level field
            0
        )
        
        # Extract author identifier - try many possible fields
        author_id_str = (
            user_obj.get("id_str") or           # Twitter API format
            user_obj.get("id") or               # Numeric ID
            raw_tweet.get("authorId") or        # Apify might use this
            raw_tweet.get("author_id") or       # Alternative format
            ""
        )
        
        # Extract screen_name/username - try all possible fields
        screen_name = (
            user_obj.get("screen_name") or
            user_obj.get("username") or
            user_obj.get("name") or
            raw_tweet.get("username") or
            raw_tweet.get("author_name") or
            ""
        )
        
        # Debug logging for first few tweets to understand structure
        if not hasattr(self, '_logged_author_debug'):
            logger.info(f"📊 DEBUG: Analyzing Apify data structure...")
            logger.info(f"   Top-level keys: {list(raw_tweet.keys())[:15]}")  # First 15 keys
            if user_obj:
                logger.info(f"   User object keys: {list(user_obj.keys())[:10]}")
                logger.info(f"   Extracted - screen_name: '{screen_name}', id: '{author_id_str}', followers: {followers_count}")
            self._logged_author_debug = True
        
        return {
            "id": raw_tweet.get("id") or raw_tweet.get("id_str") or raw_tweet.get("tweetId"),
            "full_text": raw_tweet.get("full_text") or raw_tweet.get("text", ""),
            "created_at": created_at,
            "favorite_count": raw_tweet.get("favorite_count") or raw_tweet.get("likeCount", 0),
            "retweet_count": raw_tweet.get("retweet_count") or raw_tweet.get("retweetCount", 0),
            "reply_count": raw_tweet.get("reply_count") or raw_tweet.get("replyCount", 0),
            "has_media": bool(raw_tweet.get("entities", {}).get("media") or raw_tweet.get("media")),
            "author": {
                "id": str(author_id_str) if author_id_str else "",  # Author ID for diversity tracking
                "screen_name": screen_name,                          # Username (may be empty)
                "followers_count": followers_count,
                "verified": user_obj.get("verified") or user_obj.get("isVerified") or raw_tweet.get("isVerified", False),
            },
            "raw": raw_tweet,  # Keep original for debugging
        }
    
    def _save_checkpoint(self, data: List[Dict], query_idx: int = 0, processed_queries: List[str] = None):
        """Save collection checkpoint with actual data INCLUDING global author diversity tracking"""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "pairs_collected": len(data),
            "current_query_idx": query_idx,
            "processed_queries": processed_queries or [],
            "author_counts": self.author_counts,  # CRITICAL: Save global author tracking
            "unique_authors": len(self.author_counts),
        }
        
        # Save checkpoint metadata
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save actual data to checkpoint file
        checkpoint_data_file = self.checkpoint_file.parent / "checkpoint_data.jsonl"
        with open(checkpoint_data_file, "w") as f:
            for pair in data:
                f.write(json.dumps(pair) + "\n")
        
        logger.info(f"Checkpoint saved: {len(data)} pairs (query {query_idx}), {len(self.author_counts)} unique authors globally")
    
    def _load_checkpoint(self) -> tuple:
        """Load checkpoint if exists INCLUDING global author diversity tracking
        
        Returns:
            (existing_data, query_idx, processed_queries)
        """
        if not self.checkpoint_file.exists():
            return [], 0, []
        
        try:
            # Load checkpoint metadata
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            # CRITICAL: Restore global author diversity tracking
            self.author_counts = checkpoint.get("author_counts", {})
            
            # Load actual data
            checkpoint_data_file = self.checkpoint_file.parent / "checkpoint_data.jsonl"
            existing_data = []
            if checkpoint_data_file.exists():
                with open(checkpoint_data_file, "r") as f:
                    for line in f:
                        if line.strip():
                            existing_data.append(json.loads(line))
            
            query_idx = checkpoint.get("current_query_idx", 0)
            processed_queries = checkpoint.get("processed_queries", [])
            
            logger.info(f"Checkpoint loaded: {len(existing_data)} pairs, {len(self.author_counts)} unique authors, resuming from query {query_idx}")
            return existing_data, query_idx, processed_queries
        
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return [], 0, []
    
    def _clear_checkpoint(self):
        """Clear checkpoint files after successful completion"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            checkpoint_data_file = self.checkpoint_file.parent / "checkpoint_data.jsonl"
            if checkpoint_data_file.exists():
                checkpoint_data_file.unlink()
            logger.info("Checkpoint files cleared")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")
    
    def reset_author_tracking(self):
        """Reset global author diversity tracking (use with caution!)"""
        old_count = len(self.author_counts)
        self.author_counts = {}
        logger.warning(f"Author tracking reset! Cleared {old_count} authors from tracking.")
        return old_count
    
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

