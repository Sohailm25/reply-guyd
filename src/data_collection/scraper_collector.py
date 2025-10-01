"""
Web Scraping-based Twitter Data Collector
Backup/alternative to Apify using direct scraping

Based on current_research.md: twscrape as recommended tool
Warning: Violates Twitter ToS but legally defensible for public data
"""

import os
import time
import json
import logging
import asyncio
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ScraperCollector:
    """
    Direct web scraping collector (fallback if Apify budget exceeded)
    
    Note: This is a simplified implementation. For production use:
    - Consider using twscrape library (requires Twitter accounts)
    - Use residential proxies to avoid blocking
    - Implement robust retry logic
    - Respect rate limits
    
    Legal note: Scraping public data is legally defensible (X Corp. v. Bright Data)
    but violates Twitter ToS. Use responsibly and ethically.
    """
    
    def __init__(self, config_path: str = "config/data_collection_config.yaml"):
        """Initialize scraper with configuration"""
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Setup HTTP client with realistic headers
        self.client = httpx.AsyncClient(
            timeout=self.config["scraping"]["timeout"],
            follow_redirects=True,
            headers=self._get_random_headers(),
        )
        
        # Storage
        self.raw_data_dir = Path(self.config["storage"]["raw_data_dir"])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.rate_limit_delay = self.config["scraping"]["rate_limit_delay"]
        self.max_retries = self.config["scraping"]["max_retries"]
        
        logger.info("Scraper collector initialized")
        logger.warning("⚠️  Web scraping violates Twitter ToS. Use responsibly.")
    
    def _get_random_headers(self) -> Dict:
        """Generate realistic browser headers to avoid detection"""
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ]
        
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def collect_tweets_and_replies(
        self,
        search_query: str,
        max_tweets: int = 100,
        max_replies_per_tweet: int = 50
    ) -> List[Dict]:
        """
        Collect tweets and replies via web scraping
        
        NOTE: This is a PLACEHOLDER implementation showing the structure.
        For actual scraping, you should use:
        
        1. twscrape library (recommended from current_research.md):
           - Requires 3-10 Twitter accounts
           - Handles rate limiting via account rotation
           - More reliable than custom scraping
        
        2. Commercial API like TwitterAPI.io or Bright Data:
           - Better reliability
           - Legal compliance handled by vendor
           - ~$200-500/month for adequate volume
        
        Args:
            search_query: Twitter search query
            max_tweets: Maximum tweets to collect
            max_replies_per_tweet: Maximum replies per tweet
        
        Returns:
            List of tweet-reply pairs
        """
        logger.warning("⚠️  Direct scraping is not fully implemented in this template")
        logger.warning("⚠️  Recommended: Use Apify or twscrape library instead")
        
        # Placeholder return
        return []
    
    async def _scrape_search_results(self, query: str, max_results: int) -> List[Dict]:
        """
        Scrape Twitter search results
        
        PLACEHOLDER: Real implementation would:
        1. Make request to Twitter search
        2. Parse HTML/JSON response
        3. Extract tweet data
        4. Handle pagination
        """
        logger.info(f"Would scrape search: {query} (max: {max_results})")
        
        # In reality, you'd make requests like:
        # url = f"https://twitter.com/search?q={encoded_query}&f=live"
        # response = await self.client.get(url)
        # tweets = self._parse_search_html(response.text)
        
        return []
    
    async def _scrape_tweet_replies(self, tweet_id: str, max_replies: int) -> List[Dict]:
        """
        Scrape replies to a specific tweet
        
        PLACEHOLDER: Real implementation would access tweet conversation
        """
        logger.info(f"Would scrape replies for tweet: {tweet_id}")
        
        # In reality:
        # url = f"https://twitter.com/i/status/{tweet_id}"
        # response = await self.client.get(url)
        # replies = self._parse_replies_html(response.text)
        
        return []
    
    def save_to_file(self, data: List[Dict], filename: str = None):
        """Save collected data to file"""
        if filename is None:
            filename = f"scraper_collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath = self.raw_data_dir / filename
        
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Saved {len(data)} pairs to {filepath}")
        return filepath
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class TwscrapeCollector:
    """
    Wrapper for twscrape library (recommended approach from current_research.md)
    
    Advantages:
    - Account rotation built-in
    - Better rate limit handling
    - More reliable than custom scraping
    
    Requirements:
    - 3-10 Twitter accounts (purchase for $50-100 each)
    - Residential proxies ($50-500/month)
    
    Installation:
    pip install twscrape
    """
    
    def __init__(self, config_path: str = "config/data_collection_config.yaml"):
        """Initialize twscrape collector"""
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Check if twscrape is available
        try:
            import twscrape
            self.twscrape = twscrape
            logger.info("twscrape library available")
        except ImportError:
            logger.error("twscrape not installed. Install with: pip install twscrape")
            logger.error("See: https://github.com/vladkens/twscrape")
            raise
        
        self.api = None
        self.raw_data_dir = Path(self.config["storage"]["raw_data_dir"])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup_accounts(self, accounts: List[Dict]):
        """
        Setup Twitter accounts for scraping
        
        Args:
            accounts: List of dicts with keys: username, password, email, email_password
        
        Example:
            accounts = [
                {
                    "username": "account1",
                    "password": "pass1",
                    "email": "email1@example.com",
                    "email_password": "emailpass1"
                },
                # ... more accounts for rotation
            ]
        """
        from twscrape import API, AccountsPool
        
        # Initialize API with accounts pool
        self.api = API()
        
        # Add accounts
        for acc in accounts:
            await self.api.pool.add_account(
                username=acc["username"],
                password=acc["password"],
                email=acc["email"],
                email_password=acc["email_password"],
            )
        
        # Login all accounts
        await self.api.pool.login_all()
        
        logger.info(f"Setup {len(accounts)} Twitter accounts for scraping")
    
    async def collect_tweets_and_replies(
        self,
        search_query: str,
        max_tweets: int = 100,
        max_replies_per_tweet: int = 50
    ) -> List[Dict]:
        """
        Collect tweets and replies using twscrape
        
        This is the RECOMMENDED approach from current_research.md
        """
        if self.api is None:
            raise ValueError("Must call setup_accounts() first")
        
        tweet_reply_pairs = []
        
        # Search for tweets
        tweet_count = 0
        async for tweet in self.api.search(search_query, limit=max_tweets):
            tweet_count += 1
            logger.info(f"Processing tweet {tweet_count}/{max_tweets}: {tweet.id}")
            
            # Filter by engagement
            if not self._passes_tweet_filters(tweet):
                continue
            
            # Get replies (twscrape doesn't directly support conversation search)
            # You'd need to use search with conversation_id
            conversation_query = f"conversation_id:{tweet.id}"
            
            replies = []
            async for reply in self.api.search(conversation_query, limit=max_replies_per_tweet):
                if reply.id != tweet.id:  # Exclude original tweet
                    replies.append(reply)
            
            # Filter replies
            filtered_replies = [r for r in replies if self._passes_reply_filters(r, tweet)]
            
            # Create training pairs
            for reply in filtered_replies:
                pair = self._create_training_pair(tweet, reply)
                tweet_reply_pairs.append(pair)
            
            logger.info(f"Found {len(filtered_replies)} high-quality replies")
            
            # Rate limiting
            await asyncio.sleep(self.config["scraping"]["rate_limit_delay"])
        
        logger.info(f"Collection complete: {len(tweet_reply_pairs)} pairs")
        return tweet_reply_pairs
    
    def _passes_tweet_filters(self, tweet) -> bool:
        """Check if tweet passes collection filters"""
        filters = self.config["collection"]["tweet_filters"]
        
        likes = tweet.likeCount or 0
        retweets = tweet.retweetCount or 0
        
        if not (filters["min_likes"] <= likes <= filters["max_likes"]):
            return False
        
        if not (filters["min_retweets"] <= retweets <= filters["max_retweets"]):
            return False
        
        return True
    
    def _passes_reply_filters(self, reply, original_tweet) -> bool:
        """Check if reply passes quality filters"""
        filters = self.config["collection"]["reply_filters"]
        
        # Engagement
        likes = reply.likeCount or 0
        if not (filters["min_likes"] <= likes <= filters["max_likes"]):
            return False
        
        # Author followers
        followers = reply.user.followersCount or 0
        if not (filters["min_follower_count"] <= followers <= filters["max_follower_count"]):
            return False
        
        # Timing
        time_diff = (reply.date - original_tweet.date).total_seconds()
        if not (filters["min_time_delay_seconds"] <= time_diff <= filters["max_time_delay_seconds"]):
            return False
        
        # Content
        text = reply.rawContent or ""
        if not (filters["min_length"] <= len(text) <= filters["max_length"]):
            return False
        
        if filters["has_media"] is False and reply.media:
            return False
        
        if filters["has_urls"] is False and reply.links:
            return False
        
        return True
    
    def _create_training_pair(self, tweet, reply) -> Dict:
        """Create training pair from twscrape objects"""
        return {
            "tweet_id": str(tweet.id),
            "tweet": tweet.rawContent,
            "tweet_author": tweet.user.username,
            "tweet_likes": tweet.likeCount or 0,
            "tweet_retweets": tweet.retweetCount or 0,
            "tweet_created_at": tweet.date.isoformat(),
            
            "reply_id": str(reply.id),
            "reply": reply.rawContent,
            "reply_author": reply.user.username,
            "reply_author_followers": reply.user.followersCount or 0,
            "reply_likes": reply.likeCount or 0,
            "reply_retweets": reply.retweetCount or 0,
            "reply_created_at": reply.date.isoformat(),
            "reply_time_diff_seconds": (reply.date - tweet.date).total_seconds(),
            
            "collected_at": datetime.utcnow().isoformat(),
        }
    
    def save_to_file(self, data: List[Dict], filename: str = None):
        """Save collected data"""
        if filename is None:
            filename = f"twscrape_collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath = self.raw_data_dir / filename
        
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Saved {len(data)} pairs to {filepath}")
        return filepath


async def main_twscrape():
    """Example usage with twscrape (RECOMMENDED)"""
    logging.basicConfig(level=logging.INFO)
    
    collector = TwscrapeCollector()
    
    # Setup accounts (you need to provide real accounts)
    accounts = [
        {
            "username": os.getenv("TWITTER_ACCOUNT_1_USERNAME"),
            "password": os.getenv("TWITTER_ACCOUNT_1_PASSWORD"),
            "email": os.getenv("TWITTER_ACCOUNT_1_EMAIL"),
            "email_password": os.getenv("TWITTER_ACCOUNT_1_EMAIL_PASSWORD"),
        },
        # Add 2-9 more accounts for better rate limit handling
    ]
    
    await collector.setup_accounts(accounts)
    
    # Collect data
    search_query = "AI OR artificial intelligence lang:en min_faves:200"
    pairs = await collector.collect_tweets_and_replies(
        search_query=search_query,
        max_tweets=10,
        max_replies_per_tweet=30
    )
    
    # Save
    collector.save_to_file(pairs)
    
    print(f"\nCollected {len(pairs)} training pairs")


if __name__ == "__main__":
    # To use twscrape, uncomment:
    # asyncio.run(main_twscrape())
    
    print("⚠️  Scraping collector requires additional setup")
    print("Recommended: Use Apify (apify_collector.py) or twscrape")
    print("\nFor twscrape setup:")
    print("1. pip install twscrape")
    print("2. Acquire 3-10 Twitter accounts")
    print("3. Configure accounts in .env")
    print("4. Run: python src/data_collection/scraper_collector.py")

