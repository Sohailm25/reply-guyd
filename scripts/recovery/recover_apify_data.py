#!/usr/bin/env python3
"""
Apify Data Recovery Script

This script recovers ALL tweet-reply pairs from Apify's cloud storage.
Even though our local checkpoint files were overwritten, Apify stores
the raw dataset from each actor run in their cloud storage!

This can recover the "lost" 1,713 pairs from Queries 1-6.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from apify_client import ApifyClient
from dotenv import load_dotenv
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ApifyDataRecovery:
    """Recover data from Apify's cloud storage"""
    
    def __init__(self, config_path: str = "config/data_collection_config.yaml"):
        """Initialize Apify client"""
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Apify client
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables")
        
        self.client = ApifyClient(api_token)
        self.actor_id = self.config["apify"]["actor_id"]
        
        # Setup output directory
        self.output_dir = Path("data/recovered")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Apify recovery initialized for actor: {self.actor_id}")
    
    def list_recent_runs(self, hours_ago: int = 24) -> List[Dict]:
        """
        List all actor runs from the last N hours.
        
        Args:
            hours_ago: How many hours back to look (default 24)
        
        Returns:
            List of run metadata (id, startedAt, finishedAt, status, etc.)
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        
        logger.info(f"Fetching actor runs since {cutoff_time.isoformat()} UTC")
        
        runs = []
        try:
            # Get runs for this actor
            runs_list = self.client.actor(self.actor_id).runs().list(limit=100)
            
            # Debug: Check what we're getting
            logger.info(f"Runs list type: {type(runs_list)}")
            logger.info(f"Has items attr: {hasattr(runs_list, 'items')}")
            
            # Handle different response formats
            if hasattr(runs_list, 'items'):
                items = runs_list.items
            elif isinstance(runs_list, dict) and 'items' in runs_list:
                items = runs_list['items']
            elif isinstance(runs_list, list):
                items = runs_list
            else:
                logger.error(f"Unexpected runs_list structure: {type(runs_list)}")
                return []
            
            logger.info(f"Processing {len(items) if hasattr(items, '__len__') else 'unknown'} run items")
            
            for idx, run in enumerate(items):
                try:
                    # Handle both dict and object formats
                    if isinstance(run, dict):
                        started_at = run.get("startedAt") or run.get("started_at")
                        run_id = run.get("id")
                        status = run.get("status")
                        finished_at = run.get("finishedAt") or run.get("finished_at")
                        dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
                    else:
                        # Object format
                        started_at = getattr(run, "startedAt", None) or getattr(run, "started_at", None)
                        run_id = getattr(run, "id", None)
                        status = getattr(run, "status", None)
                        finished_at = getattr(run, "finishedAt", None) or getattr(run, "finished_at", None)
                        dataset_id = getattr(run, "defaultDatasetId", None) or getattr(run, "default_dataset_id", None)
                    
                    if not started_at:
                        logger.warning(f"Run {idx} missing startedAt field, skipping")
                        continue
                    
                    # Parse datetime
                    if isinstance(started_at, str):
                        run_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    else:
                        run_started = started_at
                    
                    # Only include runs from after cutoff time
                    if run_started >= cutoff_time:
                        runs.append({
                            "id": run_id,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "status": status,
                            "dataset_id": dataset_id,
                            "stats": run.get("stats", {}) if isinstance(run, dict) else {},
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing run {idx}: {e}")
                    continue
            
            # Sort by start time (oldest first)
            runs.sort(key=lambda x: x["started_at"])
            
            logger.info(f"Found {len(runs)} runs in the last {hours_ago} hours")
            return runs
        
        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def fetch_dataset(self, dataset_id: str) -> List[Dict]:
        """
        Fetch all items from an Apify dataset.
        
        Args:
            dataset_id: The Apify dataset ID
        
        Returns:
            List of raw tweet objects from Apify
        """
        if not dataset_id:
            logger.warning("No dataset_id provided, skipping")
            return []
        
        try:
            logger.info(f"Fetching dataset: {dataset_id}")
            
            items = []
            for item in self.client.dataset(dataset_id).iterate_items():
                items.append(item)
            
            logger.info(f"✅ Fetched {len(items)} items from dataset {dataset_id}")
            return items
        
        except Exception as e:
            logger.error(f"Error fetching dataset {dataset_id}: {e}")
            return []
    
    def recover_all_data(self, hours_ago: int = 12, start_time_cst: str = None) -> Dict:
        """
        Recover ALL data from recent Apify runs.
        
        Args:
            hours_ago: How many hours back to look
            start_time_cst: Optional start time in CST (e.g., "2025-10-02 09:10:00")
        
        Returns:
            Dict with recovered data and statistics
        """
        # List all recent runs
        runs = self.list_recent_runs(hours_ago)
        
        if not runs:
            logger.warning("No runs found in the specified time range")
            return {"runs": 0, "datasets": 0, "total_items": 0, "data": []}
        
        # Filter by start time if provided
        if start_time_cst:
            # Convert CST to UTC (CST is UTC-6, but we need to account for CDT which is UTC-5)
            # Assuming CDT (Central Daylight Time, UTC-5)
            cutoff = datetime.strptime(start_time_cst, "%Y-%m-%d %H:%M:%S")
            cutoff_utc = cutoff + timedelta(hours=5)  # Convert CDT to UTC
            
            runs = [r for r in runs if datetime.fromisoformat(r["started_at"].replace("Z", "+00:00")) >= cutoff_utc]
            logger.info(f"Filtered to {len(runs)} runs after {start_time_cst} CST")
        
        # Fetch all datasets
        all_items = []
        datasets_fetched = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RECOVERING DATA FROM {len(runs)} APIFY RUNS")
        logger.info(f"{'='*60}\n")
        
        for idx, run in enumerate(runs, 1):
            logger.info(f"Run {idx}/{len(runs)}:")
            logger.info(f"  ID: {run['id']}")
            logger.info(f"  Started: {run['started_at']}")
            logger.info(f"  Status: {run['status']}")
            
            if run["dataset_id"]:
                items = self.fetch_dataset(run["dataset_id"])
                all_items.extend(items)
                datasets_fetched += 1
                logger.info(f"  Items: {len(items)}")
            else:
                logger.info(f"  ⚠️  No dataset ID (run may have failed)")
            
            logger.info("")
        
        # Deduplicate by tweet ID
        unique_items = {}
        for item in all_items:
            tweet_id = item.get("id_str") or item.get("id")
            if tweet_id:
                unique_items[tweet_id] = item
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RECOVERY SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total runs found: {len(runs)}")
        logger.info(f"Datasets fetched: {datasets_fetched}")
        logger.info(f"Total items: {len(all_items)}")
        logger.info(f"Unique tweets: {len(unique_items)}")
        
        return {
            "runs": len(runs),
            "datasets": datasets_fetched,
            "total_items": len(all_items),
            "unique_items": len(unique_items),
            "data": list(unique_items.values()),
        }
    
    def save_recovered_data(self, data: List[Dict], filename: str = None):
        """Save recovered data to file"""
        if not filename:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"recovered_tweets_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"✅ Saved {len(data)} recovered tweets to: {filepath}")
        return filepath


def main():
    """Main recovery script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover data from Apify cloud storage")
    parser.add_argument(
        "--hours",
        type=int,
        default=12,
        help="How many hours back to search (default: 12)"
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start time in CST format: 'YYYY-MM-DD HH:MM:SS' (e.g., '2025-10-02 09:10:00')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Initialize recovery
    recovery = ApifyDataRecovery()
    
    # Recover all data
    logger.info(f"\n🔍 Starting data recovery...")
    if args.start_time:
        logger.info(f"   From: {args.start_time} CST onwards")
    else:
        logger.info(f"   From: Last {args.hours} hours")
    
    result = recovery.recover_all_data(
        hours_ago=args.hours,
        start_time_cst=args.start_time
    )
    
    # Save recovered data
    if result["data"]:
        filepath = recovery.save_recovered_data(result["data"], args.output)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ RECOVERY COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Recovered file: {filepath}")
        logger.info(f"Total unique tweets: {result['unique_items']}")
        logger.info(f"\nNext step: Process these tweets to extract reply pairs")
        logger.info(f"Command: python scripts/recovery/process_recovered_tweets.py")
    else:
        logger.warning("No data recovered. Check the time range and actor runs.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

