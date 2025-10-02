#!/usr/bin/env python3
"""
Main Data Collection Script
Orchestrates the complete data collection pipeline

Usage:
    python scripts/collect_data.py --method apify --target 1000
    python scripts/collect_data.py --method scraper --target 800
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.apify_collector import ApifyCollector
from src.data_collection.scraper_collector import TwscrapeCollector
from src.data_collection.data_validator import DataValidator
from src.data_collection.data_cleaner import DataCleaner
import yaml


def setup_logging(log_dir: str = "output/logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/data_collection_config.yaml"):
    """Load configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_with_apify(config: dict, target_pairs: int, logger, resume: bool = False) -> list:
    """Collect data using Apify with checkpoint/resume support"""
    logger.info("=" * 60)
    logger.info("COLLECTION METHOD: Apify")
    logger.info("=" * 60)
    
    collector = ApifyCollector()
    
    # Get search queries from config
    search_queries = config["collection"]["search_queries"]
    
    # Calculate tweets per query
    tweets_per_query = target_pairs // len(search_queries) + 1
    
    # Try to resume from checkpoint
    all_pairs = []
    start_query_idx = 0
    processed_queries = []
    
    if resume:
        all_pairs, start_query_idx, processed_queries = collector._load_checkpoint()
        if all_pairs:
            logger.info(f"🔄 RESUMING from checkpoint:")
            logger.info(f"   Existing pairs: {len(all_pairs)}")
            logger.info(f"   Resuming from query {start_query_idx + 1}/{len(search_queries)}")
            logger.info(f"   Processed queries: {len(processed_queries)}")
    
    for idx, query in enumerate(search_queries):
        # Skip already processed queries
        if idx < start_query_idx:
            logger.info(f"\n--- Skipping Query {idx+1}/{len(search_queries)} (already processed) ---")
            continue
        
        logger.info(f"\n--- Query {idx+1}/{len(search_queries)} ---")
        logger.info(f"Query: {query}")
        
        try:
            pairs = collector.collect_tweets_and_replies(
                search_query=query,
                max_tweets=tweets_per_query,
                max_replies_per_tweet=config["collection"]["reply_filters"]["max_likes"]
            )
            
            all_pairs.extend(pairs)
            processed_queries.append(query)
            
            logger.info(f"Collected {len(pairs)} pairs from this query")
            logger.info(f"Total so far: {len(all_pairs)} pairs")
            
            # Save checkpoint after each query
            if collector.checkpoint_enabled:
                collector._save_checkpoint(all_pairs, idx, processed_queries)
            
            # Check if target reached
            if len(all_pairs) >= target_pairs:
                logger.info(f"Target reached: {len(all_pairs)} >= {target_pairs}")
                break
                
        except Exception as e:
            logger.error(f"Error with query '{query}': {e}")
            logger.error(f"Saving checkpoint before continuing...")
            if collector.checkpoint_enabled:
                collector._save_checkpoint(all_pairs, idx, processed_queries)
            continue
    
    # Clear checkpoint on successful completion
    if len(all_pairs) >= target_pairs or idx == len(search_queries) - 1:
        logger.info("✅ Collection complete - clearing checkpoint")
        collector._clear_checkpoint()
    
    return all_pairs


def collect_with_scraper(config: dict, target_pairs: int, logger) -> list:
    """Collect data using web scraping (twscrape)"""
    logger.info("=" * 60)
    logger.info("COLLECTION METHOD: Web Scraping (twscrape)")
    logger.info("=" * 60)
    
    logger.warning("Twscrape requires Twitter accounts to be configured")
    logger.warning("See .env.example for required environment variables")
    
    # Check for required env vars
    required_vars = [
        "TWITTER_ACCOUNT_1_USERNAME",
        "TWITTER_ACCOUNT_1_PASSWORD",
        "TWITTER_ACCOUNT_1_EMAIL",
        "TWITTER_ACCOUNT_1_EMAIL_PASSWORD",
    ]
    
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Cannot proceed with scraping. Use Apify instead or configure accounts.")
        return []
    
    # Implementation would use TwscrapeCollector here
    logger.error("Twscrape collection not fully implemented in this template")
    logger.error("Recommended: Use Apify method instead")
    
    return []


def validate_data(pairs: list, config: dict, logger) -> list:
    """Validate collected data"""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    
    validator = DataValidator(config["collection"]["quality"])
    
    results = validator.validate_batch(pairs)
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  Total: {results['stats']['total']}")
    logger.info(f"  Valid: {results['stats']['valid']} ({results['stats']['valid_percentage']:.1f}%)")
    logger.info(f"  Invalid: {results['stats']['invalid']}")
    
    if results['stats']['rejection_reasons']:
        logger.info(f"\nRejection Reasons:")
        for reason, count in sorted(
            results['stats']['rejection_reasons'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            logger.info(f"  - {reason}: {count}")
    
    return results['valid_pairs']


def clean_data(pairs: list, config: dict, logger) -> list:
    """Clean and deduplicate data"""
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING & DEDUPLICATION")
    logger.info("=" * 60)
    
    cleaner = DataCleaner(config["collection"]["quality"])
    
    cleaned_pairs = cleaner.clean_batch(pairs)
    
    # Analyze cleaned dataset
    stats = cleaner.analyze_dataset(cleaned_pairs)
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total pairs: {stats['total_pairs']}")
    logger.info(f"  Unique tweets: {stats['topics']['unique_tweets']}")
    logger.info(f"  Unique authors: {stats['authors']['unique_count']}")
    logger.info(f"  Reply length: {stats['reply_length']['mean']:.1f} chars (avg)")
    logger.info(f"  Reply engagement: {stats['engagement']['mean_likes']:.1f} likes (avg)")
    
    return cleaned_pairs


def save_final_data(pairs: list, output_dir: str, logger):
    """Save final processed data"""
    logger.info("\n" + "=" * 60)
    logger.info("SAVING DATA")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl_file = output_path / f"training_data_{timestamp}.jsonl"
    
    with open(jsonl_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    logger.info(f"Saved {len(pairs)} pairs to {jsonl_file}")
    
    # Save summary statistics
    stats_file = output_path / f"collection_stats_{timestamp}.json"
    stats = {
        "collection_timestamp": timestamp,
        "total_pairs": len(pairs),
        "unique_tweets": len(set(p["tweet_id"] for p in pairs)),
        "unique_authors": len(set(p["reply_author"] for p in pairs)),
        "engagement": {
            "mean_likes": float(sum(p["reply_likes"] for p in pairs) / len(pairs)),
            "max_likes": max(p["reply_likes"] for p in pairs),
            "min_likes": min(p["reply_likes"] for p in pairs),
        },
        "sample_pairs": pairs[:5] if len(pairs) >= 5 else pairs,  # First 5 as examples
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_file}")
    
    return jsonl_file, stats_file


def main():
    """Main data collection pipeline"""
    parser = argparse.ArgumentParser(description="Collect Twitter data for LoRA training")
    parser.add_argument(
        "--method",
        choices=["apify", "scraper"],
        default="apify",
        help="Collection method (default: apify)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target number of training pairs (default: 1000)"
    )
    parser.add_argument(
        "--config",
        default="config/data_collection_config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip cleaning/deduplication step"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists (automatically loads progress)"
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("TWITTER DATA COLLECTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Target pairs: {args.target}")
    logger.info(f"Output: {args.output_dir}")
    
    # Step 1: Collect raw data
    if args.method == "apify":
        raw_pairs = collect_with_apify(config, args.target, logger, resume=args.resume)
    else:
        raw_pairs = collect_with_scraper(config, args.target, logger)
    
    if not raw_pairs:
        logger.error("No data collected. Exiting.")
        return 1
    
    logger.info(f"\nRaw collection complete: {len(raw_pairs)} pairs")
    
    # Step 2: Validate
    if not args.skip_validation:
        valid_pairs = validate_data(raw_pairs, config, logger)
    else:
        logger.info("Skipping validation (--skip-validation)")
        valid_pairs = raw_pairs
    
    if not valid_pairs:
        logger.error("No valid pairs after validation. Exiting.")
        return 1
    
    # Step 3: Clean and deduplicate
    if not args.skip_cleaning:
        final_pairs = clean_data(valid_pairs, config, logger)
    else:
        logger.info("Skipping cleaning (--skip-cleaning)")
        final_pairs = valid_pairs
    
    if not final_pairs:
        logger.error("No pairs after cleaning. Exiting.")
        return 1
    
    # Step 4: Save final data
    jsonl_file, stats_file = save_final_data(final_pairs, args.output_dir, logger)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final dataset: {len(final_pairs)} training pairs")
    logger.info(f"Data file: {jsonl_file}")
    logger.info(f"Stats file: {stats_file}")
    
    # Quality check
    if len(final_pairs) < config["collection"]["min_pairs"]:
        logger.warning(f"⚠️  Dataset below minimum ({len(final_pairs)} < {config['collection']['min_pairs']})")
        logger.warning("⚠️  Consider collecting more data")
    elif len(final_pairs) > config["collection"]["max_pairs"]:
        logger.info(f"✓ Dataset exceeds target! ({len(final_pairs)} > {config['collection']['target_pairs']})")
    else:
        logger.info(f"✓ Dataset within target range")
    
    logger.info("\n⚠️  IMPORTANT: Manual review recommended!")
    logger.info("Quality is critical for small datasets. Review samples:")
    logger.info(f"  head -n 10 {jsonl_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

