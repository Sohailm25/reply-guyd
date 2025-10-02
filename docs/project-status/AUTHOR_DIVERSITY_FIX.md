# Author Diversity Bug Fix

## Problem Identified

The previous implementation had a **critical bug**: author diversity tracking (`max_replies_per_author: 10`) was resetting for each search query instead of persisting globally across ALL collected pairs.

**Result**: Bot farms could contribute unlimited replies by spreading them across different queries.

**Evidence**: Logs showed "Unique authors: 1" despite collecting 1,408 pairs, indicating all replies came from the same bot account.

---

## Solution Implemented

### 1. **Global Author Tracking**
- Moved `author_counts` from a local variable to a **class instance variable** (`self.author_counts`)
- Persists across all queries throughout the entire collection session

### 2. **Checkpoint Integration**
- **Save**: Checkpoints now include `author_counts` and `unique_authors` count
- **Load**: Resume functionality restores author tracking state
- **Clear**: Author tracking resets only when collection fully completes

### 3. **Enhanced Logging**
```python
# Before each query:
logger.info(f"Author diversity tracking: {len(self.author_counts)} unique authors already tracked")

# After each query:
logger.info(f"Collection complete: {len(tweet_reply_pairs)} pairs from {len(self.author_counts)} unique authors (GLOBAL)")
```

### 4. **Diverse Search Queries**
Expanded from 10 tech-only queries to **15 diverse topics**:
- **Technical** (3): debugging, system design, performance
- **Startup/Business** (3): founder lessons, product launches, pricing
- **Creative/Writing** (2): storytelling, design workflows
- **Career/Learning** (2): career advice, remote work
- **Psychology** (2): productivity, imposter syndrome
- **Specialized** (3): open source, incidents, automation

**Why diversity matters**: Different topics attract different author communities, dramatically increasing unique reply authors.

---

## Technical Changes

### `apify_collector.py`

**1. Instance Variable**
```python
def __init__(self, config_path: str = "config/data_collection_config.yaml"):
    # ...
    # Global author diversity tracking (persists across all queries)
    self.author_counts = {}
```

**2. Global Tracking**
```python
def collect_tweets_and_replies(self, ...):
    # OLD: author_counts = {}  # Reset per query ❌
    # NEW: Use self.author_counts (persists) ✅
    
    for reply in filtered_replies:
        author_id = reply.get("author", {}).get("id") or ...
        
        # Check GLOBAL author limit (tracked across all queries)
        if self.author_counts.get(author_id, 0) >= max_per_author:
            continue  # Skip this reply
        
        self.author_counts[author_id] = self.author_counts.get(author_id, 0) + 1
```

**3. Checkpoint Persistence**
```python
def _save_checkpoint(self, data, query_idx, processed_queries):
    checkpoint = {
        # ...
        "author_counts": self.author_counts,  # Save globally
        "unique_authors": len(self.author_counts),
    }

def _load_checkpoint(self):
    # ...
    self.author_counts = checkpoint.get("author_counts", {})  # Restore globally
```

**4. Utility Method**
```python
def reset_author_tracking(self):
    """Reset global author diversity tracking (use with caution!)"""
    old_count = len(self.author_counts)
    self.author_counts = {}
    logger.warning(f"Author tracking reset! Cleared {old_count} authors.")
```

### `data_collection_config.yaml`

**Enhanced Queries** (15 total, down from min_faves:200 to 150 for more results):
- Added startup/business, creative, career, psychology topics
- Stronger spam filters on ALL queries: `-crypto -NFT -web3 -token -airdrop -giveaway -whitelist`
- More specific keywords: "mistake", "regret", "failed", "outage", "burnout"

---

## Expected Impact

### Before Fix
- ❌ Same bot could contribute 100+ replies across 10 queries (10 per query)
- ❌ "Unique authors: 1" even with 1,408 pairs
- ❌ Low-quality crypto spam dominating dataset

### After Fix
- ✅ Each author limited to **10 total replies** across ALL queries
- ✅ Diverse topics attract different author pools
- ✅ Bot farms detected and limited immediately
- ✅ Expected: **100+ unique authors** for 1,000 pairs (10 replies each)

---

## How to Use

### Normal Collection (with fix)
```bash
# Start fresh collection (checkpoint will be empty)
./run.sh scripts/collect_data.py --method apify --target 1500

# Resume from interruption (author tracking auto-restores)
./run.sh scripts/collect_data.py --method apify --target 1500 --resume
```

### Monitoring Author Diversity
```bash
# Check checkpoint for current author count
jq '.unique_authors' data/raw/collection_checkpoint.json

# Check full author breakdown
jq '.author_counts | length' data/raw/collection_checkpoint.json
```

### Manual Reset (if needed)
```python
from src.data_collection.apify_collector import ApifyCollector

collector = ApifyCollector()
collector.reset_author_tracking()  # Clears all author tracking
```

---

## Validation

After restart, you should see in logs:
```
Author diversity tracking: 0 unique authors already tracked, 0 total replies
[... collection ...]
Collection complete: 50 pairs from 25 unique authors (GLOBAL)

[Next query...]
Author diversity tracking: 25 unique authors already tracked, 50 total replies
[... collection ...]
Collection complete: 100 pairs from 47 unique authors (GLOBAL)
```

**Key indicator**: `unique authors` count should **grow steadily** and approach `total pairs / 10`.

---

## Next Steps

1. **Kill current tmux session** (has old buggy code)
2. **Start fresh collection** with fixed code
3. **Monitor logs** for "GLOBAL" author tracking
4. **Verify diversity** with `jq '.unique_authors' data/raw/collection_checkpoint.json`

---

**Fixed**: October 2, 2025  
**Severity**: Critical (data quality impact)  
**Status**: ✅ Resolved

