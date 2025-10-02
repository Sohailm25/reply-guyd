# 🔄 Collection Restart Instructions

**Date**: October 2, 2025  
**Reason**: Critical author diversity bug fixed + enhanced search queries

---

## ⚠️ What Happened

Your previous collection run had a **critical bug**:

**Problem**: Author diversity tracking (`max_replies_per_author: 10`) was resetting for each search query instead of persisting globally.

**Impact**: 
- Collected 1,408 pairs from **ONLY 1 AUTHOR** (bot farm)
- Validation correctly rejected 72.1% (980/1408) for low quality
- Final dataset: 376 pairs, all from same author ❌

**Root Cause**: `author_counts = {}` was initialized inside `collect_tweets_and_replies()`, resetting for each query.

---

## ✅ Fixes Implemented

### 1. Global Author Tracking
- `author_counts` is now a **class instance variable** (`self.author_counts`)
- Persists across **ALL queries** for the entire collection session
- Saved/loaded in checkpoints for resume functionality

### 2. Enhanced Search Queries (10 → 15 queries)
**Before**: Only tech topics (limited author pool)
```
- debugging, system design, performance
- learning, career
- product, API, database
- TypeScript, testing, open source
```

**After**: Diverse topics across 5 categories
```
Technical (3): debugging, system design, performance
Startup/Business (3): founder mistakes, product launches, pricing
Creative/Writing (2): storytelling, design workflows  
Career/Learning (2): career advice, remote work
Psychology (2): productivity, imposter syndrome
Specialized (3): open source burnout, incidents, automation
```

**Why**: Different topics attract different author communities → more diverse reply authors.

### 3. Better Spam Filters
- Lowered `min_faves: 200 → 150` (more results, still high quality)
- Added stronger exclusions: `-crypto -NFT -web3 -token -airdrop -giveaway -whitelist`
- More specific keywords: "mistake", "regret", "failed", "outage", "burnout"

---

## 🚀 How to Restart

### Step 1: Clean Up Old Data (Optional)
```bash
# Remove contaminated data from bot farm
rm -f data/processed/training_data_20251002_*.jsonl
rm -f data/processed/collection_stats_20251002_*.json

# Clear checkpoint (or keep it - the bug is fixed either way)
rm -f data/raw/collection_checkpoint.json
rm -f data/raw/checkpoint_data.jsonl
```

### Step 2: Start Fresh Collection

**Option A: Direct run** (for testing)
```bash
./run.sh scripts/collect_data.py --method apify --target 1500
```

**Option B: Tmux session** (recommended for resilience)
```bash
# Create new tmux session
tmux new -s collection

# Run collection
./run.sh scripts/collect_data.py --method apify --target 1500

# Detach (Ctrl+B, then D)
# Collection continues in background

# Reattach later
tmux attach -t collection
```

### Step 3: Monitor Author Diversity

**Watch the logs** for the critical indicator:
```
Author diversity tracking: 25 unique authors already tracked, 50 total replies
Collection complete: 100 pairs from 47 unique authors (GLOBAL)
```

**Key**: `unique authors` should **grow steadily** and approach `total pairs / 10`.

**Check checkpoint**:
```bash
# See unique author count
jq '.unique_authors' data/raw/collection_checkpoint.json

# See total pairs
jq '.pairs_collected' data/raw/collection_checkpoint.json

# Ideal ratio
echo "Target ratio: 10 pairs per author"
```

---

## 📊 Expected Results

### Before Fix (Old Run)
- 1,408 raw pairs collected
- **1 unique author** (100% bot farm) ❌
- 27.9% validation pass rate
- 376 final pairs (all from same author)

### After Fix (New Run)
- 1,500 raw pairs target
- **100-150 unique authors** (10-15 pairs each) ✅
- 60-70% validation pass rate (better filtering)
- ~900-1,000 final pairs (diverse authors)
- Manual curation → 800 best pairs

---

## 🔍 How to Verify Fix is Working

### During Collection
Look for these log lines:
```
[Query 1]
Author diversity tracking: 0 unique authors already tracked, 0 total replies
Collection complete: 50 pairs from 25 unique authors (GLOBAL)

[Query 2]
Author diversity tracking: 25 unique authors already tracked, 50 total replies  ← PERSISTS!
Collection complete: 100 pairs from 47 unique authors (GLOBAL)
```

### After Collection
```bash
# Check final stats
cat data/processed/collection_stats_*.json | jq '.statistics.unique_authors'

# Should see: 80-120 unique authors (not 1!)
```

---

## 💡 What Changed in the Code

**File**: `src/data_collection/apify_collector.py`

```python
# BEFORE (Bug)
def collect_tweets_and_replies(self, ...):
    author_counts = {}  # ❌ Resets each query
    for query in queries:
        # ... collection logic
        # author_counts resets here!

# AFTER (Fixed)
def __init__(self):
    self.author_counts = {}  # ✅ Persists across all queries

def collect_tweets_and_replies(self, ...):
    # Use self.author_counts (global)
    for reply in replies:
        if self.author_counts.get(author_id, 0) >= 10:
            continue  # Skip if author already has 10 replies GLOBALLY
        self.author_counts[author_id] += 1

def _save_checkpoint(self, ...):
    checkpoint = {
        # ...
        "author_counts": self.author_counts  # ✅ Save to checkpoint
    }

def _load_checkpoint(self):
    self.author_counts = checkpoint.get("author_counts", {})  # ✅ Restore from checkpoint
```

---

## 📝 New Search Query Examples

**Tech** (familiar territory):
```
(debugging OR "code review") (tips OR advice) lang:en min_faves:150 -crypto -NFT -airdrop
```

**Business** (different author pool):
```
(startup OR founder) (mistake OR lesson) lang:en min_faves:150 -crypto -web3 -token
```

**Creative** (totally different audience):
```
(writing OR storytelling) (craft OR technique) lang:en min_faves:150 -NFT -airdrop
```

**Career** (broad appeal):
```
(career OR job) (advice OR transition) lang:en min_faves:150 -crypto -web3 -airdrop
```

**Psychology** (thoughtful replies):
```
(productivity OR burnout) (struggle OR solution) lang:en min_faves:150 -crypto -NFT -token
```

---

## ✅ Success Criteria

After collection completes, you should see:

1. **Author Diversity**: 80+ unique authors (not 1!)
2. **Validation Rate**: 60-70% (up from 27.9%)
3. **Final Dataset**: ~900 valid pairs
4. **Manual Curation**: Select best 800 pairs
5. **Quality**: No crypto spam, diverse topics, thoughtful replies

---

## 🆘 Troubleshooting

**Q: What if I still see "Unique authors: 1"?**  
A: Check that you're running the **new code** (not cached). Verify with:
```bash
grep -n "GLOBAL" src/data_collection/apify_collector.py
# Should see: "unique authors (GLOBAL)" in the file
```

**Q: Can I resume from the old checkpoint?**  
A: Yes, but it won't have `author_counts` saved. Better to start fresh.

**Q: How do I reset author tracking manually?**  
A: Delete checkpoint or use:
```python
from src.data_collection.apify_collector import ApifyCollector
collector = ApifyCollector()
collector.reset_author_tracking()
```

---

**Ready to collect high-quality, diverse data!** 🚀

Run: `./run.sh scripts/collect_data.py --method apify --target 1500`

