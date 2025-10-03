# 🔍 DATA RECOVERY: Retrieving Lost Pairs from Apify Storage

## 💡 The Discovery

**Great insight!** Even though our local `checkpoint_data.jsonl` was overwritten, **Apify stores ALL the raw tweet data** from each actor run in their cloud storage!

This means we can potentially recover the tweets from Queries 1-6, though we'll need to understand what exactly Apify stored.

---

## 🤔 What Apify Actually Stores

### Important Distinction

**What Apify DOES store:**
- Raw tweets from search queries (the main tweets we searched for)
- Each actor run creates a dataset with ID `defaultDatasetId`
- These datasets persist in Apify's cloud storage

**What Apify DOES NOT store:**
- The replies we fetched (those were fetched separately per tweet)
- Our processed tweet-reply pairs
- Our filtering/quality checks

---

## 📊 Recovery Options

### Option 1: Recover Raw Tweets Only (Fast, Partial Recovery)

**What you get:**
- All main tweets from Queries 1-6 (~500-1000 tweets)
- Tweet text, metadata, engagement stats

**What you DON'T get:**
- The replies (those were fetched separately)
- The tweet-reply pairs we created

**Cost:** Free (just retrieving what we already paid for)

**Command:**
```bash
./run.sh python scripts/recovery/recover_apify_data.py --start-time "2025-10-02 09:10:00"
```

---

### Option 2: Re-Fetch Replies for Recovered Tweets (Complete, Costs Money)

**What you get:**
- All main tweets from Queries 1-6
- Fresh replies for each tweet
- Complete tweet-reply pairs

**Cost:** 
- Fetching replies again: ~$15-20 (since we already have the main tweets)
- Less than original $35 because we don't re-search

**Process:**
1. Recover raw tweets from Apify storage (free)
2. For each tweet, fetch replies again (costs API calls)
3. Apply our quality filters
4. Get back the ~1,713 pairs (or similar count)

---

## 🎯 Recommended Approach

### Reality Check First

Let's see what Apify actually stored:

```bash
# Run the recovery script to see what we have
./run.sh python scripts/recovery/recover_apify_data.py \
    --start-time "2025-10-02 09:10:00" \
    --hours 12

# This will show you:
# - How many actor runs were recorded
# - How many datasets were created  
# - How many tweets are stored
# - What data is actually available
```

**Expected output:**
```
Found 6 runs in the last 12 hours
Run 1/6:
  ID: abc123...
  Started: 2025-10-02T15:10:21.000Z
  Status: SUCCEEDED
  Items: 257 tweets

... (5 more runs)

RECOVERY SUMMARY:
===================
Total runs found: 6
Datasets fetched: 6
Total items: 1,200
Unique tweets: 1,150
✅ Saved 1,150 recovered tweets to: data/recovered/recovered_tweets_20251002_162800.jsonl
```

---

## 💰 Cost Analysis

### Option A: Fresh Collection (What I Recommended Earlier)

**Cost:** ~$40 for 2,000 pairs  
**Result:** Clean, fresh data with new system  
**Time:** 6-8 hours  
**Risk:** 0% (new system is bulletproof)

### Option B: Recover + Re-fetch Replies

**Cost:** ~$15-20 to re-fetch replies  
**Result:** Original ~1,713 pairs from Queries 1-6  
**Time:** 3-4 hours  
**Risk:** Low (same data, just re-processed)

### Option C: Combination (Best?)

**Cost:** ~$55-60 total  
**Result:** ~3,700+ pairs (1,713 recovered + 2,000 fresh)  
**Time:** 9-12 hours  
**Risk:** 0% (double the data!)

---

## 🛠️ Step-by-Step Recovery Process

### Step 1: Check What's Available

```bash
# See what Apify has stored
./run.sh python scripts/recovery/recover_apify_data.py \
    --start-time "2025-10-02 09:10:00" \
    --output tweets_morning_collection.jsonl
```

**This tells you:**
- How many tweets are recoverable
- If it's worth re-fetching replies

---

### Step 2: Analyze the Recovered Data

```bash
# Count recovered tweets
wc -l data/recovered/recovered_tweets_*.jsonl

# View a sample
head -n 5 data/recovered/recovered_tweets_*.jsonl | jq '.'
```

**Look for:**
- Are these the tweets from Queries 1-6?
- Do they have engagement data?
- Are they high-quality tweets worth re-processing?

---

### Step 3: Decide on Next Steps

**If recovered tweets look good:**

Create a script to re-fetch replies for these specific tweets:

```bash
# TODO: Create re-fetch script
./run.sh python scripts/recovery/refetch_replies_for_recovered.py \
    --input data/recovered/recovered_tweets_*.jsonl \
    --output data/recovered/recovered_pairs.jsonl
```

**This would:**
1. Load recovered tweets
2. For each tweet, call Apify to get current replies
3. Apply our quality filters
4. Save as tweet-reply pairs

**Cost:** ~$15-20 (only fetching replies, not searching)

---

## 🎯 My Recommendation

### Evaluate First, Then Decide

1. **Run the recovery script** to see what's available (5 minutes, FREE)
   ```bash
   ./run.sh python scripts/recovery/recover_apify_data.py --start-time "2025-10-02 09:10:00"
   ```

2. **Check the output:**
   - If you have ~1,000-1,500 tweets: Consider re-fetching replies (~$15-20)
   - If you have fewer: Might not be worth it, just do fresh collection

3. **Best case scenario:**
   - Recover tweets: FREE
   - Re-fetch replies: ~$20
   - Fresh collection: ~$40
   - **Total: ~$60 for 3,500+ pairs** (excellent value!)

---

## ⚠️ Important Notes

### What This Recovery Script Does

✅ Lists all actor runs from specified time  
✅ Fetches datasets from Apify cloud storage  
✅ Deduplicates tweets by ID  
✅ Saves raw tweet data  

### What This Recovery Script DOESN'T Do

❌ Fetch replies (those require new API calls)  
❌ Create tweet-reply pairs (need replies first)  
❌ Apply quality filters (can do after recovery)  

---

## 🔍 Understanding the Cost

### Why Re-fetching Replies Costs Less

**Original collection:**
- Search for tweets: ~$10 (1,000 search results)
- Fetch replies: ~$25 (50 replies × 1,000 tweets)
- **Total: ~$35**

**Recovery approach:**
- Search for tweets: $0 (already have them!)
- Fetch replies: ~$15-20 (replies for recovered tweets only)
- **Total: ~$15-20** ✅

---

## 🚀 Quick Start Commands

### 1. See What's Available (FREE)
```bash
./run.sh python scripts/recovery/recover_apify_data.py \
    --start-time "2025-10-02 09:10:00"
```

### 2. Check the Results
```bash
# How many tweets recovered?
wc -l data/recovered/*.jsonl

# What do they look like?
head -n 3 data/recovered/*.jsonl | jq '.'
```

### 3. Decide Your Path
- **If 1,000+ tweets:** Consider re-fetching replies (~$20)
- **If <500 tweets:** Just do fresh collection (~$40)
- **If unsure:** Let me know the count and I'll advise!

---

## 📞 What to Do Next

**Run this command NOW:**
```bash
./run.sh python scripts/recovery/recover_apify_data.py --start-time "2025-10-02 09:10:00"
```

**Then tell me:**
1. How many runs were found?
2. How many unique tweets were recovered?
3. Do you want to re-fetch replies for them?

**Based on that, I'll help you decide the best path forward!** 🎯

