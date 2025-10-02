# Data Collection Resume Guide

## Overview

The data collection pipeline now supports **automatic checkpointing and resume**, allowing you to safely run long collections without fear of losing progress due to:
- Internet disconnections
- System crashes
- Apify API timeouts
- Manual interruptions (Ctrl+C)

---

## How It Works

### **Automatic Checkpointing**

The collector automatically saves checkpoints:
- **After each query completes** (saves all collected data + progress)
- **On errors** (saves before continuing to next query)
- **Location**: `data/raw/collection_checkpoint.json` + `checkpoint_data.jsonl`

### **What Gets Saved:**
```json
{
  "timestamp": "2025-10-02T12:30:45",
  "pairs_collected": 450,
  "current_query_idx": 3,
  "processed_queries": ["query1", "query2", "query3"]
}
```

Plus all actual tweet-reply pairs in `checkpoint_data.jsonl`.

---

## Usage

### **1. Start a New Collection**

```bash
# Standard start (no resume flag needed)
./run.sh python scripts/collect_data.py --method apify --target 1500
```

### **2. Resume After Interruption**

If your collection is interrupted, simply add `--resume`:

```bash
# Resume from last checkpoint
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

**What happens:**
- ✅ Loads existing pairs from checkpoint
- ✅ Skips already-processed queries
- ✅ Continues from where you left off
- ✅ Maintains all quality filters and validation

**Example output:**
```
🔄 RESUMING from checkpoint:
   Existing pairs: 450
   Resuming from query 4/10
   Processed queries: 3

--- Skipping Query 1/10 (already processed) ---
--- Skipping Query 2/10 (already processed) ---
--- Skipping Query 3/10 (already processed) ---
--- Query 4/10 ---
...
```

---

## Running in Background (Recommended)

For long collections (3,500+ pairs), run in background to survive:
- Terminal closures
- SSH disconnections
- Sleep/hibernation

### **Option 1: tmux (RECOMMENDED)**

```bash
# Start tmux session
tmux new -s datacollection

# Run collection
./run.sh python scripts/collect_data.py --method apify --target 3500

# Detach from session (collection keeps running)
# Press: Ctrl+B, then D

# Reattach later to check progress
tmux attach -t datacollection

# If tmux is already running (after reconnect):
tmux attach -t datacollection
```

**Benefits:**
- ✅ Survives SSH disconnections
- ✅ Can reattach to see live logs
- ✅ Easy to monitor progress

### **Option 2: nohup**

```bash
# Run with nohup (output to file)
nohup ./run.sh python scripts/collect_data.py --method apify --target 3500 > collection.log 2>&1 &

# Check progress
tail -f collection.log

# Find process ID
ps aux | grep collect_data.py

# Kill if needed
kill <PID>
```

**Benefits:**
- ✅ Simpler than tmux
- ✅ Logs saved to file

**Drawbacks:**
- ❌ Can't interact with terminal
- ❌ Harder to stop gracefully

---

## Monitoring Progress

### **Check Current Status**

```bash
# 1. Check checkpoint file
cat data/raw/collection_checkpoint.json

# 2. Count collected pairs
wc -l data/raw/checkpoint_data.jsonl

# 3. Watch live logs
tail -f output/logs/data_collection_*.log | grep -E "(Query [0-9]+/|Total so far)"
```

### **Check Final Results**

```bash
# After completion, check processed data
ls -lh data/processed/training_data_*.jsonl

# Count final pairs
wc -l data/processed/training_data_*.jsonl
```

---

## Common Scenarios

### **Scenario 1: Internet Disconnection**

```bash
# Collection stops mid-query
# ^C or connection lost

# Reconnect, then resume:
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

**Result:** Continues from last completed query (loses only current query progress).

---

### **Scenario 2: Apify API Error**

```bash
# Collection encounters Apify timeout on Query 5

2025-10-02 12:45:30 - ERROR - Error with query '...': Timeout
2025-10-02 12:45:30 - ERROR - Saving checkpoint before continuing...
```

**Result:** Checkpoint saved automatically, continues to Query 6.

---

### **Scenario 3: Manual Stop (Ctrl+C)**

```bash
# You decide to stop after 500 pairs
# Press Ctrl+C

# Resume later to reach target:
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

**Result:** Resumes from last checkpoint (Query 4/10 with 500 pairs).

---

### **Scenario 4: System Crash/Reboot**

```bash
# Mac restarts unexpectedly

# After reboot, check what was saved:
cat data/raw/collection_checkpoint.json

# Resume:
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

**Result:** Loads all pairs from checkpoint, continues collection.

---

## Complete Workflow Example

### **Goal: Collect 3,500 pairs for manual curation**

```bash
# 1. Start collection in tmux
tmux new -s collection

# 2. Run with resume flag (safe to restart)
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume

# 3. Detach from tmux (Ctrl+B, D)

# 4. Check progress periodically
tmux attach -t collection

# 5. If internet drops or you stop it:
#    Just run the same command again - it will resume!

# 6. After completion, checkpoint is automatically cleared
```

**Timeline:**
- **Duration**: ~5-8 hours for 3,500 raw pairs
- **Checkpoints**: Every query (~30-45 minutes)
- **Expected final**: 1,200-1,500 pairs after validation

---

## Checkpoint Management

### **View Checkpoint Status**

```bash
# Check if checkpoint exists
ls data/raw/collection_checkpoint.json

# View checkpoint details
cat data/raw/collection_checkpoint.json | jq .

# Count pairs in checkpoint
wc -l data/raw/checkpoint_data.jsonl
```

### **Manually Clear Checkpoint**

If you want to **start fresh** (discard checkpoint):

```bash
# Remove checkpoint files
rm data/raw/collection_checkpoint.json
rm data/raw/checkpoint_data.jsonl

# Now run without --resume
./run.sh python scripts/collect_data.py --method apify --target 1500
```

### **Automatic Clearing**

Checkpoints are **automatically cleared** when:
- ✅ Collection completes successfully (reaches target)
- ✅ All queries processed
- ✅ Final data saved to `data/processed/`

---

## Best Practices

1. **Always use `--resume` for large collections**
   ```bash
   ./run.sh python scripts/collect_data.py --method apify --target 3500 --resume
   ```
   - If no checkpoint exists, it starts fresh
   - If checkpoint exists, it resumes
   - No downside to always using it!

2. **Use tmux for collections > 2 hours**
   - Protects against SSH disconnections
   - Allows monitoring without disrupting collection

3. **Monitor progress periodically**
   ```bash
   # Check pairs collected so far
   cat data/raw/collection_checkpoint.json | jq .pairs_collected
   ```

4. **Don't delete checkpoint files manually** unless starting fresh
   - Let the system manage them
   - They're automatically cleaned on completion

---

## Troubleshooting

### **Problem: "Resuming from query 0" (not actually resuming)**

**Cause:** Checkpoint file is corrupted or missing.

**Fix:**
```bash
# Check if checkpoint exists
ls -la data/raw/collection_checkpoint.json

# If missing or corrupt, start fresh
rm data/raw/collection_checkpoint.json data/raw/checkpoint_data.jsonl
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

---

### **Problem: Resume loads old data from previous run**

**Cause:** Checkpoint wasn't cleared after last successful run.

**Fix:**
```bash
# Clear old checkpoint
rm data/raw/collection_checkpoint.json data/raw/checkpoint_data.jsonl

# Start fresh
./run.sh python scripts/collect_data.py --method apify --target 1500
```

---

### **Problem: tmux session lost after reboot**

**Cause:** tmux sessions don't survive reboots.

**Fix:** Use `tmux-resurrect` plugin OR rely on checkpoint system:
```bash
# After reboot, checkpoint is still there
./run.sh python scripts/collect_data.py --method apify --target 1500 --resume
```

---

## Summary

### **Key Commands:**

```bash
# New collection (auto-checkpoints)
./run.sh python scripts/collect_data.py --method apify --target 3500

# Resume after interruption
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume

# Run in background with tmux
tmux new -s collection
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume
# Ctrl+B, D to detach

# Monitor progress
tmux attach -t collection
tail -f output/logs/data_collection_*.log

# Check checkpoint status
cat data/raw/collection_checkpoint.json | jq .
```

### **Safety Features:**

- ✅ Automatic checkpoints every query
- ✅ Survives internet disconnections
- ✅ Survives Apify API errors
- ✅ No data loss on interruption
- ✅ Can safely Ctrl+C and resume later

### **For Your 3,500 Pair Collection:**

```bash
# Recommended workflow
tmux new -s collection
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume
# Ctrl+B, D

# Check back in a few hours:
tmux attach -t collection
```

**Expected outcome:** ~1,200-1,500 pairs after validation, ready for manual curation to 800.

