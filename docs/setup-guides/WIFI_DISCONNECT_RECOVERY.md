# WiFi Disconnect Recovery Guide

## 🔌 What Happens When WiFi Disconnects

### ✅ Will Survive:
- **Tmux session**: Runs locally, stays alive
- **Terminal process**: Protected by tmux
- **Checkpoint data**: Already saved to disk (57 pairs, 40 authors)
- **Your collected data**: Safe in `data/raw/checkpoint_data.jsonl`

### ❌ Will Fail:
- **Active Apify API call**: Times out after 300 seconds
- **Collection script**: Crashes with network error
- **Current query**: Incomplete (will be retried on resume)

---

## 🏢 When You Get to Work (After Reconnecting WiFi)

### Step 1: Check Tmux Session Status
```bash
tmux list-sessions
```

You should see:
```
collection-new: 1 windows (created Thu Oct  2 09:09:41 2025)
```

---

### Step 2: Reattach and Check Process Status
```bash
tmux attach -t collection-new
```

**You'll see one of two scenarios:**

#### Scenario A: Process Crashed (Most Likely)
```
Error: Timeout
Error collecting tweets via Apify...
(back to shell prompt)
```

#### Scenario B: Process Still Running (Unlikely)
```
Processing tweet X/Y...
(still scrolling logs)
```

---

### Step 3: Resume Collection

#### If Process Crashed:
```bash
# You're already in the tmux session, just restart with --resume flag
./run.sh scripts/collect_data.py --method apify --target 1500 --resume
```

**What `--resume` does:**
- Loads checkpoint file
- Restores 57 pairs already collected ✓
- Restores 40 unique author tracking ✓
- Skips first 2 completed queries ✓
- Continues from query 3

#### If Process Still Running:
```bash
# Just let it continue, detach again with:
Ctrl+B, then D
```

---

### Step 4: Verify Resume Worked

After restarting, check the logs:
```bash
tail -20 output/logs/data_collection_*.log
```

You should see:
```
🔄 RESUMING from checkpoint:
   Existing pairs: 57
   Resuming from query 3/15
   Processed queries: 2
Author diversity tracking: 40 unique authors already tracked, 57 total replies
```

---

## 🎯 Quick Reference Commands

### Check if collection is still running:
```bash
tmux attach -t collection-new
# Look for active logs or error message
```

### Resume from checkpoint:
```bash
# Inside tmux session
./run.sh scripts/collect_data.py --method apify --target 1500 --resume
```

### Check what you have saved:
```bash
jq '.' data/raw/collection_checkpoint.json
```

### View collected pairs:
```bash
wc -l data/raw/checkpoint_data.jsonl
# Should show: 57 pairs
```

---

## ⚠️ Important Notes

1. **Don't delete checkpoint files** - They contain your progress!
2. **Always use `--resume` flag** - Otherwise you'll start from scratch
3. **Author tracking will persist** - The 40 unique authors are saved in checkpoint
4. **Queries won't re-run** - Completed queries 1-2 will be skipped

---

## 🆘 If Something Goes Wrong

### Checkpoint file is missing/corrupted:
```bash
# Check if it exists
ls -lh data/raw/collection_checkpoint.json

# If missing, you lost the checkpoint (rare)
# But collected pairs might still be in processed/
ls -lh data/processed/
```

### Resume flag doesn't work:
```bash
# Manually check what you have
cat data/raw/collection_checkpoint.json | jq '{pairs, authors, queries}'

# If checkpoint looks good, try running without resume first
# Then interrupt and add --resume
```

### Want to start fresh instead:
```bash
# Delete checkpoint
rm -f data/raw/collection_checkpoint.json data/raw/checkpoint_data.jsonl

# Start new collection
./run.sh scripts/collect_data.py --method apify --target 1500
```

---

**TL;DR**: 
1. ✅ Your progress is saved (57 pairs, 40 authors)
2. 🔄 Reattach: `tmux attach -t collection-new`
3. 🚀 Resume: `./run.sh scripts/collect_data.py --method apify --target 1500 --resume`

