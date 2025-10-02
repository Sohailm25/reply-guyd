# Data Quality Improvements - Oct 2, 2025

## Problem Analysis

The initial data collection yielded **376 pairs from 1,408 raw pairs**, with **critical quality issues**:

1. **376 replies from ONE author** - Bot farm/engagement farming
2. **72% rejection rate** during validation
3. **980 pairs (70%) rejected** for "Insufficient engagement"
4. **Generic crypto spam** accepted:
   - "Evening fren"
   - "Web3 and Ai, is the greatest revolution ever"
   - "Revolution is the power of change"
   - "Stay locked in on both @recallnet and @RaylsLabs"

### Root Causes:
1. **Generic search queries** → Attracted crypto promotional tweets
2. **min_likes: 1 for replies** → Accepted bot engagement
3. **No spam detection** for generic phrases
4. **No author diversity checks** → Collected from single bot account
5. **Validator min_likes: 2** → Still too lenient

---

## Comprehensive Solution Implemented

### 1. **Targeted Search Queries** (Config: `search_queries`)

**Before:**
```yaml
- "AI OR artificial intelligence lang:en min_faves:200 until:2025-09-30"
- "coding OR programming OR developer lang:en min_faves:200 until:2025-09-30"
```

**After - SPECIFIC, SUBSTANTIVE TOPICS:**
```yaml
# Technical deep-dives (require actual knowledge)
- "(debugging OR \"code review\" OR refactoring) (tips OR advice OR best practices) lang:en min_faves:200 -crypto -NFT -giveaway until:2025-09-30"
- "(system design OR architecture OR scalability) (challenges OR lessons OR experience) lang:en min_faves:200 -crypto -NFT -web3 until:2025-09-30"

# Career/learning (real discussions)
- "(learning OR teaching OR mentoring) (programming OR coding OR software) (advice OR tips OR mistakes) lang:en min_faves:200 -crypto -NFT until:2025-09-30"

# Product/design discussions
- "(product OR UX OR design) (decision OR tradeoff OR lesson) lang:en min_faves:200 -crypto -NFT -token until:2025-09-30"

# Specific tech discussions
- "(TypeScript OR React OR Node) (pattern OR antipattern OR gotcha) lang:en min_faves:200 -crypto -web3 until:2025-09-30"
- "(testing OR CI OR deployment) (strategy OR automation OR challenge) lang:en min_faves:200 -NFT -token until:2025-09-30"
```

**Impact:** Target discussions requiring expertise, explicitly exclude crypto spam keywords.

---

### 2. **Stricter Engagement Thresholds** (Config: `reply_filters`)

**Before:**
```yaml
min_likes: 1          # Too low - accepted bot spam
max_likes: 1000
min_follower_count: 100
max_follower_count: 50000
```

**After:**
```yaml
min_likes: 5          # 5x increase - filter bot engagement
max_likes: 500        # Lowered - avoid viral outliers
min_follower_count: 200  # 2x increase - better bot filtering
max_follower_count: 20000  # Reduced from 50k - avoid influencer advantage
```

**Impact:** Significantly raises quality bar for engagement metrics.

---

### 3. **Author Diversity Enforcement** (New Feature)

**Config Addition:**
```yaml
max_replies_per_author: 10  # NEW - Prevent bot farms
```

**Code Implementation** (`apify_collector.py` lines 93-123):
- Tracks unique author IDs across entire collection
- Limits replies from any single author to 10
- Logs diversity statistics: "X pairs from Y unique authors"

**Impact:** Prevents the "376 replies from 1 author" problem.

---

### 4. **Crypto Spam Detection** (Config: `quality.crypto_spam_keywords`)

**New Keywords List:**
```yaml
crypto_spam_keywords:
  - "gm", "ser", "fren", "wagmi", "ngmi", "degen"
  - "wen", "anon", "based and", "stay locked in"
  - "revolution", "innovative", "onchain"
  - "airdrop", "whitelist"
```

**Validator Implementation** (`data_validator.py` line 276-285):
```python
def _has_crypto_spam(self, text: str) -> bool:
    """Check if text contains crypto spam keywords"""
    text_lower = text.lower()
    for keyword in self.crypto_spam_keywords:
        if keyword.lower() in text_lower:
            return True
    return False
```

**Impact:** Automatically rejects engagement farming replies.

---

### 5. **Word Diversity Check** (Config: `quality.min_unique_words`)

**New Filter:**
```yaml
min_unique_words: 8  # Require actual sentences, not "gm fren"
```

**Validator Implementation** (`data_validator.py` line 287-296):
```python
def _count_unique_words(self, text: str) -> int:
    """Count unique meaningful words (excluding stop words)"""
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', ...}
    meaningful_words = [w for w in words if len(w) > 2 and w not in stop_words]
    return len(set(meaningful_words))
```

**Examples:**
- ✅ "Debugging async issues in Node.js requires careful attention to promise chains and error handling" → 14 unique words
- ❌ "Evening fren" → 2 unique words → REJECTED
- ❌ "gm ser wagmi" → 0 unique words (all are spam keywords) → REJECTED

**Impact:** Filters out low-effort, generic responses.

---

### 6. **Generic Phrase Detection** (Config: `quality.max_generic_phrase_ratio`)

**New Filter:**
```yaml
max_generic_phrase_ratio: 0.5  # Flag if >50% is generic phrases
```

**Generic Phrases List** (`data_validator.py` line 48-52):
```python
self.generic_phrases = [
    "congrats", "congratulations", "amazing", "awesome", "great", "nice",
    "love this", "love it", "this is great", "so cool", "incredible",
    "thanks for sharing", "excited", "can't wait", "looking forward",
]
```

**Validator Implementation** (`data_validator.py` line 298-307):
```python
def _calculate_generic_ratio(self, text: str) -> float:
    """Calculate ratio of generic phrases in text"""
    text_lower = text.lower()
    generic_count = sum(1 for phrase in self.generic_phrases if phrase in text_lower)
    # Normalize by text length
    return min(1.0, generic_count / max(1, len(text) / 100))
```

**Impact:** Filters out low-information engagement replies.

---

### 7. **Increased Validation Threshold** (Config: `quality.min_engagement_for_validation`)

**Before:**
```python
if reply_likes < 2:  # Line 244
    return False
```

**After:**
```yaml
min_engagement_for_validation: 3  # Up from 2
```

```python
if reply_likes < self.min_engagement:  # Line 271
    return False
```

**Impact:** Double-validation ensures only substantive replies pass.

---

## Expected Outcomes

### **Quality Improvements:**
1. **Author Diversity:** Should see 50-200+ unique authors (not 1)
2. **Content Quality:** Substantive technical discussions, not "gm fren"
3. **Engagement Authenticity:** 5+ likes indicates real value
4. **Zero Crypto Spam:** Explicit keyword blocking

### **Collection Efficiency:**
- **May collect fewer pairs** (quality over quantity)
- **Target 800-1,000 pairs** minimum (not 1,200)
- **Higher pass rate** through validation (should see 50-70% vs 28%)

### **Training Data Characteristics:**
- Replies demonstrate **technical knowledge**
- Varied **writing styles** (multiple authors)
- **Replicable patterns** (no celebrity/bot advantages)
- **Substantive engagement** (not generic praise)

---

## Validation Checklist

After running collection, verify:

```bash
# 1. Check unique authors (should be >> 1)
jq -r '.reply_author' data/processed/training_data_*.jsonl | sort -u | wc -l

# 2. Sample random replies
jq -r '.reply' data/processed/training_data_*.jsonl | shuf | head -n 20

# 3. Check engagement distribution
jq -r '.reply_likes' data/processed/training_data_*.jsonl | sort -n | uniq -c

# 4. Check for crypto spam (should be 0)
jq -r '.reply' data/processed/training_data_*.jsonl | grep -iE "(gm|fren|wagmi|ser|degen)" | wc -l
```

---

## Configuration Summary

**Key Changes:**
- ✅ 10 specific search queries with `-crypto` exclusions
- ✅ `min_likes: 1 → 5` (5x increase)
- ✅ `min_follower_count: 100 → 200` (2x increase)
- ✅ `max_follower_count: 50000 → 20000` (2.5x decrease)
- ✅ `max_replies_per_author: 10` (NEW)
- ✅ `min_unique_words: 8` (NEW)
- ✅ `crypto_spam_keywords: [15 keywords]` (NEW)
- ✅ `max_generic_phrase_ratio: 0.5` (NEW)
- ✅ `min_engagement_for_validation: 3` (up from 2)

**Files Modified:**
1. `config/data_collection_config.yaml` - Updated filters
2. `src/data_collection/data_validator.py` - Added 3 new quality checks
3. `src/data_collection/apify_collector.py` - Added author diversity tracking

---

## Next Steps

1. **Run test collection** with 50 pairs:
   ```bash
   ./run.sh python scripts/collect_data.py --method apify --target 50
   ```

2. **Review quality** using validation checklist above

3. **If quality is good**, run full collection:
   ```bash
   ./run.sh python scripts/collect_data.py --method apify --target 1000
   ```

4. **Manual review** of 50-100 random samples before training

---

## Rationale

These changes implement the **"Small, Curated Dataset"** strategy from `current_research.md`:

> "800-1,200 meticulously curated examples outperform 10k+ noisy examples"

By prioritizing **quality over quantity**, we ensure the model learns from **substantive, replicable engagement patterns** rather than bot spam or celebrity effects.

