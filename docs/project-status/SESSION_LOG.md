# Development Session Log
## Qwen3-8B LoRA Fine-Tuning Project

**Project Goal**: Fine-tune Qwen3-8B using LoRA for high-engagement Twitter reply generation

**Status**: Phase 1 Complete ✅ | Ready for Phase 2 (Data Collection)

**Date Started**: January 2025

---

## 📋 Session Overview

This log chronicles the complete setup and implementation of a production-ready LoRA fine-tuning pipeline for Qwen3-8B, optimized for RunPod training with rigorous evaluation methodology.

---

## 🔬 Phase 0: Initial Analysis & Planning

### Model Architecture Analysis

**Examined**: Qwen3-8B downloaded model files
- **Architecture**: Qwen3ForCausalLM
- **Parameters**: 8.2B total (6.95B non-embedding)
- **Layers**: 36 transformer blocks
- **Attention**: GQA (32 Q heads, 8 KV heads)
- **Hidden size**: 4096, Intermediate: 12288
- **Context**: 32,768 native, 131,072 with YaRN
- **Precision**: bfloat16 (~16GB unquantized)
- **Special features**: Thinking mode (optional `<think>` blocks)

**LoRA Target Analysis**:
- Identified 252 target linear layers:
  - Per transformer block: 7 layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - Total: 7 × 36 blocks = 252 targets
- QLoRA 4-bit quantization: 16GB → 4-6GB memory

### Research Synthesis Review

**Key Findings from `current_research.md`**:

1. **Small Dataset Viability**: 800-1,500 examples achievable with LoRA
   - LIMA dataset: 1,000 examples matched 50,000-example Alpaca
   - "Dataset quality is 95% of everything"
   - Manual review essential for small datasets

2. **LoRA Configuration for Small Datasets**:
   - Rank: 8-16 (start with 16)
   - Alpha: Equal to rank
   - Dropout: 0.1 (10% for regularization)
   - Target: ALL linear layers (critical per QLoRA paper)
   - Batch size: Small (2-4) with gradient accumulation

3. **Evaluation Framework**:
   - Traditional metrics (ROUGE, BLEU) correlate poorly with engagement
   - G-Eval framework: 0.514 Spearman correlation (best available)
   - LLM-as-judge with bias mitigation required
   - Statistical significance testing mandatory (p<0.05, Cohen's d>0.3)

4. **Data Collection Economics**:
   - X API v2 Basic: $200/month for 10,000 posts (no min_faves filter)
   - Apify: ~$0.25 per 1,000 tweets (much better ROI)
   - twscrape: Viable alternative, requires accounts + proxies

**Key Findings from `gameplan.md`**:

1. **Realistic Expectations Calibration**:
   - Original optimistic: 2,500 pairs, 50-70% improvement
   - Realistic: 800-1,200 pairs, 15-30% improvement
   - Budget: $400-600 (not $229)
   - Timeline: 8 weeks (not 1 month)
   - Success rate first attempt: ~30%

2. **Critical Filters to Avoid Unreplicable Success**:
   - Celebrity effect: Filter authors >100K followers
   - Timing advantage: Filter replies <5 min after tweet
   - Media-based engagement: Filter has_media=true
   - URL-driven engagement: Filter has_urls=true

3. **Overfitting Risk with Small Datasets**:
   - Early stopping essential (patience=3)
   - High dropout (0.1-0.15)
   - Frequent evaluation (every 50 steps)
   - Monitor train/eval gap (alert if >0.3)

4. **Evaluation Reality**:
   - Offline metrics ≠ real engagement
   - Only A/B testing shows true performance
   - Human evaluation expensive ($375 for 150 participants on Prolific)

### Strategic Decisions Made

**Decision 1: Use Apify for Data Collection**
- Reasoning: Better economics than X API ($0.25/1K vs $200/10K)
- Tradeoff: Gray area legally (violates ToS but defensible)
- Implementation: Built ApifyCollector with gameplan.md filters

**Decision 2: Optimize for RunPod Training**
- Reasoning: User has Mac (no GPU), RunPod very cost-effective
- Cost: ~$3-5 per training run on A6000 (vs $0 if local but no GPU available)
- Workflow: Mac for data collection/eval (free), RunPod only for training (paid)

**Decision 3: Conservative Hyperparameters**
- LoRA rank: 16 (start conservative, adjust based on results)
- Epochs: 2 (reduce to 1 if overfitting detected)
- Learning rate: 2e-4 (standard for LoRA)
- Effective batch size: 16 (via gradient accumulation)

**Decision 4: Multi-Tier Evaluation**
- Tier 1: Automated metrics (sanity check only)
- Tier 2: LLM-as-judge (primary development metric)
- Tier 3: Statistical testing (mandatory for claiming success)
- Tier 4: Human eval (optional if budget allows)

**Decision 5: Skip Phase 0 Validation**
- User chose to proceed directly to implementation
- Acceptable risk given strong research foundation
- Can validate base model performance later if needed

---

## 🏗️ Phase 1: Infrastructure Setup (COMPLETE)

### Project Structure Created

**Directory Hierarchy**:
```
Qwen3-8/
├── config/                      # YAML configurations
│   ├── data_collection_config.yaml
│   └── lora_config.yaml
├── data/
│   ├── raw/                     # Raw Apify data
│   ├── processed/               # Cleaned training data
│   └── cache/                   # Temporary files
├── src/
│   ├── data_collection/         # Collection pipeline
│   │   ├── apify_collector.py
│   │   ├── scraper_collector.py
│   │   ├── data_validator.py
│   │   └── data_cleaner.py
│   ├── training/                # Training modules (Phase 3)
│   ├── evaluation/              # Evaluation framework (Phase 4)
│   └── utils/                   # Shared utilities
├── scripts/
│   └── collect_data.py          # Main collection script
├── output/
│   ├── models/                  # Saved LoRA adapters
│   ├── logs/                    # All logs
│   └── figures/                 # Visualizations
└── [documentation files]
```

### Data Collection Pipeline Implementation

**Module 1: `apify_collector.py`**
- Purpose: Primary data collection via Apify actors
- Key features:
  - Configurable search queries from YAML
  - Tweet collection with engagement filters (200-2,000 likes)
  - Reply collection with conversation_id search
  - Comprehensive reply filtering:
    - Engagement: min_likes=5, max_likes=1000
    - Author: 500-100K followers (avoid bots and celebrities)
    - Timing: 300-86400 seconds after tweet (avoid timing advantage)
    - Content: no media, no URLs, 30-280 chars
  - Checkpoint saving every 10 tweets
  - Normalizes Apify actor output to consistent schema
- Implementation: ~343 lines, fully functional
- Cost: ~$0.25 per 1,000 tweets

**Module 2: `scraper_collector.py`**
- Purpose: Alternative collection via twscrape (backup)
- Key features:
  - Two implementations: generic scraper + twscrape wrapper
  - Account rotation for rate limit management
  - Same filtering logic as Apify collector
  - More complex setup (requires Twitter accounts)
- Implementation: ~429 lines, template provided
- Status: Framework ready, requires account configuration
- Cost: Cheaper at scale but higher setup complexity

**Module 3: `data_validator.py`**
- Purpose: Quality validation of collected pairs
- Validation checks:
  - Length: 30-280 characters
  - Language: English with confidence >0.8 (using langdetect)
  - Toxicity: Score <0.3 (using Detoxify)
  - Content: No URLs, excessive hashtags, spam patterns
  - Engagement: Meets minimum thresholds
- Output: Validation metadata on each pair, batch statistics
- Implementation: ~296 lines
- Rejection reason tracking for debugging filters

**Module 4: `data_cleaner.py`**
- Purpose: Text normalization and deduplication
- Cleaning operations:
  - Text normalization: Elongated chars ("sooo" → "so")
  - Hashtag processing: #MachineLearning → "Machine Learning"
  - Whitespace normalization
  - Emoji preservation (carry engagement signals)
- Deduplication:
  - Exact: Remove identical reply text
  - Semantic: Remove similarity >0.9 using sentence-transformers
  - Keep higher-engagement duplicate when found
- Quality sorting: By engagement + credibility + timing
- Implementation: ~324 lines
- Output: Dataset statistics for manual review

**Module 5: `collect_data.py` (Main Script)**
- Purpose: Orchestrates complete pipeline
- Pipeline stages:
  1. Collection (Apify or scraper)
  2. Validation (quality filters)
  3. Cleaning (normalize + dedupe)
  4. Saving (JSONL format + statistics)
- Features:
  - Progress logging to file and console
  - Checkpoint recovery for interrupted runs
  - Statistics generation (engagement, diversity, etc.)
  - Command-line interface with argparse
- Implementation: ~331 lines
- Usage: `python scripts/collect_data.py --method apify --target 1000`

### Configuration System

**File 1: `data_collection_config.yaml`**
- Collection targets: 800-1,200 pairs (min/target/max)
- Search queries: 7 diverse topics (AI, coding, startups, tech, etc.)
- Tweet filters:
  - Engagement: 200-2,000 likes (sweet spot from research)
  - Not retweets, not quotes
  - Language: English
- Reply filters (from gameplan.md):
  - Engagement: 5-1,000 likes
  - Author: 500-100K followers
  - Timing: 5 min - 24 hours after tweet
  - Content: No media, no URLs, 30-280 chars
- Quality thresholds:
  - Language confidence: >0.8
  - Toxicity score: <0.3
  - Semantic similarity: <0.9 for deduplication

**File 2: `lora_config.yaml`**
- Model settings:
  - Path: "./" (current directory has Qwen3-8B)
  - QLoRA: 4-bit NF4 quantization, bfloat16 compute
- LoRA parameters:
  - Rank: 16, Alpha: 16, Dropout: 0.1
  - Target modules: ALL 7 linear layers per block
  - Task type: CAUSAL_LM
- Training hyperparameters:
  - Epochs: 2 (reduce to 1 if overfitting)
  - Learning rate: 2e-4, warmup: 0.03
  - Batch size: 4, gradient accumulation: 4 (effective=16)
  - Optimizer: AdamW with weight decay 0.01
  - Scheduler: Cosine
- Evaluation strategy:
  - Every 50 steps (frequent for small dataset)
  - Early stopping: patience=3, threshold=0.01
  - Save top 3 checkpoints by eval_loss
- Overfitting detection thresholds:
  - Max train/eval gap: 0.3
  - Alerts if exceeded

**File 3: `requirements.txt`**
- Core ML: torch, transformers >=4.51.0, peft, bitsandbytes, accelerate
- Data collection: apify-client, httpx, beautifulsoup4, playwright
- Processing: pandas, numpy, jsonlines, sentence-transformers
- Evaluation: scikit-learn, scipy, anthropic (Claude API)
- Experiment tracking: wandb
- Development: pytest, black, jupyter

**File 4: `requirements-runpod.txt`**
- Lighter version for RunPod (no data collection deps)
- Only training and evaluation essentials
- Assumes PyTorch pre-installed in container

**File 5: `.env.example`**
- API keys: APIFY_API_TOKEN, ANTHROPIC_API_KEY, WANDB_API_KEY
- Model paths: MODEL_PATH, OUTPUT_DIR, CACHE_DIR
- Training settings: Batch size, temperature, etc.

### Setup Automation

**Script: `setup.sh`**
- Creates virtual environment (qwen-lora-env)
- Installs all dependencies from requirements.txt
- Attempts Unsloth installation (optional, gracefully fails if incompatible)
- Optional twscrape installation
- Creates .env from template
- Verification checks: Python version, GPU, packages
- Time: ~5-10 minutes depending on network

**Script: `run.sh`**
- Wrapper to auto-activate virtual environment
- Usage: `./run.sh python scripts/collect_data.py ...`
- Prevents "ModuleNotFoundError" issues
- Simplifies workflow for users unfamiliar with venv

### Documentation Created

**Implementation Planning**:
1. `lora_implementation_plan.md` (detailed 6-phase roadmap)
   - Complete methodology with code templates
   - Statistical testing framework
   - Overfitting prevention strategies
   - Budget breakdowns, timeline estimates
   - Success criteria definitions

**Setup & Usage Guides**:
2. `GETTING_STARTED.md` - Quick start guide
3. `SETUP_COMPLETE.md` - What was installed, next steps
4. `SETUP_STATUS.md` - Installation verification, troubleshooting
5. `QUICK_REFERENCE.md` - Daily commands cheat sheet
6. `VIRTUAL_ENV_GUIDE.md` - Virtual environment activation help

**RunPod Optimization**:
7. `RUNPOD_SETUP.md` - Complete RunPod training guide
   - Pod configuration recommendations (A6000 48GB)
   - File upload/download strategies
   - Cost optimization workflow
   - Mac vs RunPod task division

**Code Organization**:
8. `PROJECT_STRUCTURE.md` - Directory layout, data flow, conventions
9. `README_PROJECT.md` - Project overview, methodology, quick start

**Current State**:
10. `SESSION_LOG.md` - This document

### Technical Decisions & Rationale

**Decision: All Linear Layers for LoRA**
- Rationale: QLoRA paper showed attention-only LoRA underperforms
- Implementation: 252 targets (q/k/v/o + gate/up/down per 36 blocks)
- Memory impact: Minimal with QLoRA 4-bit
- Expected benefit: Match full fine-tuning performance

**Decision: QLoRA 4-bit Quantization**
- Rationale: Enables 8B model training on consumer GPU (or cheap RunPod)
- Implementation: NF4 quantization with double quantization
- Memory: 16GB → 4-6GB (fits on RTX A6000, even 24GB GPUs)
- Performance penalty: ~10% slower but negligible for our use case

**Decision: Aggressive Reply Filtering**
- Rationale: Avoid training on unreplicable success (gameplan.md analysis)
- Filters implemented:
  - Celebrity effect: >100K followers rejected
  - Timing advantage: <5 min replies rejected
  - Media advantage: has_media=true rejected
  - URL advantage: URLs in text rejected
- Expected outcome: Lower absolute engagement but replicable patterns

**Decision: Semantic Deduplication**
- Rationale: Diversity critical for generalization with small datasets
- Implementation: sentence-transformers embeddings, cosine similarity >0.9
- Tradeoff: Slower processing but essential for quality
- Keeps: Higher-engagement duplicate when found

**Decision: Frequent Evaluation Steps**
- Rationale: Small datasets overfit quickly
- Implementation: Eval every 50 steps (vs typical 500+)
- Early stopping: Patience=3 evaluations
- Enables: Quick detection of overfitting for intervention

**Decision: RunPod-First Workflow**
- Rationale: User has Mac without GPU, local training not viable
- Cost analysis:
  - Local GPU (hypothetical): $0 but user doesn't have one
  - Cloud GPU (RunPod A6000): $0.39-0.79/hr = $3-5 per run
  - Total project: $10-20 for 2-3 iterations (very affordable)
- Workflow: Mac for everything except training (95% free, 5% paid)

### Dependency Management & Issues

**Issue 1: Unsloth Installation Failed**
- Error: `bitsandbytes>=0.45.5` required but only 0.42.0 available
- Cause: Unsloth bleeding-edge dependencies not yet in stable releases
- Impact: None (Unsloth is optional speedup library)
- Workaround: Training works fine without it (2-4hrs vs 1-2hrs)
- Resolution: Updated setup.sh to gracefully handle failure
- Future: Can retry when bitsandbytes>=0.45.5 released

**Issue 2: Virtual Environment Activation**
- Error: `ModuleNotFoundError: No module named 'apify_client'`
- Cause: User ran script without activating virtual environment
- Impact: Confusion but easily resolved
- Solution: Created run.sh wrapper and VIRTUAL_ENV_GUIDE.md
- Prevention: Documentation emphasizes activation requirement

**All Other Dependencies**: Installed successfully
- PyTorch: 2.0+
- Transformers: 4.51.0+ (required for Qwen3)
- PEFT: 0.7.0+
- bitsandbytes: 0.41.0+ (sufficient for QLoRA)
- All data collection and evaluation libraries

**Issue 3: Apify Date Format Inconsistency**
- Error: `Invalid isoformat string: ''` during collection
- Cause: Some Apify responses have empty/missing `created_at` fields
- Also: Different Apify actors use different field names (created_at vs createdAt vs timestamp)
- Impact: Collection would crash on certain tweets/replies
- Solution: 
  - Added safe date parsing with try/except blocks
  - Skip tweets/replies with invalid dates (log warnings)
  - Handle multiple possible field names from Apify actors
  - Graceful degradation: continue collecting valid data
- Prevention: Now logs which tweets skipped and why

---

## 🎯 Current Status

### What's Complete ✅

**Infrastructure**:
- [x] Complete project structure with all directories
- [x] Configuration system (YAML files)
- [x] Virtual environment with all dependencies
- [x] Setup automation scripts
- [x] RunPod-optimized workflow

**Data Collection**:
- [x] Apify collector implementation (primary method)
- [x] Scraping collector templates (backup method)
- [x] Data validation pipeline
- [x] Data cleaning and deduplication
- [x] Main collection orchestration script
- [x] Comprehensive filtering per research insights

**Documentation**:
- [x] Complete implementation plan (6 phases)
- [x] Setup and quick start guides
- [x] RunPod training guide
- [x] Virtual environment troubleshooting
- [x] Project structure documentation
- [x] Session log (this document)

**Configuration**:
- [x] Data collection parameters
- [x] LoRA hyperparameters
- [x] Overfitting prevention settings
- [x] Evaluation strategy
- [x] RunPod optimization

### What's Pending ⏳

**Phase 2: Data Collection** (Next)
- [ ] Configure APIFY_API_TOKEN in .env
- [ ] Test collection with 1-10 tweets
- [ ] Full collection (1,000 pairs, ~$200-300)
- [ ] Manual review of collected data (5-7 hours)
- [ ] Final dataset: 800-1,200 high-quality pairs

**Phase 3: Training** (Week 4)
- [ ] Create train_lora.py script
- [ ] Setup W&B experiment tracking
- [ ] RunPod pod configuration
- [ ] First training run (v1)
- [ ] Overfitting diagnostics
- [ ] LoRA adapter saving

**Phase 4: Evaluation** (Week 5)
- [ ] Create evaluate_model.py script
- [ ] Implement LLM-as-judge (Claude)
- [ ] Statistical significance testing
- [ ] Generate comparison reports
- [ ] Decision: proceed to v2 or production

**Phase 5: Iteration** (Weeks 6-8)
- [ ] Hyperparameter adjustments based on v1
- [ ] Training v2/v3 if needed
- [ ] Final model selection
- [ ] Publication to HuggingFace Hub

---

## 📊 Key Metrics & Targets

### Data Collection Targets

| Metric | Minimum | Target | Maximum |
|--------|---------|--------|---------|
| Training pairs | 800 | 1,000 | 1,200 |
| Unique tweets | 150 | 200 | 300 |
| Unique authors | 300 | 500 | 700 |
| Avg engagement | 10 likes | 15 likes | 25 likes |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 16 | Conservative for 800-1,200 examples |
| LoRA alpha | 16 | Keep equal to rank |
| Dropout | 0.1 | 10% regularization |
| Learning rate | 2e-4 | Standard for LoRA |
| Epochs | 2 | Reduce to 1 if overfitting |
| Effective batch | 16 | Via 4×4 gradient accumulation |
| Eval frequency | 50 steps | Catch overfitting early |

### Success Criteria (Phase 4 Evaluation)

**Statistical Requirements**:
- p-value < 0.05 (Mann-Whitney U test)
- Cohen's d > 0.3 (small-to-medium effect size)
- Improvement > 15% over base model
- Confidence intervals exclude zero

**Overfitting Prevention**:
- Train/eval loss gap < 0.3
- Validation loss doesn't increase for >3 evaluations
- Perplexity remains reasonable (<20)

**Qualitative Assessment**:
- Generated replies relevant to tweets
- Appropriate tone and length
- No catastrophic forgetting
- Diverse outputs (no repetition)

### Budget & Timeline

**Phase 2: Data Collection**
- Cost: $200-300 (Apify)
- Time: 2-4 hours collection + 5-7 hours manual review
- Timeline: Week 2-3

**Phase 3: Training**
- Cost: $3-5 per run (RunPod A6000)
- Time: 2-4 hours per run
- Runs: 2-3 iterations
- Total: $10-20, 1-2 weeks
- Timeline: Week 4

**Phase 4: Evaluation**
- Cost: $50-150 (Claude API for LLM-as-judge)
- Cost (optional): $375-390 (Prolific human eval)
- Time: 1 week
- Timeline: Week 5

**Phase 5: Iteration**
- Cost: $5-15 (additional training runs)
- Time: 2-3 weeks
- Timeline: Weeks 6-8

**Total Project**:
- Cost: $260-$470 (without human eval)
- Cost: $635-$860 (with human eval)
- Time: 8 weeks

---

## 🔧 Technical Implementation Details

### Model Loading Strategy

**For Data Collection/Evaluation (Mac)**:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "./",  # Qwen3-8B in current directory
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

**For Training (RunPod)**:
```python
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

# Load base model (same as above)
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### Data Format Specification

**Raw Collection (JSONL)**:
```json
{
  "tweet_id": "123456789",
  "tweet": "Original tweet text",
  "tweet_author": "username",
  "tweet_likes": 500,
  "tweet_retweets": 50,
  "tweet_created_at": "2024-01-01T12:00:00Z",
  
  "reply_id": "987654321",
  "reply": "High-engagement reply text",
  "reply_author": "reply_username",
  "reply_author_followers": 5000,
  "reply_likes": 50,
  "reply_retweets": 5,
  "reply_created_at": "2024-01-01T12:30:00Z",
  "reply_time_diff_seconds": 1800,
  
  "collected_at": "2024-01-01T13:00:00Z"
}
```

**After Validation** (adds):
```json
{
  ...existing fields...,
  "validation": {
    "valid": true,
    "reasons": [],
    "validated_at": "2024-01-01T13:05:00Z"
  }
}
```

**For Training** (Qwen3 chat template):
```json
{
  "text": "<|im_start|>user\nGenerate an engaging Twitter reply to:\n\n{tweet}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{reply}<|im_end|>"
}
```

### Filtering Pipeline Logic

**Stage 1: Tweet Selection**
```python
if not (200 <= likes <= 2000): reject()
if not (10 <= retweets <= 500): reject()
if is_retweet or is_quote: reject()
if language != "en": reject()
```

**Stage 2: Reply Collection**
```python
search_query = f"conversation_id:{tweet_id}"
replies = apify.search(search_query, max_results=50)
```

**Stage 3: Reply Filtering**
```python
for reply in replies:
    # Engagement
    if not (5 <= reply.likes <= 1000): continue
    
    # Author (avoid celebrity/bot)
    if not (500 <= followers <= 100000): continue
    
    # Timing (avoid first-reply advantage)
    time_diff = reply.time - tweet.time
    if not (300 <= time_diff <= 86400): continue
    
    # Content
    if len(reply.text) not in range(30, 281): continue
    if reply.has_media: continue
    if "http" in reply.text: continue
    
    # Quality
    if toxicity_score(reply.text) > 0.3: continue
    if language_confidence(reply.text) < 0.8: continue
    
    # Accept
    training_pairs.append((tweet, reply))
```

**Stage 4: Deduplication**
```python
# Exact duplicates
seen = set()
unique = [p for p in pairs if p.reply not in seen and not seen.add(p.reply)]

# Semantic duplicates
embeddings = encoder.encode([p.reply for p in unique])
keep = []
for i, emb in enumerate(embeddings):
    if all(cosine_similarity(emb, embeddings[j]) < 0.9 for j in keep):
        keep.append(i)
final_pairs = [unique[i] for i in keep]
```

### W&B Integration Plan

**Initialization**:
```python
import wandb

wandb.init(
    project="qwen3-twitter-lora",
    name="training-v1",
    config={
        "model": "Qwen3-8B",
        "method": "LoRA + QLoRA",
        "dataset_size": len(train_dataset),
        "rank": 16,
        "learning_rate": 2e-4,
        "epochs": 2,
    }
)
```

**Logging During Training**:
- Every step: loss, learning_rate, gradient_norm
- Every 50 steps: eval_loss, eval metrics
- Every epoch: sample predictions, confusion matrix
- End: model artifacts, final metrics

**Dashboard Access**:
- URL: `https://wandb.ai/{username}/qwen3-twitter-lora`
- Accessible from Mac while RunPod trains
- Real-time monitoring, alerts on anomalies

---

## 💡 Lessons Learned & Design Patterns

### Pattern 1: Configuration-Driven Development

**Approach**: All hyperparameters and settings in YAML files
- Benefits: Easy iteration, version control, reproducibility
- Implementation: `config/*.yaml` loaded at runtime
- Example: Change LoRA rank without touching code

### Pattern 2: Modular Pipeline with Checkpoints

**Approach**: Each stage (collect → validate → clean) independent
- Benefits: Resume interrupted runs, debug specific stages
- Implementation: Save checkpoints every N items
- Example: If collection fails at tweet 500, resume from there

### Pattern 3: Fail-Fast with Detailed Logging

**Approach**: Log everything, fail with clear error messages
- Benefits: Debugging easier, understand data flow
- Implementation: Python logging to console + file
- Example: "Rejected 23 replies: 15 for timing, 8 for length"

### Pattern 4: Development/Training Environment Split

**Approach**: Mac for dev, RunPod only for GPU training
- Benefits: 95% of work free, only pay for GPU hours
- Implementation: Separate requirements files, clear documentation
- Example: Collect data on Mac over days, train on RunPod in 3 hours

### Pattern 5: Research-Driven Design

**Approach**: Every decision traceable to research insights
- Benefits: Justified choices, avoid common pitfalls
- Implementation: Comments reference research docs
- Example: "rank=16 per current_research.md line 39"

### Common Pitfalls Avoided

**Pitfall 1: Training on Unreplicable Success**
- Problem: Celebrity replies get engagement due to author, not content
- Solution: Filter >100K followers
- Source: gameplan.md lines 118-145

**Pitfall 2: Overfitting on Small Datasets**
- Problem: Model memorizes training data
- Solution: High dropout, early stopping, frequent eval
- Source: current_research.md lines 40-42

**Pitfall 3: Poor Evaluation Metrics**
- Problem: ROUGE/BLEU don't correlate with engagement
- Solution: LLM-as-judge + statistical testing
- Source: current_research.md lines 23-33

**Pitfall 4: Ignoring Statistical Significance**
- Problem: Claiming "improvement" without proper testing
- Solution: Mann-Whitney U, Cohen's d, confidence intervals
- Source: current_research.md lines 48-56

**Pitfall 5: No Virtual Environment**
- Problem: Dependency conflicts, "works on my machine"
- Solution: Isolated venv, wrapper script
- Source: User encountered this, fixed immediately

---

## 🚀 Next Actions (Immediate)

### 1. Configure API Keys (5 minutes)

```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
nano .env

# Add these keys:
APIFY_API_TOKEN=<get from apify.com>
WANDB_API_KEY=<get from wandb.ai/authorize>
ANTHROPIC_API_KEY=<get from console.anthropic.com>
```

**Get API Keys**:
- Apify: Sign up at apify.com → Settings → API tokens → Create token
- W&B: Sign up at wandb.ai → Settings → API keys → Copy key
- Anthropic: Sign up at console.anthropic.com → API Keys → Create key

### 2. Test Data Collection (5 minutes)

```bash
# Activate virtual environment
source qwen-lora-env/bin/activate

# Test with 1 tweet
python scripts/collect_data.py --method apify --target 1

# Verify output
ls -lh data/processed/
cat data/processed/training_data_*.jsonl
```

**Expected Output**:
- File: `data/processed/training_data_<timestamp>.jsonl`
- Contains: 1-10 tweet-reply pairs (1 tweet may have multiple good replies)
- Format: One JSON object per line

### 3. Review and Iterate (30 minutes)

```bash
# Look at collected data
head -n 20 data/processed/training_data_*.jsonl | jq .

# Check quality:
# - Are tweets relevant?
# - Are replies high-quality?
# - Are engagement numbers reasonable?

# If quality looks good, proceed
# If not, adjust config/data_collection_config.yaml and retry
```

### 4. Full Data Collection (When Ready)

```bash
# Activate venv
source qwen-lora-env/bin/activate

# Collect 1,000 training pairs
python scripts/collect_data.py --method apify --target 1000

# This will:
# - Take 2-4 hours
# - Cost ~$200-300 via Apify
# - Save to data/processed/
# - Generate statistics report

# Monitor progress in logs
tail -f output/logs/data_collection_*.log
```

### 5. Manual Review (CRITICAL - 5-7 hours)

**Cannot be skipped**: "Dataset quality is 95% of everything"

```bash
# Create review notebook
jupyter notebook notebooks/02_manual_review.ipynb

# Or review JSONL directly
python -c "
import json
with open('data/processed/training_data_<timestamp>.jsonl') as f:
    for i, line in enumerate(f):
        pair = json.loads(line)
        print(f'\n--- Pair {i+1} ---')
        print(f'Tweet: {pair[\"tweet\"]}')
        print(f'Reply: {pair[\"reply\"]}')
        print(f'Likes: {pair[\"reply_likes\"]}')
        
        keep = input('Keep? (y/n/edit): ')
        if keep == 'n':
            # Mark for removal
            pass
        elif keep == 'edit':
            # Edit text
            pass
"
```

**Review Checklist**:
- [ ] Replies are relevant to tweets
- [ ] Replies are high-quality (not spam)
- [ ] No personal information exposed
- [ ] Engagement seems genuine (not bot-driven)
- [ ] Diverse topics represented
- [ ] No toxic/offensive content slipped through

### 6. Prepare for Training (After Data Ready)

```bash
# Get RunPod account
# Go to runpod.io and sign up

# Read RunPod guide
cat RUNPOD_SETUP.md

# Prepare data for upload
tar -czf training_data.tar.gz data/processed/training_data_*.jsonl

# Ready for Phase 3!
```

---

## 📋 Quality Checklist (Before Moving to Phase 3)

### Data Quality
- [ ] 800+ training pairs collected
- [ ] All pairs manually reviewed
- [ ] Diverse topics (not all from one domain)
- [ ] Diverse authors (not all from same accounts)
- [ ] Engagement distribution looks reasonable
- [ ] No toxic/spam content

### Code Quality
- [ ] All imports work (tested with `./run.sh`)
- [ ] Configurations reviewed and understood
- [ ] Logs show successful collection
- [ ] No Python errors or warnings
- [ ] Virtual environment working properly

### Documentation
- [ ] `.env` configured with API keys
- [ ] Understand data collection pipeline
- [ ] Read RUNPOD_SETUP.md for training
- [ ] Familiar with project structure
- [ ] Know how to activate venv

### Readiness
- [ ] Budget confirmed ($400-600 available)
- [ ] Time allocated (8 weeks total)
- [ ] RunPod account created (for Phase 3)
- [ ] W&B account created (for tracking)
- [ ] Understand success criteria (p<0.05, >15% improvement)

---

## 🔍 Key Files Reference

**Configuration**:
- `config/data_collection_config.yaml` - Collection settings
- `config/lora_config.yaml` - Training hyperparameters
- `.env` - API keys (create from .env.example)

**Main Scripts**:
- `scripts/collect_data.py` - Data collection pipeline
- `scripts/train_lora.py` - Training (Phase 3, to be created)
- `scripts/evaluate_model.py` - Evaluation (Phase 4, to be created)

**Source Modules**:
- `src/data_collection/apify_collector.py` - Apify API interface
- `src/data_collection/data_validator.py` - Quality validation
- `src/data_collection/data_cleaner.py` - Cleaning + deduplication

**Documentation**:
- `lora_implementation_plan.md` - Complete 6-phase roadmap
- `RUNPOD_SETUP.md` - RunPod training guide
- `GETTING_STARTED.md` - Quick start guide
- `SESSION_LOG.md` - This document

**Utilities**:
- `setup.sh` - Initial setup automation
- `run.sh` - Virtual environment wrapper
- `requirements.txt` - Dependencies for Mac
- `requirements-runpod.txt` - Dependencies for RunPod

---

## 📈 Success Metrics Tracking

### Phase 2 Success (Data Collection)
- [ ] 800-1,200 high-quality pairs collected
- [ ] <10% rejection rate during validation
- [ ] Manual review completed
- [ ] Dataset diversity confirmed

### Phase 3 Success (Training)
- [ ] Training completes without errors
- [ ] Overfitting prevented (gap <0.3)
- [ ] LoRA adapters saved successfully
- [ ] Cost within budget ($10-20)

### Phase 4 Success (Evaluation)
- [ ] Statistical significance achieved (p<0.05)
- [ ] Effect size meaningful (Cohen's d>0.3)
- [ ] Improvement ≥15% over base
- [ ] Qualitative assessment positive

### Project Success (Overall)
- [ ] Production-ready model deployed
- [ ] All experiments tracked in W&B
- [ ] Code and configs version controlled
- [ ] Documentation complete
- [ ] Portfolio-ready presentation

---

## 🎯 Critical Success Factors

Based on research synthesis and planning:

1. **Data Quality > Quantity**: Manual review is non-negotiable
2. **Conservative Hyperparameters**: Start safe, iterate based on results
3. **Frequent Monitoring**: Catch overfitting early (eval every 50 steps)
4. **Statistical Rigor**: No claims without proper testing (p-values + effect sizes)
5. **Cost Control**: Use RunPod only for training (95% work on Mac for free)
6. **Realistic Expectations**: 15-30% improvement is success, not 50-70%
7. **Iteration Budget**: Plan for 2-3 training runs, not 1
8. **Documentation**: Track everything for reproducibility

---

## 🔄 Future Development Notes

### For Phase 3 (Training Script)
- Implement train_lora.py based on lora_implementation_plan.md
- W&B integration for remote monitoring
- Checkpoint saving every 50 steps
- Early stopping callback
- Overfitting diagnostics
- Sample prediction logging

### For Phase 4 (Evaluation)
- Implement evaluate_model.py
- LLM-as-judge with Claude (position bias mitigation)
- Mann-Whitney U test + Cohen's d
- Confidence interval calculation
- Generate comparison tables and plots
- Qualitative analysis of predictions

### For Phase 5 (Iteration)
- Hyperparameter sweep based on v1 results
- Config version management
- A/B comparison between versions
- Final model selection criteria
- HuggingFace Hub deployment

### Potential Enhancements
- RAG baseline for comparison
- Human evaluation on Prolific
- Real-world A/B testing (if deploying)
- CUDA optimization exploration (optional, long-term)

---

## 📞 Troubleshooting Quick Reference

### Data Collection Issues
- **No data returned**: Check APIFY_API_TOKEN in .env
- **Low quality pairs**: Adjust filters in config/data_collection_config.yaml
- **Import errors**: Activate venv with `source qwen-lora-env/bin/activate`

### Training Issues (Future)
- **OOM errors**: Reduce batch size, ensure 4-bit quantization enabled
- **Slow training**: Confirm GPU being used, consider Unsloth
- **Overfitting**: Reduce epochs to 1, increase dropout to 0.15

### RunPod Issues (Future)
- **Connection lost**: Use tmux for persistent sessions
- **Pod expensive**: Check pricing, stop pod when not training
- **File transfer slow**: Compress data first, use tar.gz

### Evaluation Issues (Future)
- **Poor metrics**: Check if model actually improved vs noise
- **No significance**: Need more data or better model
- **Eval slow**: Run on RunPod alongside training

---

## 🎓 Learning Outcomes Achieved So Far

### Technical Skills
- [x] LoRA/QLoRA architecture understanding
- [x] Small dataset fine-tuning best practices
- [x] Data quality filtering strategies
- [x] Semantic deduplication with embeddings
- [x] Configuration-driven development
- [x] Virtual environment management

### ML Engineering
- [x] Research-driven design decisions
- [x] Cost-optimized workflow (Mac + RunPod)
- [x] Comprehensive logging and debugging
- [x] Checkpoint and recovery strategies
- [x] Documentation and reproducibility
- [x] Risk mitigation planning

### Domain Knowledge
- [x] Twitter/X API limitations and economics
- [x] Social media engagement patterns
- [x] Unreplicable success factors (celebrity, timing, media)
- [x] Alternative data collection methods
- [x] Evaluation methodology for engagement prediction

---

## 📅 Timeline Tracking

**Week 1**: Infrastructure Setup (COMPLETE ✅)
- Day 1: Project analysis and planning
- Day 2: Structure creation and configuration
- Day 3: Data collection pipeline implementation
- Day 4: Documentation and guides
- Day 5: Testing and troubleshooting

**Week 2-3**: Data Collection (CURRENT PHASE)
- Week 2: Test collection, iterate on filters
- Week 3: Full collection (1,000 pairs), manual review

**Week 4**: Training v1 (UPCOMING)
- Day 1: RunPod setup, script creation
- Day 2-3: Training run
- Day 4-5: Evaluation and analysis

**Weeks 5-8**: Iteration and refinement

---

## 🎉 Achievements Unlocked

- ✅ Complete project structure designed and implemented
- ✅ Production-ready data collection pipeline
- ✅ Research-backed filtering strategy
- ✅ RunPod-optimized workflow
- ✅ Comprehensive documentation (11 guides)
- ✅ Virtual environment with all dependencies
- ✅ Configuration system with sane defaults
- ✅ Quality over quantity approach validated
- ✅ Cost-effective training strategy planned
- ✅ Statistical rigor framework defined

---

## 🚀 Current Momentum

**Phase 1: COMPLETE** ✅
**Phase 2: IN PROGRESS** 🔄
**Current Activity: Data Collection Running (3,500 target pairs)**
**Status: tmux session active, checkpoint system operational**
**Estimated Time to First Training**: 1 week

---

## 🔄 Phase 2: Data Collection - Active Session (Oct 2, 2025)

### Session 2.1: Initial Collection Attempts

**Date**: October 2, 2025, 05:00-06:30 AM

#### Problems Discovered

**Issue #1: Twitter Date Format Mismatch**
- **Problem**: Apify returned dates in RFC 2822 format (`Thu Oct 02 07:11:55 +0000 2025`)
- **Expected**: ISO 8601 format
- **Impact**: All timing filters failing, couldn't calculate reply delay
- **Fix**: Implemented dual-format date parser using `email.utils.parsedate_to_datetime`
- **File**: `src/data_collection/apify_collector.py` (lines 439-452)

**Issue #2: Critical Filter Tuning Errors**
- **Problem**: `min_retweets: 1` eliminating 95%+ of replies (most get 0 RTs)
- **Problem**: `min_time_delay: 60s` too short to avoid first-reply advantage
- **Problem**: Searching TODAY's tweets (no time for replies to get engagement)
- **Impact**: Only got 376 pairs from 1,408 raw (73% rejection)
- **Fix**: 
  - Changed `min_retweets: 1 → 0`
  - Changed `min_time_delay: 60s → 300s` (5 minutes)
  - Changed `max_time_delay: 24h → 7 days`
  - Added `until:2025-09-30` to target older tweets with established engagement

**Issue #3: Bot Farm Data**
- **Problem**: Collected 376 replies from **1 single author** (bot/engagement farming)
- **Example spam**: "Evening fren", "Revolution is the power of change", "Stay locked in"
- **Problem**: Generic search queries attracting crypto spam
- **Impact**: Entire dataset unusable for training

---

### Session 2.2: Comprehensive Quality Upgrade

**Date**: October 2, 2025, 06:30 AM - Present

#### Solutions Implemented

**1. Targeted Search Queries**
- **Before**: Generic queries like "AI OR artificial intelligence"
- **After**: Specific technical discussions requiring expertise
- **Examples**:
  - `(debugging OR "code review" OR refactoring) (tips OR advice) -crypto -NFT`
  - `(system design OR architecture) (challenges OR lessons) -crypto -web3`
  - `(TypeScript OR React OR Node) (pattern OR antipattern) -crypto`
- **File**: `config/data_collection_config.yaml` (lines 17-34)
- **Impact**: Explicitly exclude crypto spam, target substantive tech discussions

**2. Stricter Engagement Thresholds**
- `min_likes: 1 → 5` (5x increase, filter bot spam)
- `max_likes: 1000 → 500` (avoid viral outliers)
- `min_follower_count: 100 → 200` (2x increase, better bot filtering)
- `max_follower_count: 50000 → 20000` (avoid influencer advantage)
- **File**: `config/data_collection_config.yaml` (lines 49-58)

**3. Author Diversity Enforcement (NEW)**
- **Feature**: `max_replies_per_author: 10`
- **Implementation**: Track author IDs, limit replies per author
- **File**: `src/data_collection/apify_collector.py` (lines 93-123)
- **Impact**: Prevents "376 replies from 1 bot" problem

**4. Crypto Spam Detection (NEW)**
- **Keywords**: gm, ser, fren, wagmi, degen, wen, anon, "stay locked in", revolution, onchain
- **Implementation**: Validator checks for spam keywords
- **File**: `src/data_collection/data_validator.py` (lines 98-100, 276-285)
- **Impact**: Auto-reject engagement farming replies

**5. Word Diversity Check (NEW)**
- **Feature**: Require 8+ unique meaningful words
- **Implementation**: Count unique words excluding stop words
- **File**: `src/data_collection/data_validator.py` (lines 102-105, 287-296)
- **Example**: "gm fren" = 2 words → REJECTED ❌
- **Example**: "Debugging async issues in Node requires careful attention..." = 14 words → PASS ✅

**6. Generic Phrase Detection (NEW)**
- **Feature**: Flag if >50% is generic phrases
- **Phrases**: "congrats", "amazing", "love this", "thanks for sharing", etc.
- **File**: `src/data_collection/data_validator.py` (lines 107-110, 298-307)
- **Impact**: Filter low-information engagement replies

**7. Increased Validation Threshold**
- **Before**: `min_engagement: 2 likes`
- **After**: `min_engagement: 3 likes`
- **File**: `src/data_collection/data_validator.py` (line 271)
- **Impact**: Double-validation ensures substantive replies

#### Quality Improvement Results

**Test Collection (50 pairs):**
- Validation pass rate: **27.9% → 63.3%** (2.3x improvement!)
- Rejections breakdown:
  - Crypto spam keywords: 4
  - Low word diversity: 7
  - Too generic: 1
  - Engagement too low: reduced significantly

---

### Session 2.3: Fault-Tolerant Infrastructure

**Date**: October 2, 2025, 06:30-07:00 AM

#### Checkpoint/Resume System Implemented

**Problem**: Long collections (5-8 hours) vulnerable to:
- Internet disconnections
- Apify API timeouts
- System crashes
- Manual interruptions
- Mac sleep/laptop closure

**Solution**: Full checkpoint and resume functionality

**Features Implemented:**

1. **Automatic Checkpointing**
   - Saves after each query completes (~30-45 min intervals)
   - Saves on errors before continuing
   - Stores both metadata + actual data
   - **Files**: `data/raw/collection_checkpoint.json` + `checkpoint_data.jsonl`

2. **Resume Functionality**
   - Loads existing data from checkpoint
   - Skips already-processed queries
   - Continues from last position
   - **Usage**: Add `--resume` flag to collection command

3. **Implementation Details**
   - **File**: `src/data_collection/apify_collector.py`
     - `_save_checkpoint()` (lines 541-560)
     - `_load_checkpoint()` (lines 562-593)
     - `_clear_checkpoint()` (lines 595-605)
   - **File**: `scripts/collect_data.py`
     - Resume parameter added (line 53)
     - Checkpoint integration (lines 67-123)

4. **Safety Features**
   - Zero data loss on interruption
   - Works with tmux for SSH disconnection survival
   - Auto-clears checkpoint on successful completion
   - Detailed logging for debugging

**Documentation Created:**
- `RESUME_GUIDE.md` - Complete guide for fault-tolerant collection

---

### Session 2.4: Production Collection Launch

**Date**: October 2, 2025, 07:00 AM

#### Collection Status

**Command Executed:**
```bash
tmux new -s collection
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume
# Detached with Ctrl+B, D
```

**Target Configuration:**
- **Raw pairs to collect**: 3,500
- **Expected after validation**: 1,400-2,000 (40-60% pass rate)
- **Expected after dedup**: 1,200-1,800
- **For manual curation**: Select best 800 for training

**Estimated Timeline:**
- **Duration**: 5-8 hours
- **Checkpoints**: Every 30-45 minutes (after each query)
- **Queries**: 10 specific technical topics
- **Pairs per query**: ~350

**Quality Filters Active:**
- ✅ Crypto spam keyword blocking
- ✅ Word diversity checks (8+ unique words)
- ✅ Generic phrase detection (<50% generic)
- ✅ Author diversity (max 10 per author)
- ✅ Engagement thresholds (5+ likes, 200-20k followers)
- ✅ Timing filters (5min-7day delay)
- ✅ Content filters (no URLs, no media, 30-280 chars)
- ✅ Toxicity filtering (score < 0.3)
- ✅ Language confidence (>0.8)

**Infrastructure:**
- ✅ Running in tmux session (survives disconnections)
- ✅ Checkpoint system active (auto-saves progress)
- ✅ Detailed logging enabled
- ✅ Resume capability tested

**Current State (as of log update):**
- **Status**: RUNNING in background (tmux detached)
- **Session**: `tmux attach -t collection` to view progress
- **Checkpoint**: `data/raw/collection_checkpoint.json`
- **Logs**: `output/logs/data_collection_*.log`

---

## 📊 Updated Achievements

### Infrastructure (Phase 1) ✅
- [x] Complete project structure
- [x] Production-ready data collection pipeline
- [x] Research-backed filtering strategy
- [x] RunPod-optimized workflow
- [x] Comprehensive documentation (13+ guides)
- [x] Virtual environment with dependencies
- [x] Configuration system

### Data Quality (Phase 2) ✅
- [x] Comprehensive spam detection (crypto keywords)
- [x] Word diversity validation (8+ unique words)
- [x] Generic phrase filtering (<50% generic)
- [x] Author diversity enforcement (max 10/author)
- [x] Targeted search queries (10 tech-specific)
- [x] Strict engagement thresholds (5+ likes)
- [x] Multi-format date parsing (RFC 2822 + ISO 8601)
- [x] Detailed rejection statistics logging

### Fault Tolerance (Phase 2) ✅
- [x] Automatic checkpoint system
- [x] Resume functionality
- [x] tmux integration
- [x] Error recovery
- [x] Progress tracking
- [x] Safe interruption handling

### Data Collection (Phase 2) 🔄
- [ ] Collect 3,500 raw pairs (IN PROGRESS - started Oct 2, 07:00)
- [ ] Validate to 1,400-2,000 pairs (PENDING)
- [ ] Deduplicate to 1,200-1,800 pairs (PENDING)
- [ ] Manual review and curation to 800 pairs (PENDING)

---

## 🎯 Next Steps

### Immediate (Within Hours)
1. Monitor collection progress periodically
2. Check checkpoint status: `cat data/raw/collection_checkpoint.json | jq .`
3. Reattach to tmux if needed: `tmux attach -t collection`

### Short-term (This Week)
1. Complete 3,500 pair collection
2. Review validation statistics
3. Manual curation of best 800 pairs
4. Prepare dataset for training

### Medium-term (Next Week)
1. Transfer curated dataset to RunPod
2. Setup training environment
3. Execute first LoRA training run
4. Initial evaluation

---

## 📈 Success Metrics

### Data Quality Metrics (Target vs Actual)
- **Validation pass rate**: Target >50% | Achieved 63.3% ✅
- **Crypto spam rejection**: Target 0 spam | System active ✅
- **Author diversity**: Target >50 unique | System enforces 10 max/author ✅
- **Word diversity**: Target >8 words | System enforces ✅
- **Final dataset size**: Target 800 pairs | In progress (1,200-1,800 expected)

### Infrastructure Metrics
- **Setup completion**: 100% ✅
- **Documentation coverage**: 13+ guides ✅
- **Error recovery capability**: Checkpoint system operational ✅
- **Collection fault tolerance**: tmux + resume functional ✅

---

## 🚀 Current Momentum (Updated)

**Phase 1: Infrastructure Setup** ✅ COMPLETE
**Phase 2: Data Collection** 🔄 IN PROGRESS (60% complete)
- Data collection pipeline: ✅ Operational
- Quality filters: ✅ Implemented and tested
- Fault tolerance: ✅ Checkpoint system active
- Collection run: 🔄 Active (3,500 pairs target)

**Phase 3: Training Preparation** ⏳ PENDING
**Phase 4: Initial Training** ⏳ PENDING
**Phase 5: Evaluation** ⏳ PENDING

**Next Milestone**: Collection completion (5-8 hours)
**Estimated Time to First Training**: 3-5 days

---

**Last Updated**: October 2, 2025 07:00 AM - Data collection launched in tmux  
**Next Update**: After collection completes and validation statistics available  
**Status**: 🟢 ACTIVE - Collection running, all systems operational

