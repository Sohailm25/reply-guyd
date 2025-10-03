# Qwen3-8B LoRA Fine-Tuning for Twitter Replies

## Overview

This project implements Parameter-Efficient Fine-Tuning (PEFT) of Qwen3-8B (8.2B parameters) using Low-Rank Adaptation (LoRA) to generate high-engagement Twitter replies. The implementation prioritizes dataset quality over quantity, rigorous evaluation methodology, and statistical validation of results.

**Key Characteristics:**
- Small, curated dataset approach (800-1,200 pairs vs typical 10,000+)
- Multi-tier evaluation including LLM-as-judge and statistical significance testing
- Full experiment tracking and reproducibility
- Realistic expectations based on research synthesis (15-30% improvement target)

## Methodology

### Model Configuration

**Base Model:** Qwen3-8B (Qwen3ForCausalLM)
- 8.2B total parameters (6.95B non-embedding)
- 36 transformer layers, GQA attention (32 Q heads, 8 KV heads)
- 32,768 token context window (131,072 with YaRN)
- bfloat16 precision

**LoRA Configuration:**
- Rank: 16 (conservative for small dataset)
- Alpha: 16 (matching rank for 100% scaling)
- Dropout: 0.1 (regularization)
- Target modules: All 7 linear layers per block (252 total targets)
  - Attention: q_proj, k_proj, v_proj, o_proj
  - MLP: gate_proj, up_proj, down_proj
- Training: 2 epochs, learning rate 2e-4, effective batch size 16

**QLoRA Optimizations:**
- 4-bit NF4 quantization with double quantization
- Memory reduction: 16GB → 4-6GB
- Compute dtype: bfloat16 for numerical stability

### Dataset Construction

**Target:** 800-1,200 meticulously curated tweet-reply pairs

**Collection Method:** Apify Twitter scraper (primary) or twscrape (fallback)

**Quality Filters (from gameplan.md analysis):**

Engagement thresholds:
- Original tweets: 200-2,000 likes
- Replies: 5-500 likes
- Author followers: 200-20,000 (excludes bots and celebrities)

Timing constraints:
- Minimum 5 minutes post-tweet (avoids first-reply advantage)
- Maximum 7 days post-tweet (ensures engagement has stabilized)

Content requirements:
- Text-only (no media or URLs that could drive engagement)
- Length: 30-280 characters
- English language (confidence > 0.8)
- Non-toxic (Detoxify score < 0.3)
- Semantic uniqueness (similarity < 0.9)

Spam detection:
- Crypto keyword filtering (15+ terms)
- Word diversity requirement (8+ unique words)
- Generic phrase detection (< 50% generic content)
- Author diversity enforcement (max 10 replies per author)

**Rationale:** These filters remove unreplicable success factors (celebrity status, perfect timing, media content) to focus on text-based patterns the model can learn.

## Evaluation Framework

### Tier 1: Automated Metrics (Baseline Only)

Standard NLP metrics for sanity checking:
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU score
- Perplexity

Note: Research shows these correlate poorly with engagement. Used only to detect catastrophic failures.

### Tier 2: LLM-as-Judge (Primary Evaluation)

**Protocol:**
- Model: Claude 3.5 Sonnet
- Method: Pairwise comparison with chain-of-thought reasoning
- Bias mitigation: Position swapping for each comparison
- Output: Win rate percentage

**Advantages:**
- Correlates better with human judgment than automated metrics
- Cost-effective compared to human evaluation
- Reproducible and scalable

### Tier 3: Statistical Validation (Critical)

**Tests:**
- Mann-Whitney U (non-parametric, robust to outliers)
- Cohen's d (effect size calculation)
- Confidence intervals (95%)

**Success Criteria (all must be met):**
- p-value < 0.05 (statistical significance)
- Improvement > 15% (practical significance)
- Cohen's d > 0.3 (meaningful effect size)

### Tier 4: Human Evaluation (Optional)

Platform: Prolific (if budget permits)
- Sample size: 150 participants
- Cost: ~$375
- Only pursued if Tier 2-3 results are promising

## Implementation Details

### Data Collection Pipeline

**Primary Method: Apify**
```bash
python scripts/collect_data.py --method apify --target 3500 --resume
```

Characteristics:
- Cost: $200-300 for sufficient raw data
- Reliability: High, handles rate limits automatically
- Checkpoint system: Fault-tolerant, can resume after interruptions
- Expected yield: 40-60% pass validation filters

**Fallback Method: twscrape**

Requires:
- 3-10 Twitter accounts configured
- Proxy rotation setup
- Higher maintenance, lower cost at scale

### Training Execution

**Environment:** RunPod RTX A6000 (48GB VRAM)
- Cost: $0.39-0.79/hour
- Duration: 2-6 hours per training run
- Total compute cost: $5-20

**Monitoring:**
- Real-time loss curves via Weights & Biases
- Gradient norm tracking
- Train/eval gap monitoring (alert if > 0.3)
- Sample generation every 50 steps

**Overfitting Prevention:**
- Early stopping (patience=3 evaluations)
- Frequent evaluation on hold-out set
- High dropout rate (0.1, adjustable to 0.15)
- Conservative epoch count (2, reducible to 1)

### Hyperparameter Iteration Strategy

**If overfitting observed:**
- Reduce rank: 16 → 8
- Increase dropout: 0.1 → 0.15
- Reduce epochs: 2 → 1

**If underfitting observed:**
- Increase rank: 16 → 32
- Increase epochs: 2 → 3
- Verify data quality (manual review subset)

**If style drift observed:**
- Collect more diverse training examples
- Adjust generation temperature
- Review filter criteria for over-restriction

## Project Structure

```
Qwen3-8/
├── config/
│   ├── data_collection_config.yaml    # Filter thresholds, search queries
│   └── lora_config.yaml                # Training hyperparameters
├── data/
│   ├── raw/                            # Unprocessed collected data
│   └── processed/                      # Filtered, validated datasets
├── src/
│   ├── data_collection/
│   │   ├── apify_collector.py         # Apify API integration
│   │   ├── scraper_collector.py       # twscrape fallback
│   │   ├── data_validator.py          # Quality filters
│   │   └── data_cleaner.py            # Deduplication
│   ├── training/                       # LoRA training modules (Phase 3)
│   ├── evaluation/                     # Multi-tier eval (Phase 4)
│   └── utils/                          # Shared utilities
├── scripts/
│   ├── collect_data.py                 # Data collection orchestration
│   ├── train_lora.py                   # Training execution (Phase 3)
│   └── evaluate_model.py               # Evaluation suite (Phase 4)
├── output/
│   ├── models/                         # Trained LoRA adapters
│   ├── logs/                           # Training and collection logs
│   └── figures/                        # Evaluation visualizations
└── requirements.txt
```

See `PROJECT_STRUCTURE.md` for detailed documentation.

## Installation

### Prerequisites

**Development (Mac):**
- Python 3.8+
- 20GB disk space
- No GPU required (data collection and evaluation only)

**Training (RunPod):**
- RunPod account (runpod.io)
- RTX A6000 (48GB VRAM) recommended
- See `RUNPOD_SETUP.md` for complete setup guide

### Setup

```bash
# 1. Navigate to repository
cd Qwen3-8

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Configure environment
cp .env.example .env
# Add API keys: APIFY_API_TOKEN, ANTHROPIC_API_KEY, WANDB_API_KEY
```

The setup script creates a Python virtual environment and installs all dependencies from `requirements.txt`.

## Usage

### Phase 1: Data Collection

```bash
# Collect raw data (target 3,500 for ~1,200-1,800 after filtering)
./run.sh python scripts/collect_data.py --method apify --target 3500 --resume

# The --resume flag enables checkpoint recovery for fault tolerance
```

**Checkpoint System:**
- Saves progress every query (~30-45 minutes)
- Survives internet disconnections, API timeouts, system crashes
- Run with --resume flag to continue from last checkpoint
- See `RESUME_GUIDE.md` for details

**Expected Timeline:**
- Raw collection: 5-8 hours
- Validation pass rate: 40-60%
- Manual review: 2-3 days
- Final curated dataset: 800-1,200 pairs

### Phase 2: Training

```bash
# Transfer curated dataset to RunPod
# See RUNPOD_SETUP.md for complete instructions

# Execute training
python scripts/train_lora.py --config config/lora_config.yaml

# Monitor in Weights & Biases dashboard
```

### Phase 3: Evaluation

```bash
# Run multi-tier evaluation suite
python scripts/evaluate_model.py --model output/models/lora-v1

# Generates:
# - Automated metrics (ROUGE, BLEU)
# - LLM-as-judge comparisons
# - Statistical significance tests
# - Visualizations and report
```

## Configuration

### Data Collection Settings

File: `config/data_collection_config.yaml`

```yaml
collection:
  target_pairs: 1000
  search_queries:
    # Specific technical topics to avoid crypto spam
    - "(debugging OR \"code review\") (tips OR advice) -crypto -NFT"
    - "(system design OR architecture) (challenges OR lessons) -crypto"
    # ... 8 more targeted queries

  tweet_filters:
    min_likes: 100
    max_likes: 2000
    min_retweets: 10
    is_retweet: false

  reply_filters:
    min_likes: 5
    max_likes: 500
    min_follower_count: 200
    max_follower_count: 20000
    min_time_delay_seconds: 300      # 5 minutes
    max_time_delay_seconds: 604800   # 7 days
    has_media: false
    has_urls: false
    max_replies_per_author: 10       # Author diversity

  quality:
    min_language_confidence: 0.8
    max_toxicity_score: 0.3
    min_unique_words: 8
    max_generic_phrase_ratio: 0.5
    semantic_similarity_threshold: 0.9
    crypto_spam_keywords: [gm, fren, wagmi, ser, degen, ...]
```

### Training Settings

File: `config/lora_config.yaml`

```yaml
lora:
  rank: 16
  alpha: 16
  dropout: 0.1
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  num_epochs: 2
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  eval_steps: 50
  save_steps: 50
  logging_steps: 10

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
```

## Budget Estimate

| Component | Optimistic | Realistic | Conservative |
|-----------|------------|-----------|--------------|
| Data Collection (Apify) | $200 | $300 | $400 |
| Training (RunPod A6000) | $5 | $10 | $20 |
| Evaluation (Claude API) | $50 | $100 | $150 |
| Human Evaluation (optional) | $0 | $0 | $390 |
| **Total** | **$255** | **$410** | **$960** |

Realistic target: $400-600

## Known Limitations

Based on research synthesis (see `gameplan.md` for detailed analysis):

1. **First attempt success rate:** ~30% probability. Plan for 2-3 training iterations.

2. **Dataset size constraint:** 800-1,200 pairs is below ideal (2,500+), but research shows viability with aggressive curation.

3. **Realistic improvement expectation:** 15-30% over base model. Claims of 50-70% improvement are typically cherry-picked examples.

4. **Timing advantage unsolved:** Model improves reply quality, not response speed. Early-reply advantage remains for real deployment.

5. **Offline evaluation ≠ online performance:** Only A/B testing in production shows true engagement impact.

6. **Overfitting risk:** Small datasets require careful monitoring and regularization.

## Experiment Tracking

All training runs logged to Weights & Biases:

**Tracked Metrics:**
- Training and validation loss curves
- Learning rate schedule
- Gradient norms
- All hyperparameters
- Sample generations at checkpoints
- Evaluation results (all tiers)
- Model artifacts (LoRA adapters)

**Dashboard:** `wandb.ai/{username}/qwen3-twitter-lora`

## Technical Implementation Notes

### LoRA Target Selection

All 7 linear projection layers per transformer block are targeted (252 total):
- **Attention projections:** q_proj, k_proj, v_proj, o_proj (4 layers)
- **MLP projections:** gate_proj, up_proj, down_proj (3 layers)

This comprehensive targeting is feasible due to LoRA's parameter efficiency (< 1% of base model parameters).

### Memory Optimization

QLoRA 4-bit quantization reduces memory footprint:
- Base model (bfloat16): ~16GB
- Quantized model (NF4): ~4GB
- LoRA parameters: ~150MB (rank 16)
- Activation memory: ~2GB
- **Total:** ~6-7GB, fits comfortably in 48GB VRAM

### Statistical Power Analysis

For target effect size (Cohen's d = 0.3) with alpha = 0.05 and power = 0.8:
- Minimum sample size: ~350 comparisons per group
- Planned: 800-1,200 pairs provides sufficient power
- Mann-Whitney U test is robust to non-normal distributions

## Documentation

- **[Implementation Plan](lora_implementation_plan.md):** Detailed 6-phase roadmap with research-backed methodology
- **[Project Structure](PROJECT_STRUCTURE.md):** Directory organization and file conventions
- **[Research Synthesis](current_research.md):** Literature review and best practices
- **[Critical Analysis](gameplan.md):** Realistic expectations and risk assessment
- **[Resume Guide](RESUME_GUIDE.md):** Fault-tolerant data collection workflow
- **[RunPod Setup](RUNPOD_SETUP.md):** Training environment configuration
- **[Data Quality](DATA_QUALITY_IMPROVEMENTS.md):** Filter implementation details

## Contributing Guidelines

This is a learning project with emphasis on rigorous methodology:

**Principles:**
- Reproducibility: All experiments version controlled, configs saved
- Transparency: Document failures as well as successes  
- Statistical rigor: Report p-values AND effect sizes, never cherry-pick results
- Quality over speed: Manual review is non-negotiable for small datasets

## License

Apache 2.0 (inherited from Qwen3 base model)

## Acknowledgments

**Research Foundation:**
- Research synthesis incorporates insights from 50+ papers and practitioner reports
- See `current_research.md` for detailed citations

**Software Stack:**
- Qwen Team: Base model
- HuggingFace: PEFT and Transformers libraries
- Apify: Data collection infrastructure
- Weights & Biases: Experiment tracking

## Support and Troubleshooting

**For implementation issues:**
1. Check `PROJECT_STRUCTURE.md` for file organization
2. Review `lora_implementation_plan.md` for methodology
3. Consult `gameplan.md` for realistic expectations
4. See `RESUME_GUIDE.md` for data collection recovery

**For research questions:**
- Refer to `current_research.md` for evidence base
- Check `gameplan.md` for critical analysis

---

**Project Status:** Phase 2 (Data Collection) - In Progress

**Timeline:** 8 weeks to production-ready model with evaluation

**Current Activity:** Collecting 3,500 raw pairs for curation to 800-1,200 training examples
