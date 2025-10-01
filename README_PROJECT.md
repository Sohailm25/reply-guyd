# Qwen3-8B LoRA Fine-Tuning for Twitter Replies

Fine-tuning Qwen3-8B using LoRA to generate high-engagement Twitter replies. A comprehensive implementation with rigorous evaluation and statistical validation.

## 🎯 Project Goals

- **Fine-tune** Qwen3-8B (8.2B parameters) using LoRA on 800-1,200 high-quality tweet-reply pairs
- **Achieve** statistically significant improvement (15-30%) over base model
- **Demonstrate** rigorous ML engineering: proper evaluation, statistical testing, experiment tracking
- **Create** portfolio-ready project with reproducible methodology

## 🔬 Methodology Highlights

- **LoRA Configuration**: Rank=16, all linear layers targeted (252 total), QLoRA 4-bit quantization
- **Data Quality**: Manual review of all examples, aggressive filtering, semantic deduplication
- **Evaluation**: Multi-tier approach with LLM-as-judge, automated metrics, and statistical significance testing
- **Reproducibility**: Full experiment tracking via Weights & Biases, all code version controlled

Based on extensive research synthesis (see `current_research.md` and `gameplan.md`).

## 📊 Expected Results

- **Realistic Improvement**: 15-30% better engagement than base model
- **Statistical Validation**: p-value < 0.05, Cohen's d > 0.3
- **Dataset**: 800-1,200 curated tweet-reply pairs
- **Training Cost**: $400-600 total (data collection + compute + evaluation)

## 🚀 Quick Start

### Prerequisites

**For Development (Mac):**
- Python 3.8+
- ~20GB disk space for model and data
- No GPU needed (data collection, evaluation)

**For Training (RunPod):**
- RunPod account ([runpod.io](https://runpod.io))
- RTX A6000 (48GB) recommended - $0.39-0.79/hr
- See `RUNPOD_SETUP.md` for complete guide

### Installation

```bash
# 1. Clone/navigate to repository
cd Qwen3-8

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys:
#   - APIFY_API_TOKEN (for data collection)
#   - ANTHROPIC_API_KEY (for evaluation)
#   - WANDB_API_KEY (for experiment tracking)
```

### Data Collection

```bash
# Collect 1,000 training pairs via Apify (~$200-300)
python scripts/collect_data.py --method apify --target 1000

# Review collected data (IMPORTANT!)
head -n 10 data/processed/training_data_*.jsonl
```

### Training

```bash
# Train LoRA adapters (coming in Phase 3)
python scripts/train_lora.py --config config/lora_config.yaml

# Monitor training in W&B dashboard
```

### Evaluation

```bash
# Evaluate fine-tuned vs base model (coming in Phase 4)
python scripts/evaluate_model.py --model output/models/lora-v1
```

## 📁 Project Structure

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed documentation.

```
Qwen3-8/
├── config/                      # YAML configuration files
├── data/                        # Raw and processed data
├── src/
│   ├── data_collection/         # Apify & scraping pipelines
│   ├── training/                # LoRA training modules
│   ├── evaluation/              # Evaluation framework
│   └── utils/                   # Shared utilities
├── scripts/                     # Executable Python scripts
├── output/                      # Models, logs, figures
├── requirements.txt
└── lora_implementation_plan.md  # Detailed roadmap
```

## 🔧 Configuration

### Data Collection (`config/data_collection_config.yaml`)

```yaml
collection:
  target_pairs: 1000
  
  # Tweet filters (from gameplan.md: 200-1,000 likes sweet spot)
  tweet_filters:
    min_likes: 200
    max_likes: 2000
  
  # Reply filters (avoid celebrity/timing/media advantages)
  reply_filters:
    min_likes: 5
    max_likes: 1000
    min_follower_count: 500      # Avoid bots
    max_follower_count: 100000   # Avoid celebrities
    min_time_delay_seconds: 300  # Avoid timing advantage
    has_media: false             # Can't replicate media success
```

### LoRA Training (`config/lora_config.yaml`)

```yaml
lora:
  rank: 16                  # Conservative for 800-1,200 examples
  alpha: 16                 # Keep alpha = rank
  dropout: 0.1              # Regularization for small dataset
  target_modules:           # ALL linear layers (252 total)
    - q_proj, k_proj, v_proj, o_proj
    - gate_proj, up_proj, down_proj

training:
  num_epochs: 2             # Reduce to 1 if overfitting
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16
```

## 📈 Data Collection Strategy

### Apify Method (Recommended)

- **Cost**: ~$0.25 per 1,000 tweets ($200-300 for 1,000 pairs)
- **Reliability**: High, handles rate limits automatically
- **Access**: Historical data beyond 7 days
- **Setup**: Simple, just need API token

```python
from src.data_collection import ApifyCollector

collector = ApifyCollector()
pairs = collector.collect_tweets_and_replies(
    search_query="AI lang:en min_faves:200",
    max_tweets=100,
    max_replies_per_tweet=30
)
```

### Scraping Method (Alternative)

- **Cost**: Cheaper at scale but requires Twitter accounts
- **Setup**: More complex (3-10 accounts, proxies)
- **Legal**: Violates ToS but legally defensible for public data
- **Tool**: twscrape (recommended from research)

See `src/data_collection/scraper_collector.py` for implementation.

## 🎯 Quality Filters

Based on `gameplan.md` analysis to avoid replicating unreplicable success:

### ❌ Filtered Out
- **Celebrity replies** (followers > 100K) - clout advantage
- **Too-early replies** (< 5 min) - timing advantage
- **Media-heavy replies** - can't replicate visual engagement
- **URL-based replies** - links drive engagement we can't match
- **Toxic content** (Detoxify score > 0.3)
- **Too short** (< 30 chars) or **too long** (> 280 chars)

### ✅ Kept
- Normal accounts (500-100K followers)
- Content-based success (5+ minutes after tweet)
- Text-only engagement
- English language (confidence > 0.8)
- Semantically unique (similarity < 0.9)

## 📊 Evaluation Framework

### Tier 1: Automated Metrics (Development)
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU score
- Perplexity

*Note*: Per research, these correlate poorly with engagement. Use as sanity checks only.

### Tier 2: LLM-as-Judge (Primary)
- Claude 3.5 Sonnet pairwise comparisons
- Position bias mitigation (swap positions)
- Chain-of-thought reasoning
- Win rate calculation

### Tier 3: Statistical Validation (Critical)
- Mann-Whitney U test (non-parametric)
- Cohen's d effect size
- Confidence intervals
- **Success criteria**: p < 0.05 AND improvement > 20% AND Cohen's d > 0.3

### Tier 4: Human Evaluation (Optional)
- Prolific platform (~$375 for 150 participants)
- Only if budget allows and earlier results promising

## 🔬 Training Details

### QLoRA Optimization
- **4-bit quantization** (NF4) reduces memory from 16GB → 4-6GB
- **Double quantization** for further compression
- **bfloat16 compute dtype** for numerical stability
- Enables 8B model training on consumer GPUs

### Overfitting Prevention
- **Early stopping** (patience=3 evaluations)
- **High dropout** (0.1, increase to 0.15 if needed)
- **Frequent evaluation** (every 50 steps for small dataset)
- **Train/eval gap monitoring** (alert if gap > 0.3)

### Hyperparameter Iteration
Training v1 → Analyze results → Adjust config → Training v2

Common adjustments:
- Overfitting: Reduce rank to 8, increase dropout, reduce epochs to 1
- Underfitting: Increase rank to 32, increase epochs to 3
- Style drift: Increase temperature, collect more diverse data

## 💰 Budget Breakdown

| Item | Optimistic | Realistic | Pessimistic |
|------|-----------|-----------|-------------|
| Data (Apify) | $200 | $300 | $400 |
| Training (RunPod A6000) | $5 | $10 | $20 |
| Evaluation (Claude API) | $50 | $100 | $150 |
| Human eval (optional) | $0 | $0 | $390 |
| **Total** | **$255** | **$410** | **$960** |

**Realistic budget**: $400-600

## 📝 Experiment Tracking

All experiments logged to Weights & Biases:

- **Training curves**: Loss, learning rate, gradient norms
- **Hyperparameters**: All config values
- **Evaluation metrics**: Win rates, statistical tests
- **Sample predictions**: Qualitative analysis
- **Model artifacts**: LoRA adapters

View dashboard: `wandb.ai/{username}/qwen3-twitter-lora`

## ⚠️ Known Limitations

From `gameplan.md` critical analysis:

1. **First attempt success rate**: ~30% (plan for 2-3 iterations)
2. **Dataset size**: 800-1,200 pairs (not ideal 2,500, but viable)
3. **Improvement expectation**: 15-30% (not 50-70%)
4. **Timing problem**: Model improves quality, not speed (early-reply advantage remains)
5. **Offline evaluation ≠ real engagement**: Only A/B testing shows true performance

## 🎓 Learning Outcomes

This project demonstrates:

- **LoRA/QLoRA** implementation with PEFT
- **Small dataset fine-tuning** best practices
- **Multi-tier evaluation** design
- **Statistical significance testing** in ML
- **Rigorous experiment tracking** with W&B
- **Honest limitation disclosure** (critical for credibility)

## 📚 Documentation

- **[Implementation Plan](lora_implementation_plan.md)**: Detailed 6-phase roadmap
- **[Project Structure](PROJECT_STRUCTURE.md)**: Directory organization
- **[Research Synthesis](current_research.md)**: Background research
- **[Critical Analysis](gameplan.md)**: Realistic expectations

## 🤝 Contributing

This is a learning project. Key principles:

- **Reproducibility**: All code version controlled, configs saved
- **Transparency**: Document failures as well as successes
- **Quality**: Manual review for small datasets is non-negotiable
- **Statistics**: Report p-values AND effect sizes

## 📄 License

Apache 2.0 (inherited from Qwen3 base model)

## 🙏 Acknowledgments

- **Qwen Team**: Base model
- **HuggingFace**: PEFT library
- **Research**: Current_research.md synthesizes insights from 50+ sources
- **Community**: Gameplan.md incorporates practitioner wisdom

## 📞 Support

For issues with this implementation:
1. Check `PROJECT_STRUCTURE.md` for troubleshooting
2. Review `lora_implementation_plan.md` for methodology
3. Consult `gameplan.md` for realistic expectations

---

**Status**: Phase 1 (Infrastructure) ✅ COMPLETE  
**Next**: Phase 2 (Data Collection)  
**Timeline**: 8 weeks to production-ready model

*"Dataset quality is 95% of everything. Manual review is worth the time."* - Current Research Synthesis

