# Project Structure

This document explains the organization of the Qwen3-8B LoRA fine-tuning project.

## Directory Layout

```
Qwen3-8/
├── config/                              # Configuration files
│   ├── data_collection_config.yaml      # Data collection settings
│   └── lora_config.yaml                 # LoRA training hyperparameters
│
├── data/                                # Data storage (gitignored)
│   ├── raw/                             # Raw collected data
│   ├── processed/                       # Cleaned, validated data
│   └── cache/                           # Temporary cache files
│
├── src/                                 # Source code modules
│   ├── data_collection/                 # Data collection pipeline
│   │   ├── __init__.py
│   │   ├── apify_collector.py           # Apify-based collection
│   │   ├── scraper_collector.py         # Web scraping (twscrape)
│   │   ├── data_validator.py            # Quality validation
│   │   └── data_cleaner.py              # Cleaning & deduplication
│   │
│   ├── training/                        # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py                   # LoRA training logic
│   │   ├── dataset.py                   # Dataset preparation
│   │   └── callbacks.py                 # Training callbacks
│   │
│   ├── evaluation/                      # Evaluation modules
│   │   ├── __init__.py
│   │   ├── automated_metrics.py         # ROUGE, BLEU, perplexity
│   │   ├── llm_judge.py                 # LLM-as-judge evaluation
│   │   └── statistical_tests.py         # Significance testing
│   │
│   └── utils/                           # Utility functions
│       ├── __init__.py
│       ├── logging_utils.py
│       └── model_utils.py
│
├── scripts/                             # Executable scripts
│   ├── collect_data.py                  # Main data collection script
│   ├── train_lora.py                    # Main training script
│   ├── evaluate_model.py                # Evaluation script
│   └── phase0_baseline.py               # Phase 0 validation
│
├── notebooks/                           # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_manual_review.ipynb
│   └── 03_results_analysis.ipynb
│
├── output/                              # Generated outputs (gitignored)
│   ├── models/                          # Saved models & LoRA adapters
│   ├── logs/                            # Training & collection logs
│   └── figures/                         # Generated plots & visualizations
│
├── tests/                               # Unit tests
│   ├── test_data_collection.py
│   ├── test_validation.py
│   └── test_training.py
│
├── requirements.txt                     # Python dependencies
├── setup.sh                             # Setup script
├── .env.example                         # Environment variables template
├── .gitignore                           # Git ignore rules
├── README.md                            # Main project README
├── lora_implementation_plan.md          # Detailed implementation plan
└── PROJECT_STRUCTURE.md                 # This file
```

## Key Files

### Configuration Files

- **`config/data_collection_config.yaml`**: Controls data collection behavior
  - Search queries
  - Filter thresholds (engagement, timing, content)
  - Quality criteria
  - Storage settings

- **`config/lora_config.yaml`**: LoRA training hyperparameters
  - Model settings (quantization, dtype)
  - LoRA parameters (rank, alpha, dropout)
  - Training hyperparameters (lr, batch size, epochs)
  - Evaluation strategy

### Data Collection Pipeline

1. **`apify_collector.py`**: Primary collection method
   - Uses Apify actors for reliable data access
   - ~$0.25 per 1,000 tweets
   - Handles rate limiting and pagination

2. **`scraper_collector.py`**: Backup collection method
   - Direct web scraping using twscrape
   - Requires Twitter accounts
   - More complex setup but cheaper at scale

3. **`data_validator.py`**: Quality checks
   - Length validation
   - Language detection
   - Toxicity filtering
   - Engagement thresholds

4. **`data_cleaner.py`**: Data preprocessing
   - Text normalization
   - Exact deduplication
   - Semantic deduplication (using embeddings)
   - Quality sorting

### Main Scripts

- **`scripts/collect_data.py`**: End-to-end data collection
  ```bash
  python scripts/collect_data.py --method apify --target 1000
  ```

- **`scripts/train_lora.py`**: LoRA fine-tuning
  ```bash
  python scripts/train_lora.py --config config/lora_config.yaml
  ```

- **`scripts/evaluate_model.py`**: Model evaluation
  ```bash
  python scripts/evaluate_model.py --model output/models/lora-v1
  ```

## Data Flow

```
1. COLLECTION
   ├── Search Twitter via Apify/scraping
   ├── Fetch tweets matching queries
   ├── For each tweet, fetch high-engagement replies
   └── Save raw data → data/raw/

2. VALIDATION
   ├── Load raw data
   ├── Apply quality filters
   │   ├── Length (30-280 chars)
   │   ├── Language (English, confidence > 0.8)
   │   ├── Toxicity (score < 0.3)
   │   ├── Engagement (min likes)
   │   └── Content (no URLs, media, spam)
   └── Save valid pairs → data/processed/

3. CLEANING
   ├── Load valid pairs
   ├── Normalize text
   │   ├── Elongated characters
   │   ├── Hashtags
   │   └── Whitespace
   ├── Remove duplicates
   │   ├── Exact matches
   │   └── Semantic similarity > 0.9
   ├── Sort by quality
   └── Save final dataset → data/processed/

4. TRAINING
   ├── Load processed data
   ├── Split train/eval (90/10)
   ├── Format with Qwen3 chat template
   ├── Apply LoRA to all linear layers
   ├── Train with QLoRA (4-bit quantization)
   └── Save adapters → output/models/

5. EVALUATION
   ├── Load base model + LoRA adapters
   ├── Run automated metrics (ROUGE, BLEU)
   ├── LLM-as-judge pairwise comparison
   ├── Statistical significance tests
   └── Generate reports → output/logs/
```

## Configuration Management

All settings are centralized in YAML config files:

### Data Collection Config
```yaml
collection:
  target_pairs: 1000
  search_queries: [...]
  tweet_filters: {...}
  reply_filters: {...}
  quality: {...}
```

### Training Config
```yaml
model:
  path: "./"
  quantization: {...}

lora:
  rank: 16
  alpha: 16
  dropout: 0.1
  target_modules: [...]

training:
  num_epochs: 2
  learning_rate: 2e-4
  batch_size: 4
  ...
```

## Environment Variables

Create `.env` from `.env.example` and configure:

```bash
# Required for Apify
APIFY_API_TOKEN=your_token

# Required for evaluation
ANTHROPIC_API_KEY=your_key  # Claude for LLM-as-judge
WANDB_API_KEY=your_key      # Experiment tracking

# Optional for scraping
TWITTER_ACCOUNT_1_USERNAME=...
TWITTER_ACCOUNT_1_PASSWORD=...
```

## Storage Conventions

### Raw Data Format (JSONL)
```json
{
  "tweet_id": "123",
  "tweet": "Original tweet text",
  "tweet_likes": 500,
  "reply_id": "456",
  "reply": "High-engagement reply text",
  "reply_likes": 50,
  "reply_author_followers": 5000,
  "collected_at": "2024-01-01T12:00:00"
}
```

### Processed Data Format
Same as raw but with additional fields:
- `validation`: Validation results
- Normalized text fields
- Quality scores

### Model Outputs
```
output/models/lora-v1/
├── adapter_config.json
├── adapter_model.safetensors
└── README.md
```

## Logging

All operations log to both console and files:
- Data collection: `output/logs/data_collection_TIMESTAMP.log`
- Training: `output/logs/training_TIMESTAMP.log`
- Evaluation: `output/logs/evaluation_TIMESTAMP.log`

## Best Practices

1. **Always review collected data manually**
   - Quality > quantity for small datasets
   - Spot-check samples regularly

2. **Use checkpoints**
   - Collection saves every 10 tweets
   - Training saves every 50 steps
   - Resume interrupted jobs easily

3. **Version your configs**
   - Save config snapshots with model outputs
   - Track hyperparameter changes in W&B

4. **Document experiments**
   - Use W&B for experiment tracking
   - Keep notes in notebooks/
   - Update lora_implementation_plan.md with learnings

## Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Collect data
python scripts/collect_data.py --method apify --target 1000

# 4. Review data
head -n 10 data/processed/training_data_*.jsonl

# 5. Train (placeholder - will create in Phase 3)
python scripts/train_lora.py

# 6. Evaluate (placeholder - will create in Phase 4)
python scripts/evaluate_model.py
```

## Development Workflow

1. **Phase 0**: Baseline testing (optional but recommended)
2. **Phase 1**: Infrastructure setup (DONE)
3. **Phase 2**: Data collection (use `collect_data.py`)
4. **Phase 3**: Training v1 (use `train_lora.py`)
5. **Phase 4**: Evaluation (use `evaluate_model.py`)
6. **Phase 5**: Iteration (adjust configs, retrain)

Each phase builds on previous phases and should be tracked in W&B.

## Troubleshooting

### Apify collection fails
- Check `APIFY_API_TOKEN` in `.env`
- Verify Apify account has credits
- Try reducing `max_tweets` in first test run

### Out of memory during training
- Reduce `per_device_train_batch_size` in config
- Ensure `load_in_4bit: true` is enabled
- Close other GPU applications

### Validation rejects too many pairs
- Review rejection reasons in logs
- Adjust thresholds in `data_collection_config.yaml`
- Check if data sources match expected format

## Resources

- **Implementation Plan**: `lora_implementation_plan.md` (detailed roadmap)
- **Research Notes**: `current_research.md` (background research)
- **Critical Analysis**: `gameplan.md` (realistic expectations)

