# LoRA Fine-Tuning Implementation Plan for Qwen3-8B
## Twitter Reply Generation: A Rigorous, Iterative Approach

---

## 🎯 Project Overview

**Goal:** Fine-tune Qwen3-8B using LoRA for high-engagement Twitter reply generation

**Model Architecture:**
- Base Model: Qwen3-8B (8.2B params, 6.95B non-embedding)
- 36 transformer layers, GQA (32 Q heads, 8 KV heads)
- Hidden size: 4096, Intermediate: 12288
- Context length: 32,768 (native), 131,072 (with YaRN)
- Precision: bfloat16 (~16GB)

**Target LoRA Layers (7 per transformer block × 36 layers = 252 targets):**
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

**Realistic Expectations:**
- Dataset: 800-1,200 high-quality training pairs
- Improvement: 15-30% over base model (not 50-70%)
- Iterations: 2-3 attempts to get production-ready
- Timeline: 8 weeks (not 1 month)
- Budget: $400-600 (not $229)

---

## 📋 Phase 0: Validation & Baseline (Week 1)
**Status:** MUST DO FIRST - Validates core assumptions before spending $200

### Step 0.1: Test Base Model Performance
**Why:** Modern instruction-tuned models are already good. If base model scores 7/10, fine-tuning ROI is questionable.

```python
# scripts/phase0_test_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_base_model():
    """Load Qwen3-8B in 4-bit for memory efficiency"""
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "./",  # Current directory has the model
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("./")
    return model, tokenizer

def generate_reply(model, tokenizer, tweet, enable_thinking=False):
    """Generate reply with proper Qwen3 chat template"""
    messages = [
        {"role": "user", "content": f"Generate an engaging, witty Twitter reply to this tweet:\n\n{tweet}\n\nReply (max 280 chars):"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # Test both modes
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Non-thinking mode params from README
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # Short replies
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        do_sample=True,
    )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

def evaluate_base_model():
    """Test on 20 diverse tweets across different topics"""
    test_tweets = [
        "Just launched our new API with 100x better rate limits! 🚀",
        "Spent all weekend debugging. Finally works! Coffee was worth it ☕",
        "Hot take: AI coding assistants make you worse at coding",
        "Why does my code work on Friday but breaks on Monday?",
        "Successfully automated myself out of a job. Time to learn something new!",
        # ... add 15 more diverse examples
    ]
    
    model, tokenizer = load_base_model()
    results = []
    
    for tweet in test_tweets:
        # Test both modes
        reply_thinking = generate_reply(model, tokenizer, tweet, enable_thinking=True)
        reply_normal = generate_reply(model, tokenizer, tweet, enable_thinking=False)
        
        results.append({
            "tweet": tweet,
            "reply_thinking": reply_thinking,
            "reply_normal": reply_normal,
        })
    
    # Manual rating: Score each 1-10 for engagement potential
    return results

# Decision criteria:
# - If avg score >= 7/10: Consider RAG instead of fine-tuning
# - If avg score 4-6/10: Fine-tuning likely worthwhile
# - If avg score < 4/10: Check if task is viable at all
```

**Deliverables:**
- [ ] Base model loaded in 4-bit (verify memory usage < 8GB)
- [ ] 20 test replies generated (thinking vs non-thinking modes)
- [ ] Manual quality scores recorded
- [ ] Decision: Proceed with fine-tuning? Try RAG first? Abort?

### Step 0.2: Implement RAG Baseline
**Why:** RAG might outperform fine-tuning with zero training cost

```python
# scripts/phase0_rag_baseline.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class RAGReplyGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.examples = []
        self.embeddings = None
        self.index = None
    
    def add_examples(self, examples):
        """
        examples: List[{"tweet": str, "high_engagement_reply": str}]
        """
        self.examples = examples
        tweet_texts = [ex["tweet"] for ex in examples]
        self.embeddings = self.encoder.encode(tweet_texts)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def generate_with_rag(self, new_tweet, k=3):
        """Find k similar examples and use as few-shot context"""
        query_embedding = self.encoder.encode([new_tweet])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        similar_examples = [self.examples[i] for i in indices[0]]
        
        # Build few-shot prompt
        prompt = "Here are examples of engaging Twitter replies:\n\n"
        for ex in similar_examples:
            prompt += f"Tweet: {ex['tweet']}\nReply: {ex['high_engagement_reply']}\n\n"
        
        prompt += f"Now generate an engaging reply to:\nTweet: {new_tweet}\nReply:"
        
        messages = [{"role": "user", "content": prompt}]
        # ... (use same generation logic as base model)
        
        return reply

# Test RAG with 50 manually collected high-engagement examples
# Compare: Base zero-shot vs RAG few-shot vs Fine-tuning (later)
```

**Deliverables:**
- [ ] Manually collect 50 high-quality tweet-reply pairs
- [ ] Build RAG pipeline with FAISS
- [ ] Generate 20 test replies with RAG
- [ ] Compare: Base zero-shot vs RAG (record scores)
- [ ] Decision: If RAG wins, use it. If tied, fine-tuning adds marginal value

### Step 0.3: Test X API Assumptions (If using official API)
**Why:** Verify `min_faves` works with `conversation_id` before spending $200

```python
# scripts/phase0_test_api.py
def test_api_filtering():
    """
    CRITICAL: Test if min_faves operator works as expected
    """
    # Find a known viral tweet
    test_tweet_id = "1234567890"  # Replace with real viral tweet
    
    # Test 1: Get all replies
    query1 = f"conversation_id:{test_tweet_id}"
    replies_all = client.search_recent_tweets(query1, max_results=100)
    
    # Test 2: Get filtered replies
    query2 = f"conversation_id:{test_tweet_id} min_faves:10"
    replies_filtered = client.search_recent_tweets(query2, max_results=100)
    
    print(f"All replies: {len(replies_all.data)}")
    print(f"Filtered (min_faves:10): {len(replies_filtered.data)}")
    print(f"Post consumption: {check_usage()}")
    
    if len(replies_all.data) == len(replies_filtered.data):
        print("⚠️ WARNING: min_faves filter didn't work!")
        print("You'll waste 80-90% of posts on low-engagement replies")
        print("Expected yield: 500-800 pairs (not 1,500)")
        return False
    
    return True

# Only proceed with X API if this test passes
# Otherwise, use Apify/scraping approach from current_research.md
```

**Deliverables:**
- [ ] X API filtering test (if using official API)
- [ ] Budget projection: realistic training pairs count
- [ ] Alternative data source selected (Apify/scraping if API fails)

---

## 📋 Phase 1: Infrastructure Setup (Week 2)
**Status:** Set up training pipeline with best practices

### Step 1.1: Environment Setup

```bash
# requirements.txt
transformers>=4.51.0  # Qwen3 support
torch>=2.0.0
peft>=0.7.0  # LoRA implementation
bitsandbytes>=0.41.0  # QLoRA quantization
accelerate>=0.21.0
datasets>=2.14.0
wandb>=0.15.0  # Experiment tracking
sentencepiece>=0.1.99
protobuf>=3.20.0

# Optional: Unsloth for 2x speedup (if compatible with Qwen3)
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Evaluation
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

```python
# scripts/phase1_setup.py
import wandb
import torch

def verify_environment():
    """Verify all dependencies and hardware"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test bitsandbytes
    import bitsandbytes as bnb
    print(f"bitsandbytes version: {bnb.__version__}")
    
    # Test PEFT
    from peft import LoraConfig, get_peft_model
    print(f"PEFT available: ✓")
    
    # Initialize W&B
    wandb.login()
    wandb.init(project="qwen3-twitter-lora", name="setup-test")
    wandb.finish()
    
verify_environment()
```

**Deliverables:**
- [ ] Virtual environment created
- [ ] All dependencies installed and tested
- [ ] W&B project initialized
- [ ] GPU memory profiled (should fit 4-bit model + gradients)

### Step 1.2: LoRA Configuration Module

```python
# config/lora_config.py
from peft import LoraConfig, TaskType
from dataclasses import dataclass
from typing import List

@dataclass
class Qwen3LoRAConfig:
    """Conservative LoRA config for small datasets (800-1,500 examples)"""
    
    # LoRA hyperparameters (from current_research.md)
    rank: int = 16  # Start with 16, try 8 if overfitting
    alpha: int = 16  # Keep alpha = rank
    dropout: float = 0.1  # 10% dropout for regularization
    
    # Target modules: ALL linear layers in Qwen3
    target_modules: List[str] = None
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 2  # Start with 2, reduce to 1 if overfitting
    batch_size: int = 4  # Per device
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # QLoRA quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    def __post_init__(self):
        """Define all Qwen3 linear layer targets"""
        if self.target_modules is None:
            self.target_modules = [
                # Attention projections
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                # MLP layers
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
    
    def to_peft_config(self):
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

# Usage:
# config = Qwen3LoRAConfig()
# peft_config = config.to_peft_config()
```

**Deliverables:**
- [ ] LoRA configuration class with sensible defaults
- [ ] All 7 target modules per layer specified
- [ ] Hyperparameter documentation with research citations

### Step 1.3: Dataset Preparation Pipeline

```python
# data/dataset_builder.py
from datasets import Dataset
from typing import List, Dict
import re

class TwitterDatasetBuilder:
    """Build high-quality training dataset with aggressive filtering"""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def filter_quality(self, examples: List[Dict]) -> List[Dict]:
        """
        Aggressive quality filtering per current_research.md:
        - Length: 30-280 chars
        - No URLs (can't replicate)
        - No toxic content
        - Engagement threshold
        - Deduplication
        """
        filtered = []
        
        for ex in examples:
            tweet = ex["tweet"]
            reply = ex["reply"]
            
            # Length filter
            if not (30 <= len(reply) <= 280):
                continue
            
            # No URLs
            if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', reply):
                continue
            
            # Engagement filter (from gameplan.md: normal accounts 1K-50K followers)
            if "author_followers" in ex:
                if ex["author_followers"] > 100000:  # Skip celebrities
                    continue
                if ex["author_followers"] < 500:  # Skip bots/spam
                    continue
            
            # Replicable success only (from gameplan.md)
            if "time_diff_seconds" in ex:
                if ex["time_diff_seconds"] < 300:  # Skip first 5 min (timing advantage)
                    continue
            
            # No media-heavy replies
            if ex.get("has_media", False):
                continue
            
            filtered.append(ex)
        
        return filtered
    
    def deduplicate_semantic(self, examples: List[Dict]) -> List[Dict]:
        """Remove semantic duplicates using embeddings"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        replies = [ex["reply"] for ex in examples]
        embeddings = encoder.encode(replies)
        
        keep_indices = []
        for i in range(len(embeddings)):
            is_duplicate = False
            for j in keep_indices:
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > 0.9:  # High similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        return [examples[i] for i in keep_indices]
    
    def format_for_training(self, examples: List[Dict]) -> Dataset:
        """Format as instruction-following dataset with Qwen3 chat template"""
        formatted = []
        
        for ex in examples:
            # Use Qwen3 chat template format
            messages = [
                {"role": "user", "content": f"Generate an engaging Twitter reply to:\n\n{ex['tweet']}"},
                {"role": "assistant", "content": ex['reply']}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False  # Use non-thinking mode for speed
            )
            
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def build_dataset(self, raw_examples: List[Dict], 
                     train_split=0.9) -> tuple[Dataset, Dataset]:
        """Full pipeline: filter -> dedupe -> format -> split"""
        
        print(f"Starting with {len(raw_examples)} raw examples")
        
        # Filter
        filtered = self.filter_quality(raw_examples)
        print(f"After quality filter: {len(filtered)} examples")
        
        # Deduplicate
        deduped = self.deduplicate_semantic(filtered)
        print(f"After deduplication: {len(deduped)} examples")
        
        # Manual review checkpoint
        print("\n⚠️ MANUAL REVIEW REQUIRED ⚠️")
        print("Review each example for quality. This is CRITICAL for small datasets.")
        
        # Format
        dataset = self.format_for_training(deduped)
        
        # Split
        split_point = int(len(dataset) * train_split)
        train_dataset = dataset.select(range(split_point))
        eval_dataset = dataset.select(range(split_point, len(dataset)))
        
        print(f"\nFinal dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset

# Usage:
# builder = TwitterDatasetBuilder(tokenizer)
# train_ds, eval_ds = builder.build_dataset(raw_data)
```

**Deliverables:**
- [ ] Dataset builder with quality filters
- [ ] Deduplication pipeline
- [ ] Chat template formatting
- [ ] Train/eval split (90/10)

---

## 📋 Phase 2: Data Collection & Curation (Weeks 2-3)
**Status:** Collect 800-1,200 high-quality pairs

### Step 2.1: Data Collection Strategy

**Option A: X API (if Phase 0 validated)**
```python
# data/collect_twitter_api.py
# Budget: $200 for ~1,000 training pairs (conservative estimate)
# See gameplan.md lines 514-528 for realistic collection plan
```

**Option B: Apify (recommended from current_research.md)**
```python
# data/collect_apify.py
# Budget: $200-300 for 800K-2M tweets
# Cost: ~$0.25 per 1,000 tweets
# Much better economics than official API
```

**Target Criteria:**
- 60-80 pairs per day over 14 days
- Target tweets with 200-1,000 likes (manageable reply counts)
- Focus on 5 different topics/queries
- Tweets with 15-40 replies (sweet spot)

### Step 2.2: Manual Quality Review

**CRITICAL:** Per current_research.md, "Dataset quality is 95% of everything"

```python
# data/manual_review.py
import pandas as pd
from IPython.display import display, HTML

def create_review_interface(examples):
    """
    Build simple review interface
    For 1,000 examples at 20 seconds each = 5.5 hours
    WORTH THE TIME for small datasets
    """
    reviewed = []
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1}/{len(examples)} ---")
        print(f"Tweet: {ex['tweet']}")
        print(f"Reply: {ex['reply']}")
        print(f"Engagement: {ex.get('likes', 0)} likes")
        
        keep = input("Keep? (y/n/edit): ").lower()
        
        if keep == 'y':
            reviewed.append(ex)
        elif keep == 'edit':
            new_reply = input("Enter corrected reply: ")
            ex['reply'] = new_reply
            reviewed.append(ex)
        # 'n' = discard
    
    return reviewed

# This is tedious but ESSENTIAL for small dataset success
```

**Deliverables:**
- [ ] 800-1,200 raw examples collected
- [ ] Every example manually reviewed
- [ ] Final dataset: 800-1,000 high-quality pairs
- [ ] Dataset statistics documented (length dist, topic dist, etc.)

---

## 📋 Phase 3: Initial Training Run (Week 4)
**Status:** First fine-tuning attempt with conservative hyperparameters

### Step 3.1: Training Script

```python
# scripts/train_lora.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, prepare_model_for_kbit_training
import wandb
from config.lora_config import Qwen3LoRAConfig

def train_qwen3_lora(
    model_path: str = "./",
    train_dataset = None,
    eval_dataset = None,
    output_dir: str = "./output/lora-v1",
):
    # Initialize W&B
    wandb.init(
        project="qwen3-twitter-lora",
        name="training-v1",
        config={
            "model": "Qwen3-8B",
            "method": "LoRA + QLoRA",
            "dataset_size": len(train_dataset),
        }
    )
    
    # Load config
    lora_config = Qwen3LoRAConfig()
    
    # QLoRA quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config.to_peft_config())
    model.print_trainable_parameters()
    # Should show ~0.1-1% trainable parameters
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=lora_config.num_epochs,
        per_device_train_batch_size=lora_config.batch_size,
        gradient_accumulation_steps=lora_config.gradient_accumulation_steps,
        learning_rate=lora_config.learning_rate,
        warmup_ratio=lora_config.warmup_ratio,
        weight_decay=lora_config.weight_decay,
        max_grad_norm=lora_config.max_grad_norm,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,  # Frequent evaluation for small datasets
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="wandb",
        # Early stopping via callbacks (see below)
    )
    
    # Early stopping callback
    from transformers import EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 evals
        early_stopping_threshold=0.01,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping],
    )
    
    # Monitor for overfitting
    print("\n⚠️ OVERFITTING WATCH ⚠️")
    print("Monitor train_loss vs eval_loss gap in W&B")
    print("If gap > 0.3, reduce epochs or increase dropout")
    
    # Train!
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    wandb.finish()
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    print("Check W&B for training curves")
    
    return model, tokenizer

# Run training
# train_qwen3_lora(train_dataset=train_ds, eval_dataset=eval_ds)
```

**Deliverables:**
- [ ] Training script with W&B integration
- [ ] Early stopping to prevent overfitting
- [ ] Model checkpoints saved every 50 steps
- [ ] Training curves monitored in real-time
- [ ] Final LoRA adapters saved

### Step 3.2: Overfitting Prevention Checklist

From gameplan.md (lines 378-406), common failure modes:

```python
# scripts/check_overfitting.py
def diagnose_training_run(wandb_run_id):
    """Analyze training run for common failure modes"""
    api = wandb.Api()
    run = api.run(f"your-username/qwen3-twitter-lora/{wandb_run_id}")
    
    history = run.history()
    
    # Check 1: Overfitting
    final_train_loss = history['train_loss'].iloc[-1]
    final_eval_loss = history['eval_loss'].iloc[-1]
    gap = final_eval_loss - final_train_loss
    
    if gap > 0.3:
        print("⚠️ OVERFITTING DETECTED")
        print("Solution: Reduce epochs to 1, increase dropout to 0.15")
    
    # Check 2: Underfitting
    if final_train_loss > 1.0:
        print("⚠️ UNDERFITTING DETECTED")
        print("Solution: Increase epochs to 3, increase rank to 32")
    
    # Check 3: Catastrophic forgetting
    if final_train_loss < 0.1:
        print("⚠️ POSSIBLE CATASTROPHIC FORGETTING")
        print("Solution: Reduce learning rate to 1e-4")
    
    # Check 4: Loss plateaus
    recent_losses = history['eval_loss'].iloc[-10:]
    if recent_losses.std() < 0.01:
        print("⚠️ LOSS PLATEAUED")
        print("Check if training finished early or needs more steps")
    
    return {
        "train_loss": final_train_loss,
        "eval_loss": final_eval_loss,
        "gap": gap,
        "verdict": "Good" if 0.1 < gap < 0.3 else "Needs adjustment"
    }
```

**Deliverables:**
- [ ] Training diagnostics run
- [ ] Overfitting check passed (gap < 0.3)
- [ ] Decision: Keep model or retrain with adjusted hyperparams

---

## 📋 Phase 4: Multi-Tier Evaluation (Week 5)
**Status:** Rigorous evaluation with statistical significance

### Step 4.1: Automated Metrics

```python
# evaluation/automated_metrics.py
from datasets import load_metric
import numpy as np

def compute_automated_metrics(model, tokenizer, test_dataset):
    """
    Traditional NLP metrics (limited value for engagement prediction)
    But required for completeness
    """
    rouge = load_metric('rouge')
    
    predictions = []
    references = []
    
    for example in test_dataset:
        # Generate prediction
        tweet = example['tweet']
        true_reply = example['reply']
        
        pred_reply = generate_reply(model, tokenizer, tweet)
        
        predictions.append(pred_reply)
        references.append(true_reply)
    
    # ROUGE scores
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    return {
        "rouge1": rouge_scores['rouge1'].mid.fmeasure,
        "rouge2": rouge_scores['rouge2'].mid.fmeasure,
        "rougeL": rouge_scores['rougeL'].mid.fmeasure,
    }

# Note: Per current_research.md, these correlate poorly with engagement
# Use as sanity check only, not primary metric
```

### Step 4.2: LLM-as-Judge Evaluation

```python
# evaluation/llm_judge.py
import anthropic
from typing import List, Dict

class LLMJudge:
    """
    G-Eval framework from current_research.md
    Achieves 0.514 Spearman correlation with humans
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def evaluate_pairwise(self, tweet: str, reply_a: str, reply_b: str) -> Dict:
        """
        Compare two replies with chain-of-thought reasoning
        Mitigates position bias by swapping order
        """
        
        prompt = f"""You are evaluating Twitter replies for engagement potential.

Tweet: "{tweet}"

Reply A: "{reply_a}"
Reply B: "{reply_b}"

Evaluate these replies on:
1. Engagement potential (would people like/retweet?)
2. Relevance to the original tweet
3. Tone appropriateness
4. Wit/creativity
5. Conciseness (Twitter prefers punchy)

Think step-by-step, then choose: A is better, B is better, or Tie.

Format your response as:
Reasoning: [your reasoning]
Winner: [A/B/Tie]"""

        # Run twice with swapped positions to mitigate position bias
        response1 = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.3,  # Low but not zero (current_research.md: temp=0 unreliable)
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Swap positions
        prompt_swapped = prompt.replace("Reply A:", "TEMP").replace("Reply B:", "Reply A:").replace("TEMP", "Reply B:")
        
        response2 = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt_swapped}]
        )
        
        # Parse and combine results
        winner1 = self._parse_winner(response1.content[0].text)
        winner2 = self._parse_winner(response2.content[0].text, swapped=True)
        
        # Consensus
        if winner1 == winner2:
            return {"winner": winner1, "confidence": "high"}
        else:
            return {"winner": "Tie", "confidence": "low"}
    
    def _parse_winner(self, response: str, swapped: bool = False) -> str:
        if "Winner: A" in response:
            return "B" if swapped else "A"
        elif "Winner: B" in response:
            return "A" if swapped else "B"
        else:
            return "Tie"
    
    def evaluate_model(self, base_replies: List[str], 
                       finetuned_replies: List[str],
                       tweets: List[str]) -> Dict:
        """
        Head-to-head comparison: base vs fine-tuned
        """
        wins_base = 0
        wins_finetuned = 0
        ties = 0
        
        for tweet, reply_base, reply_ft in zip(tweets, base_replies, finetuned_replies):
            result = self.evaluate_pairwise(tweet, reply_base, reply_ft)
            
            if result["winner"] == "A":  # base
                wins_base += 1
            elif result["winner"] == "B":  # fine-tuned
                wins_finetuned += 1
            else:
                ties += 1
        
        total = len(tweets)
        return {
            "base_wins": wins_base,
            "finetuned_wins": wins_finetuned,
            "ties": ties,
            "finetuned_win_rate": wins_finetuned / total,
        }

# Usage:
# judge = LLMJudge(api_key=os.getenv("ANTHROPIC_API_KEY"))
# results = judge.evaluate_model(base_replies, ft_replies, test_tweets)
```

**Deliverables:**
- [ ] LLM judge implementation with bias mitigation
- [ ] 50-100 pairwise comparisons (base vs fine-tuned)
- [ ] Win rate calculated with confidence intervals

### Step 4.3: Statistical Significance Testing

```python
# evaluation/statistical_tests.py
from scipy.stats import mannwhitneyu, ttest_ind
import numpy as np

def test_significance(base_scores, finetuned_scores, alpha=0.05):
    """
    Per current_research.md: Report p-values AND effect sizes
    Use Mann-Whitney U (non-parametric, safer choice)
    """
    
    # Mann-Whitney U test
    statistic, p_value = mannwhitneyu(
        finetuned_scores,
        base_scores,
        alternative='greater'  # H1: fine-tuned > base
    )
    
    # Effect size: Cohen's d
    mean_diff = np.mean(finetuned_scores) - np.mean(base_scores)
    pooled_std = np.sqrt((np.std(base_scores)**2 + np.std(finetuned_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    # Interpret effect size
    if cohens_d < 0.2:
        effect = "negligible"
    elif cohens_d < 0.5:
        effect = "small"
    elif cohens_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"\n=== Statistical Significance Test ===")
    print(f"Mann-Whitney U statistic: {statistic}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < alpha else 'No'}")
    print(f"\nCohen's d: {cohens_d:.3f} ({effect} effect)")
    print(f"Mean improvement: {mean_diff:.3f}")
    print(f"Fine-tuned mean: {np.mean(finetuned_scores):.3f}")
    print(f"Base mean: {np.mean(base_scores):.3f}")
    
    return {
        "p_value": p_value,
        "statistically_significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": effect,
        "mean_improvement": mean_diff,
    }

# Decision criteria from gameplan.md:
# Success = p < 0.05 AND improvement > 20% AND Cohen's d > 0.3
```

**Deliverables:**
- [ ] Statistical tests implemented
- [ ] p-value < 0.05 achieved (or not - be honest!)
- [ ] Effect size (Cohen's d) calculated
- [ ] Results table with confidence intervals
- [ ] Decision: Is fine-tuned model significantly better?

### Step 4.4: Human Evaluation (Optional - $375-390)

```python
# evaluation/prolific_eval.py
"""
Per current_research.md:
- Prolific: Highest quality platform
- Cost: $2.40-2.60 per response
- Target: 150 participants for significance
- Budget: ~$390

Only do this if:
1. LLM judge shows promising results
2. You need publication-grade validation
3. Budget allows
"""

def create_prolific_survey():
    """
    Survey design:
    - Show participant a tweet
    - Show 2 replies (base vs fine-tuned, order randomized)
    - Ask: "Which reply would you be more likely to like/retweet?"
    - Repeat for 10 tweet-reply pairs
    - Payment: $8/hour minimum (Prolific requirement)
    - Time: ~5 minutes = $0.67 per participant
    - Add 30% platform fee = $0.87 per participant
    - 150 participants × $0.87 = $130.50... wait that's wrong
    
    Actually per current_research.md:
    - $1.90-2.00 per response + 25-30% fee = $2.40-2.60
    - 150 participants × $2.50 = $375
    """
    
    # Build survey on Prolific platform (web interface)
    # Export results for analysis
    pass

# Only include if budget allows and earlier results are promising
```

**Deliverables (if doing human eval):**
- [ ] Prolific survey designed and launched
- [ ] 150 responses collected
- [ ] Inter-rater reliability calculated
- [ ] Human preference % for fine-tuned model

---

## 📋 Phase 5: Iteration & Refinement (Weeks 6-8)
**Status:** Hyperparameter tuning based on Phase 4 results

### Step 5.1: Diagnose Issues & Adjust

```python
# scripts/phase5_iterate.py

def analyze_v1_results(eval_results):
    """
    Based on Phase 4 evaluation, determine next steps
    """
    
    # Scenario 1: Overfitting (high train, low eval)
    if eval_results["train_eval_gap"] > 0.3:
        return {
            "issue": "Overfitting",
            "adjustments": {
                "num_epochs": 1,  # Reduce from 2
                "lora_dropout": 0.15,  # Increase from 0.1
                "rank": 8,  # Reduce from 16
            }
        }
    
    # Scenario 2: Underfitting (both losses high)
    if eval_results["eval_loss"] > 1.0:
        return {
            "issue": "Underfitting",
            "adjustments": {
                "num_epochs": 3,  # Increase from 2
                "rank": 32,  # Increase from 16
                "learning_rate": 3e-4,  # Increase from 2e-4
            }
        }
    
    # Scenario 3: Style drift (all replies sound similar)
    if eval_results["diversity_score"] < 0.5:
        return {
            "issue": "Style drift / lack of diversity",
            "adjustments": {
                "temperature": 0.8,  # Increase sampling temperature
                "add_diverse_data": True,  # Collect more varied examples
            }
        }
    
    # Scenario 4: Marginal improvement (< 15%)
    if eval_results["improvement"] < 0.15:
        return {
            "issue": "Insufficient improvement",
            "recommendation": "Consider RAG approach or collect more data"
        }
    
    # Scenario 5: Success!
    if (eval_results["improvement"] > 0.20 and 
        eval_results["p_value"] < 0.05 and
        eval_results["cohens_d"] > 0.3):
        return {
            "issue": None,
            "verdict": "SUCCESS! Model is production-ready",
            "next_steps": "Deploy and A/B test in real-world"
        }

# Run analysis and generate v2 training config
```

### Step 5.2: Training Run v2

```python
# scripts/train_lora_v2.py
# Same as train_lora.py but with adjusted hyperparameters

# Example adjustments based on v1 results:
lora_config_v2 = Qwen3LoRAConfig(
    rank=8,  # Reduced due to overfitting
    alpha=8,
    dropout=0.15,  # Increased regularization
    num_epochs=1,  # Reduced to prevent memorization
    learning_rate=1.5e-4,  # Slightly reduced
)

# Retrain and re-evaluate
```

**Deliverables:**
- [ ] v1 results analyzed
- [ ] v2 hyperparameters selected based on diagnosis
- [ ] v2 model trained
- [ ] v2 evaluation completed
- [ ] Comparison: v1 vs v2 vs base

### Step 5.3: Final Model Selection

```python
# scripts/select_final_model.py

def select_final_model(results_base, results_v1, results_v2):
    """
    Compare all models and select winner
    """
    models = [
        {"name": "Base Model", "scores": results_base},
        {"name": "LoRA v1", "scores": results_v1},
        {"name": "LoRA v2", "scores": results_v2},
    ]
    
    # Rank by composite score
    for model in models:
        model["composite"] = (
            0.4 * model["scores"]["engagement_score"] +
            0.3 * model["scores"]["llm_judge_win_rate"] +
            0.3 * (1 - model["scores"]["eval_loss"])
        )
    
    models.sort(key=lambda x: x["composite"], reverse=True)
    
    print("\n=== Final Model Ranking ===")
    for i, model in enumerate(models):
        print(f"{i+1}. {model['name']}: {model['composite']:.3f}")
    
    winner = models[0]
    print(f"\n🏆 Winner: {winner['name']}")
    
    # Publish to HuggingFace Hub
    if winner["name"] != "Base Model":
        print(f"Publishing {winner['name']} to HuggingFace Hub...")
        # hub.push_to_hub(...)
    
    return winner
```

**Deliverables:**
- [ ] All models compared systematically
- [ ] Final model selected with justification
- [ ] Model published to HuggingFace Hub
- [ ] W&B report generated

---

## 📋 Phase 6: Deployment & Real-World Testing (Optional)
**Status:** Production deployment and A/B testing

### Step 6.1: Deployment Options

```python
# deployment/deploy.py

# Option 1: Local inference with quantization
# - Fast on M3 Max with Metal
# - Free
# - Best for personal use

# Option 2: HuggingFace Inference API
# - $0.06 per 1K tokens
# - Easy integration
# - Good for low-volume

# Option 3: RunPod on-demand
# - $0.39/hour for A6000
# - Only pay when generating
# - Good for batch generation

# Option 4: Gradio demo
# - Free on HuggingFace Spaces
# - Perfect for portfolio
# - Shareable link
```

### Step 6.2: A/B Testing (Real-World Validation)

```python
# testing/ab_test.py
"""
The ONLY true evaluation per gameplan.md
"""

def run_ab_test(weeks=4):
    """
    Week 1-2: Base model (25 replies)
    Week 3-4: Fine-tuned model (25 replies)
    
    Control for:
    - Topic distribution
    - Time of day
    - Original tweet engagement
    
    Measure:
    - Likes per reply
    - Retweets per reply
    - Replies to your reply
    - Impressions (if accessible)
    """
    
    # Log every reply with metadata
    # After 4 weeks, compare engagement
    # Use t-test for significance
    
    pass

# Success criteria:
# Fine-tuned engagement > Base engagement by 20%+ (p < 0.05)
```

**Deliverables:**
- [ ] Deployment method selected
- [ ] Gradio demo published (portfolio)
- [ ] A/B test plan (if deploying for real use)
- [ ] Real-world engagement data (after 4 weeks)

---

## 📊 Success Metrics & Decision Points

### Phase 0 Decision Point
- **Proceed to fine-tuning if:** Base model < 6/10 AND (RAG < base OR no time for RAG)
- **Use RAG if:** RAG ≥ base model (saves $200-400)
- **Abort if:** Base model ≥ 7/10 (fine-tuning ROI too low)

### Phase 4 Decision Point
- **Success:** p < 0.05 AND improvement > 20% AND Cohen's d > 0.3
- **Marginal:** p < 0.10 OR improvement 10-20%
- **Failure:** p ≥ 0.10 OR improvement < 10%

### Final Success Criteria
1. **Technical:** Statistically significant improvement over base
2. **Practical:** 15-30% better engagement in real testing
3. **Portfolio:** W&B dashboard + GitHub repo + HF model + blog post

---

## 💰 Budget Breakdown (Realistic)

| Phase | Item | Optimistic | Realistic | Pessimistic |
|-------|------|-----------|-----------|-------------|
| 0 | Validation (API test) | $0 | $20 | $50 |
| 2 | Data collection (Apify) | $200 | $300 | $400 |
| 3 | Training v1 (RunPod A6000) | $3 | $5 | $10 |
| 4 | LLM judge (Claude API) | $50 | $100 | $150 |
| 4 | Human eval (Prolific) | $0 | $0 | $390 |
| 5 | Training v2+v3 | $6 | $10 | $20 |
| **Total (no human eval)** | | **$259** | **$435** | **$630** |
| **Total (with human eval)** | | **$259** | **$435** | **$1,020** |

**Recommendation:** Budget $500 realistically, $700 if including human evaluation

---

## 📝 Deliverables Checklist

### Code Artifacts
- [ ] `config/lora_config.py` - Hyperparameter configurations
- [ ] `data/dataset_builder.py` - Data preprocessing pipeline
- [ ] `scripts/train_lora.py` - Training script
- [ ] `evaluation/llm_judge.py` - Evaluation framework
- [ ] `evaluation/statistical_tests.py` - Significance testing
- [ ] All scripts in Git with clear documentation

### Experiment Tracking
- [ ] W&B project with all runs
- [ ] Training curves for v1, v2, v3
- [ ] Evaluation metrics dashboard
- [ ] Hyperparameter comparison plots

### Model Artifacts
- [ ] Base model (already have)
- [ ] LoRA adapters v1, v2 (saved locally)
- [ ] Final model on HuggingFace Hub
- [ ] Gradio demo deployed

### Documentation
- [ ] Technical blog post (methodology + results)
- [ ] GitHub README with reproducibility instructions
- [ ] W&B report (shareable link)
- [ ] This implementation plan (updated with results)

### Portfolio Materials
- [ ] Landing page: "Qwen3 Twitter LoRA: 23% Improvement with 900 Examples"
- [ ] Key result: "Statistically significant (p=0.003, Cohen's d=0.52)"
- [ ] Demo link + model card + code repo
- [ ] Honest limitations section

---

## 🚨 Risk Mitigation

### Technical Risks
1. **Overfitting on small dataset**
   - Mitigation: Early stopping, high dropout, 1-2 epochs max
   - Fallback: Collect more data or use RAG

2. **Base model already too good**
   - Mitigation: Phase 0 validation catches this early
   - Fallback: Use project to demonstrate evaluation methodology instead

3. **Qwen3 thinking mode complexity**
   - Mitigation: Use `enable_thinking=False` for simpler training
   - Fallback: Switch to Qwen2.5-Instruct

### Budget Risks
1. **Data collection more expensive than expected**
   - Mitigation: Start with Apify trial, validate costs
   - Fallback: Manual collection (slower but free)

2. **Multiple training iterations needed**
   - Mitigation: Conservative hyperparams reduce failures
   - Fallback: Accept v2 as final even if not perfect

### Timeline Risks
1. **Manual review takes longer than 5 hours**
   - Mitigation: Start early, review 100/day over 10 days
   - Fallback: Semi-automated review with spot checks

---

## 🎓 Learning Outcomes

By completing this project, you will master:

### Technical Skills
- LoRA/QLoRA implementation with PEFT
- Small dataset fine-tuning best practices
- Quantization (4-bit) for memory efficiency
- Modern evaluation frameworks (LLM-as-judge)
- Statistical significance testing in ML
- W&B experiment tracking

### ML Engineering
- Data quality > quantity principle
- Overfitting prevention strategies
- Hyperparameter tuning methodology
- Multi-tier evaluation design
- Production deployment considerations

### Portfolio Development
- Technical blog writing
- Model card creation
- Reproducible research practices
- Honest limitation disclosure
- Gradio demo deployment

---

## 📚 Key References

From `current_research.md`:
- LoRA with small datasets: Lines 35-45 (LIMA study, 1,000 examples)
- Evaluation frameworks: Lines 23-33 (G-Eval, LLM-as-judge)
- Statistical significance: Lines 48-56 (p-values + effect sizes)
- Overfitting prevention: Lines 40-42 (dropout, early stopping)

From `gameplan.md`:
- Realistic expectations: Lines 562-596 (15-30% improvement)
- Phase 0 validation: Lines 481-509 (test before commit)
- Failure modes: Lines 378-418 (overfitting, underfitting, etc.)
- Budget reality: Lines 561-570 ($400-600 realistic)

---

## 🚀 Next Steps

1. **Review this plan** - Does it match your goals and constraints?
2. **Set up environment** - Install dependencies, verify GPU access
3. **Start Phase 0** - Test base model performance THIS WEEK
4. **Make go/no-go decision** - Based on Phase 0 results
5. **Execute iteratively** - Follow phases, update plan as you learn

**Remember:** The goal is demonstrating **rigorous ML engineering**, not achieving state-of-the-art engagement. Honest evaluation and transparent methodology matter more than perfect results.

---

## 📞 Support & Resources

- **W&B Docs:** https://docs.wandb.ai/
- **PEFT Docs:** https://huggingface.co/docs/peft
- **Qwen3 Official Guide:** https://qwen.readthedocs.io/
- **Statistical Testing:** scipy.stats documentation
- **This repository:** All code will be well-commented

Good luck! 🎯

