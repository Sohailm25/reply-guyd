"""
Data Module for Twitter Reply Training

Handles:
- Loading processed JSONL data
- Applying Qwen3 chat template
- Train/val/test splitting
- Tokenization and batching
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TwitterReplyExample:
    """Single training example."""
    tweet_id: str
    tweet: str
    reply: str
    reply_likes: int
    reply_author: str
    
    def to_dict(self):
        return {
            'tweet_id': self.tweet_id,
            'tweet': self.tweet,
            'reply': self.reply,
            'reply_likes': self.reply_likes,
            'reply_author': self.reply_author,
        }


class TwitterReplyDataset(Dataset):
    """
    Dataset for Twitter reply generation.
    
    Applies Qwen3 chat template and tokenizes examples.
    """
    
    def __init__(
        self,
        examples: List[TwitterReplyExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        enable_thinking: bool = False,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        
        logger.info(f"Created TwitterReplyDataset with {len(examples)} examples")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Enable thinking: {enable_thinking}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as chat conversation
        messages = [
            {
                "role": "user",
                "content": f"Generate an engaging Twitter reply to this tweet:\n\n{example.tweet}\n\nReply:"
            },
            {
                "role": "assistant",
                "content": example.reply
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # We want the complete conversation
            enable_thinking=self.enable_thinking
        )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Will pad in collator
            return_tensors=None,  # Return lists, not tensors
        )
        
        # Create labels (mask prompt, keep reply)
        # Find where assistant's response starts
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:1],  # Just user message
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        
        prompt_length = len(prompt_encoding['input_ids'])
        
        # Labels: -100 for prompt tokens, actual tokens for reply
        labels = [-100] * prompt_length + encoding['input_ids'][prompt_length:]
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
        }


class TwitterReplyDataModule:
    """
    Data module for managing train/val/test splits.
    
    Handles:
    - Loading from JSONL files
    - Stratified splitting
    - Dataset creation
    - Statistics logging
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_length: int = 512,
        enable_thinking: bool = False,
        random_seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.random_seed = random_seed
        
        # Validate splits
        assert abs((train_split + val_split + test_split) - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        logger.info("=" * 60)
        logger.info("Initializing TwitterReplyDataModule")
        logger.info("=" * 60)
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Splits: train={train_split}, val={val_split}, test={test_split}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Random seed: {random_seed}")
        
        # Load and split data
        self.train_examples, self.val_examples, self.test_examples = self._load_and_split()
        
        logger.info(f"  Train examples: {len(self.train_examples)}")
        logger.info(f"  Val examples: {len(self.val_examples)}")
        logger.info(f"  Test examples: {len(self.test_examples)}")
        logger.info("=" * 60)
    
    def _load_and_split(self) -> Tuple[List[TwitterReplyExample], List[TwitterReplyExample], List[TwitterReplyExample]]:
        """
        Load data and split into train/val/test.
        
        Returns:
            (train_examples, val_examples, test_examples)
        """
        # Load all examples
        examples = []
        
        if self.data_path.is_dir():
            # Load all JSONL files in directory
            jsonl_files = list(self.data_path.glob("*.jsonl"))
            logger.info(f"Found {len(jsonl_files)} JSONL files")
            
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        examples.append(TwitterReplyExample(
                            tweet_id=data['tweet_id'],
                            tweet=data['tweet'],
                            reply=data['reply'],
                            reply_likes=data.get('reply_likes', 0),
                            reply_author=data.get('reply_author', 'unknown'),
                        ))
        else:
            # Single file
            with open(self.data_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(TwitterReplyExample(
                        tweet_id=data['tweet_id'],
                        tweet=data['tweet'],
                        reply=data['reply'],
                        reply_likes=data.get('reply_likes', 0),
                        reply_author=data.get('reply_author', 'unknown'),
                    ))
        
        logger.info(f"Loaded {len(examples)} total examples")
        
        # Convert to DataFrame for stratified splitting
        df = pd.DataFrame([ex.to_dict() for ex in examples])
        
        # Create engagement quartiles for stratification
        df['engagement_quartile'] = pd.qcut(
            df['reply_likes'],
            q=4,
            labels=[0, 1, 2, 3],
            duplicates='drop'
        )
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(self.val_split + self.test_split),
            stratify=df['engagement_quartile'],
            random_state=self.random_seed
        )
        
        # Second split: val vs test
        if self.test_split > 0:
            val_ratio = self.val_split / (self.val_split + self.test_split)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_ratio),
                stratify=temp_df['engagement_quartile'],
                random_state=self.random_seed
            )
        else:
            val_df = temp_df
            test_df = pd.DataFrame()
        
        # Convert back to examples
        train_examples = [
            TwitterReplyExample(**row.to_dict())
            for _, row in train_df.iterrows()
        ]
        val_examples = [
            TwitterReplyExample(**row.to_dict())
            for _, row in val_df.iterrows()
        ]
        test_examples = [
            TwitterReplyExample(**row.to_dict())
            for _, row in test_df.iterrows()
        ]
        
        # Log statistics
        self._log_split_statistics(train_df, val_df, test_df)
        
        return train_examples, val_examples, test_examples
    
    def _log_split_statistics(self, train_df, val_df, test_df):
        """Log statistics about the splits."""
        logger.info("\nDataset Statistics:")
        logger.info("-" * 60)
        
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(df) == 0:
                continue
                
            logger.info(f"\n{name} Set:")
            logger.info(f"  Examples: {len(df)}")
            logger.info(f"  Unique authors: {df['reply_author'].nunique()}")
            logger.info(f"  Engagement stats:")
            logger.info(f"    Mean: {df['reply_likes'].mean():.1f}")
            logger.info(f"    Median: {df['reply_likes'].median():.1f}")
            logger.info(f"    Min: {df['reply_likes'].min()}")
            logger.info(f"    Max: {df['reply_likes'].max()}")
            logger.info(f"  Text length stats:")
            logger.info(f"    Avg tweet: {df['tweet'].str.len().mean():.1f} chars")
            logger.info(f"    Avg reply: {df['reply'].str.len().mean():.1f} chars")
        
        logger.info("-" * 60)
    
    def get_train_dataset(self) -> TwitterReplyDataset:
        """Get training dataset."""
        return TwitterReplyDataset(
            self.train_examples,
            self.tokenizer,
            self.max_length,
            self.enable_thinking
        )
    
    def get_val_dataset(self) -> TwitterReplyDataset:
        """Get validation dataset."""
        return TwitterReplyDataset(
            self.val_examples,
            self.tokenizer,
            self.max_length,
            self.enable_thinking
        )
    
    def get_test_dataset(self) -> TwitterReplyDataset:
        """Get test dataset."""
        return TwitterReplyDataset(
            self.test_examples,
            self.tokenizer,
            self.max_length,
            self.enable_thinking
        )


def create_data_collator(tokenizer: PreTrainedTokenizer):
    """
    Create data collator for dynamic padding.
    
    Args:
        tokenizer: Tokenizer instance
        
    Returns:
        Data collator function
    """
    from transformers import DataCollatorForLanguageModeling
    
    # Use standard data collator with MLM disabled
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    return collator

