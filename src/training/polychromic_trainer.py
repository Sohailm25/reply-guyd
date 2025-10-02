"""
Polychromic Trainer: Diversity-Aware LoRA Fine-Tuning

Implementation of diversity-aware training objective:
    L = L_quality - λ * D(generations)

Where:
    L_quality: Standard cross-entropy loss
    D: Diversity score (semantic, BLEU, or distinct-n based)
    λ: Diversity weight hyperparameter

This is the core research contribution for the paper.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass, asdict
import wandb
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PolychromicConfig:
    """Configuration for polychromic training."""
    diversity_weight: float = 0.3
    n_generations: int = 3
    diversity_metric: str = "semantic"  # "semantic", "bleu", "distinct"
    diversity_temperature: float = 0.9
    diversity_top_p: float = 0.9
    max_generation_length: int = 100
    diversity_encoder_model: str = "all-MiniLM-L6-v2"
    
    # Computational optimizations
    compute_diversity_every_n_steps: int = 1
    cache_diversity_encoder: bool = True
    
    # Batch processing for diversity
    max_examples_for_diversity: int = 4  # Process max N examples per batch
    
    def to_dict(self):
        return asdict(self)


class PolychromicTrainer(Trainer):
    """
    Custom Trainer with diversity-aware objective.
    
    Key differences from standard Trainer:
    1. Computes quality loss (standard CE)
    2. Generates N diverse replies per example
    3. Computes diversity score
    4. Combines: total_loss = quality_loss - λ * diversity_score
    
    Computational considerations:
    - ~3x slower than standard training due to multiple generations
    - Can reduce frequency with compute_diversity_every_n_steps
    - Diversity encoder cached on GPU for speed
    """
    
    def __init__(
        self,
        polychromic_config: PolychromicConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.polychromic_config = polychromic_config
        
        # Initialize diversity encoder
        if polychromic_config.diversity_metric == "semantic":
            logger.info(f"Loading diversity encoder: {polychromic_config.diversity_encoder_model}")
            self.diversity_encoder = SentenceTransformer(
                polychromic_config.diversity_encoder_model
            )
            if polychromic_config.cache_diversity_encoder:
                self.diversity_encoder = self.diversity_encoder.to(self.args.device)
                logger.info("Diversity encoder cached on GPU")
        else:
            self.diversity_encoder = None
        
        # Tracking
        self.diversity_scores_history = []
        self.quality_losses_history = []
        self.generation_times = []
        
        logger.info("=" * 60)
        logger.info("Initialized PolychromicTrainer")
        logger.info("=" * 60)
        logger.info(f"  Diversity weight (λ): {polychromic_config.diversity_weight}")
        logger.info(f"  N generations: {polychromic_config.n_generations}")
        logger.info(f"  Diversity metric: {polychromic_config.diversity_metric}")
        logger.info(f"  Temperature: {polychromic_config.diversity_temperature}")
        logger.info(f"  Compute every N steps: {polychromic_config.compute_diversity_every_n_steps}")
        logger.info("=" * 60)
    
    def compute_diversity_score(self, texts: List[str]) -> float:
        """
        Compute diversity score for a list of generated texts.
        
        Args:
            texts: List of generated reply texts
            
        Returns:
            diversity_score: Higher = more diverse
        """
        if len(texts) < 2:
            return 0.0
        
        metric = self.polychromic_config.diversity_metric
        
        try:
            if metric == "semantic":
                return self._semantic_diversity(texts)
            elif metric == "bleu":
                return self._bleu_diversity(texts)
            elif metric == "distinct":
                return self._distinct_n_diversity(texts)
            else:
                logger.warning(f"Unknown diversity metric: {metric}, using semantic")
                return self._semantic_diversity(texts)
        except Exception as e:
            logger.error(f"Error computing diversity: {e}")
            return 0.0
    
    def _semantic_diversity(self, texts: List[str]) -> float:
        """
        Semantic diversity via average pairwise cosine distance.
        
        Higher score = more semantically diverse replies.
        """
        if self.diversity_encoder is None:
            raise ValueError("Diversity encoder not initialized")
        
        try:
            # Encode texts
            embeddings = self.diversity_encoder.encode(
                texts,
                convert_to_tensor=True,
                device=self.args.device if self.polychromic_config.cache_diversity_encoder else 'cpu',
                show_progress_bar=False
            )
            
            # Move to same device as model
            if not self.polychromic_config.cache_diversity_encoder:
                embeddings = embeddings.to(self.args.device)
            
            # Compute pairwise cosine similarities
            similarities = F.cosine_similarity(
                embeddings.unsqueeze(0),
                embeddings.unsqueeze(1),
                dim=2
            )
            
            # Get upper triangle (avoid diagonal and duplicates)
            mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
            pairwise_distances = 1 - similarities[mask]
            
            return pairwise_distances.mean().item()
        except Exception as e:
            logger.error(f"Error in semantic diversity: {e}")
            return 0.0
    
    def _bleu_diversity(self, texts: List[str]) -> float:
        """
        BLEU-based diversity (Self-BLEU inverted).
        Lower Self-BLEU = higher diversity.
        
        Returns:
            Diversity score (1 - Self-BLEU), range [0, 1]
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            scores = []
            for i, text in enumerate(texts):
                references = [
                    word_tokenize(t.lower()) 
                    for j, t in enumerate(texts) if j != i
                ]
                candidate = word_tokenize(text.lower())
                
                if references and candidate:
                    score = sentence_bleu(references, candidate, weights=(0.5, 0.5))
                    scores.append(score)
            
            if not scores:
                return 0.0
            
            # Invert: high Self-BLEU = low diversity
            avg_self_bleu = np.mean(scores)
            return 1.0 - avg_self_bleu
        except Exception as e:
            logger.error(f"Error in BLEU diversity: {e}")
            return 0.0
    
    def _distinct_n_diversity(self, texts: List[str], n: int = 2) -> float:
        """
        Distinct-n: ratio of unique n-grams to total n-grams.
        
        Args:
            texts: List of texts
            n: N-gram size (default 2 for bigrams)
            
        Returns:
            Distinct-n score, range [0, 1]
        """
        try:
            from nltk import ngrams as compute_ngrams
            from nltk.tokenize import word_tokenize
            
            all_ngrams = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                if len(tokens) >= n:
                    all_ngrams.extend(list(compute_ngrams(tokens, n)))
            
            if not all_ngrams:
                return 0.0
            
            return len(set(all_ngrams)) / len(all_ngrams)
        except Exception as e:
            logger.error(f"Error in distinct-n diversity: {e}")
            return 0.0
    
    def generate_multiple_replies(
        self,
        prompt_ids: torch.Tensor,
        n: int,
    ) -> List[str]:
        """
        Generate N diverse replies for diversity computation.
        
        Args:
            prompt_ids: Input token IDs [1, seq_len]
            n: Number of generations
            
        Returns:
            List of generated reply texts
        """
        replies = []
        
        for _ in range(n):
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        input_ids=prompt_ids,
                        max_new_tokens=self.polychromic_config.max_generation_length,
                        temperature=self.polychromic_config.diversity_temperature,
                        top_p=self.polychromic_config.diversity_top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Decode only the generated part
                    generated = outputs[0][len(prompt_ids[0]):]
                    reply = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                    
                    # Only add non-empty replies
                    if reply:
                        replies.append(reply)
                    else:
                        replies.append("[empty]")  # Placeholder for empty generation
                        
                except Exception as e:
                    logger.warning(f"Generation failed: {e}")
                    replies.append("[generation_failed]")
        
        return replies
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ):
        """
        Override compute_loss to implement polychromic objective.
        
        Loss = L_quality - λ * D(generations)
        
        This is the core research contribution.
        """
        # Standard quality loss (cross-entropy)
        outputs = model(**inputs)
        quality_loss = outputs.loss
        
        # Diversity component (expensive - compute periodically)
        diversity_score = 0.0
        if self.state.global_step % self.polychromic_config.compute_diversity_every_n_steps == 0:
            start_time = time.time()
            diversity_score = self._compute_batch_diversity(inputs)
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
        
        # Combined loss
        combined_loss = quality_loss - self.polychromic_config.diversity_weight * diversity_score
        
        # Tracking
        self.quality_losses_history.append(quality_loss.item())
        self.diversity_scores_history.append(diversity_score)
        
        # Logging to W&B
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                'train/quality_loss': quality_loss.item(),
                'train/diversity_score': diversity_score,
                'train/combined_loss': combined_loss.item(),
                'train/diversity_weight': self.polychromic_config.diversity_weight,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/epoch': self.state.epoch,
            }
            
            # Add generation time stats if available
            if self.generation_times:
                log_dict['train/avg_generation_time'] = np.mean(self.generation_times[-10:])
            
            wandb.log(log_dict, step=self.state.global_step)
        
        return (combined_loss, outputs) if return_outputs else combined_loss
    
    def _compute_batch_diversity(self, inputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute diversity score for current batch.
        
        For each example in batch:
        1. Extract prompt (everything before reply starts)
        2. Generate N diverse replies
        3. Compute diversity score
        4. Average across batch
        
        Returns:
            Average diversity score for the batch
        """
        batch_diversity_scores = []
        
        # Process each example in batch
        input_ids = inputs['input_ids']
        labels = inputs.get('labels', None)
        
        # Limit number of examples to process for computational efficiency
        num_examples = min(
            len(input_ids),
            self.polychromic_config.max_examples_for_diversity
        )
        
        for i in range(num_examples):
            try:
                # Find where the reply starts
                # Labels are -100 for prompt tokens, actual tokens for reply
                if labels is not None:
                    valid_mask = labels[i] != -100
                    if valid_mask.any():
                        reply_start = valid_mask.nonzero(as_tuple=True)[0][0].item()
                    else:
                        # Fallback if no valid labels
                        reply_start = len(input_ids[i]) // 2
                else:
                    # Fallback: assume first 70% is prompt
                    reply_start = int(len(input_ids[i]) * 0.7)
                
                # Extract prompt
                prompt_ids = input_ids[i, :reply_start].unsqueeze(0)
                
                # Generate N replies
                replies = self.generate_multiple_replies(
                    prompt_ids.to(self.args.device),
                    n=self.polychromic_config.n_generations
                )
                
                # Compute diversity for this example
                if len(replies) >= 2:
                    diversity = self.compute_diversity_score(replies)
                    batch_diversity_scores.append(diversity)
                    
            except Exception as e:
                logger.warning(f"Error computing diversity for example {i}: {e}")
                continue
        
        # Average across batch (or return 0 if no valid scores)
        if batch_diversity_scores:
            return np.mean(batch_diversity_scores)
        else:
            return 0.0
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval"
    ):
        """
        Override evaluation to include diversity metrics.
        """
        # Standard evaluation
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add diversity evaluation
        logger.info("Computing diversity metrics on validation set...")
        diversity_metrics = self._evaluate_diversity(dataloader)
        output.metrics.update(diversity_metrics)
        
        # Log to W&B
        wandb.log({
            f'{metric_key_prefix}/loss': output.metrics.get(f'{metric_key_prefix}_loss', 0),
            f'{metric_key_prefix}/diversity': diversity_metrics.get('diversity_score', 0),
            f'{metric_key_prefix}/avg_reply_length': diversity_metrics.get('avg_reply_length', 0),
            f'{metric_key_prefix}/avg_unique_words': diversity_metrics.get('avg_unique_words', 0),
        }, step=self.state.global_step)
        
        return output
    
    def _evaluate_diversity(self, dataloader, num_samples=50) -> Dict[str, float]:
        """
        Evaluate diversity on validation set.
        
        Args:
            dataloader: Validation dataloader
            num_samples: Number of examples to evaluate
            
        Returns:
            Dictionary of diversity metrics
        """
        self.model.eval()
        all_replies = []
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            # Generate replies for examples in this batch
            for i in range(len(batch['input_ids'])):
                if sample_count >= num_samples:
                    break
                
                try:
                    # Find prompt
                    labels = batch.get('labels', None)
                    if labels is not None:
                        valid_mask = labels[i] != -100
                        if valid_mask.any():
                            reply_start = valid_mask.nonzero(as_tuple=True)[0][0].item()
                        else:
                            reply_start = len(batch['input_ids'][i]) // 2
                    else:
                        reply_start = int(len(batch['input_ids'][i]) * 0.7)
                    
                    prompt_ids = batch['input_ids'][i, :reply_start].unsqueeze(0)
                    
                    # Generate multiple replies
                    replies = self.generate_multiple_replies(prompt_ids, n=5)
                    all_replies.extend(replies)
                    sample_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error in diversity evaluation: {e}")
                    continue
        
        # Compute diversity metrics
        if len(all_replies) >= 2:
            diversity_score = self.compute_diversity_score(all_replies)
        else:
            diversity_score = 0.0
        
        # Additional metrics
        avg_length = np.mean([len(r.split()) for r in all_replies]) if all_replies else 0
        avg_unique_words = np.mean([
            len(set(r.lower().split())) 
            for r in all_replies
        ]) if all_replies else 0
        
        self.model.train()
        
        return {
            'diversity_score': diversity_score,
            'avg_reply_length': avg_length,
            'avg_unique_words': avg_unique_words,
            'num_samples_evaluated': sample_count,
        }

