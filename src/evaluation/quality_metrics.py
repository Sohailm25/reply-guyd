"""
Quality Metrics for Generated Replies

Implements:
- ROUGE scores (reference-based similarity)
- BERTScore (semantic similarity)
- Perplexity (language model quality)
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_rouge_scores(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: Generated replies
        references: Ground truth replies
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        # Average scores
        return {
            'rouge1': float(np.mean(scores['rouge1'])),
            'rouge2': float(np.mean(scores['rouge2'])),
            'rougeL': float(np.mean(scores['rougeL']))
        }
        
    except Exception as e:
        logger.error(f"Error computing ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli"
) -> Dict[str, float]:
    """
    Compute BERTScore (semantic similarity).
    
    Args:
        predictions: Generated replies
        references: Ground truth replies
        model_type: BERT model to use
        
    Returns:
        Dictionary with precision, recall, F1
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            predictions,
            references,
            model_type=model_type,
            verbose=False
        )
        
        return {
            'bertscore_precision': float(P.mean().item()),
            'bertscore_recall': float(R.mean().item()),
            'bertscore_f1': float(F1.mean().item())
        }
        
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }


def compute_perplexity(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda"
) -> float:
    """
    Compute perplexity of generated texts.
    
    Lower perplexity = more fluent/natural text
    
    Args:
        texts: Generated texts
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        
    Returns:
        Average perplexity
    """
    try:
        import torch
        
        model.eval()
        perplexities = []
        
        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(text, return_tensors='pt').to(device)
                
                outputs = model(**encodings, labels=encodings['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                perplexities.append(perplexity.item())
        
        return float(np.mean(perplexities))
        
    except Exception as e:
        logger.error(f"Error computing perplexity: {e}")
        return float('inf')


def compute_engagement_correlation(
    predictions: List[str],
    ground_truth_engagement: List[int],
    predicted_engagement: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute correlation between predicted and actual engagement.
    
    Args:
        predictions: Generated replies
        ground_truth_engagement: Actual engagement (likes/retweets)
        predicted_engagement: Predicted engagement scores (optional)
        
    Returns:
        Dictionary with correlation metrics
    """
    try:
        from scipy.stats import spearmanr, pearsonr
        
        if predicted_engagement is None:
            # Use text length as simple predictor
            predicted_engagement = [len(p) for p in predictions]
        
        # Spearman correlation (rank-based, non-parametric)
        spearman_corr, spearman_p = spearmanr(
            predicted_engagement,
            ground_truth_engagement
        )
        
        # Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(
            predicted_engagement,
            ground_truth_engagement
        )
        
        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p)
        }
        
    except Exception as e:
        logger.error(f"Error computing engagement correlation: {e}")
        return {
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0
        }

