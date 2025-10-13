"""
Diversity Metrics for Text Generation

Implements:
- Self-BLEU: Lower is more diverse
- Distinct-n: Higher is more diverse  
- Semantic diversity: Higher is more diverse
- Vocabulary richness

These are critical for measuring whether polychromic training
actually increases diversity.
"""

import numpy as np
from typing import List, Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def ensure_nltk_data():
    """Ensure NLTK data is downloaded."""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        # Test if punkt works
        word_tokenize("test")
    except LookupError:
        logger.warning("NLTK punkt data missing - downloading...")
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # For newer NLTK versions
        logger.info("✓ NLTK punkt data downloaded")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")


# Download NLTK data at module import time
ensure_nltk_data()


def compute_self_bleu(texts: List[str], n_gram: int = 4) -> float:
    """
    Compute Self-BLEU score.
    
    Lower Self-BLEU = more diverse (texts are less similar to each other)
    
    Args:
        texts: List of generated texts
        n_gram: Maximum n-gram size (default 4)
        
    Returns:
        Average Self-BLEU score (lower = more diverse)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        if len(texts) < 2:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for i, text in enumerate(texts):
            # Get all other texts as references
            references = [
                word_tokenize(t.lower())
                for j, t in enumerate(texts) if j != i
            ]
            candidate = word_tokenize(text.lower())
            
            if references and candidate:
                # Weights for 1-4 grams
                weights = tuple([1.0/n_gram] * n_gram)
                score = sentence_bleu(
                    references,
                    candidate,
                    weights=weights,
                    smoothing_function=smoothing
                )
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
        
    except Exception as e:
        logger.error(f"Error computing Self-BLEU: {e}")
        return 0.0


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n metric.
    
    Higher Distinct-n = more diverse vocabulary
    
    Args:
        texts: List of generated texts
        n: N-gram size (default 2 for bigrams)
        
    Returns:
        Distinct-n score (ratio of unique n-grams to total)
    """
    try:
        from nltk import ngrams as compute_ngrams
        from nltk.tokenize import word_tokenize
        
        all_ngrams = []
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            if len(tokens) >= n:
                text_ngrams = list(compute_ngrams(tokens, n))
                all_ngrams.extend(text_ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams
        
    except Exception as e:
        logger.error(f"Error computing Distinct-{n}: {e}")
        return 0.0


def compute_semantic_diversity(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Compute semantic diversity via average pairwise cosine distance.
    
    Higher score = texts are semantically more diverse
    
    Args:
        texts: List of generated texts
        model_name: Sentence transformer model
        
    Returns:
        Average pairwise cosine distance (higher = more diverse)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        import torch.nn.functional as F
        
        if len(texts) < 2:
            return 0.0
        
        # Load encoder
        encoder = SentenceTransformer(model_name)
        
        # Encode texts
        embeddings = encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Compute pairwise cosine similarities
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(1),
            dim=2
        )
        
        # Get upper triangle (avoid diagonal and duplicates)
        mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
        pairwise_distances = 1 - similarities[mask]
        
        return float(pairwise_distances.mean().item())
        
    except Exception as e:
        logger.error(f"Error computing semantic diversity: {e}")
        return 0.0


def compute_vocabulary_richness(texts: List[str]) -> Dict[str, float]:
    """
    Compute vocabulary richness metrics.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with:
        - unique_words: Number of unique words
        - total_words: Total number of words
        - type_token_ratio: unique/total ratio
        - avg_word_length: Average word length
    """
    try:
        from nltk.tokenize import word_tokenize
        
        all_words = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            # Filter out punctuation
            words = [t for t in tokens if t.isalnum()]
            all_words.extend(words)
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        if total_words == 0:
            return {
                'unique_words': 0,
                'total_words': 0,
                'type_token_ratio': 0.0,
                'avg_word_length': 0.0
            }
        
        type_token_ratio = unique_words / total_words
        avg_word_length = np.mean([len(w) for w in all_words])
        
        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'type_token_ratio': float(type_token_ratio),
            'avg_word_length': float(avg_word_length)
        }
        
    except Exception as e:
        logger.error(f"Error computing vocabulary richness: {e}")
        return {
            'unique_words': 0,
            'total_words': 0,
            'type_token_ratio': 0.0,
            'avg_word_length': 0.0
        }


def compute_all_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute all diversity metrics at once.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with all diversity scores
    """
    logger.info(f"Computing diversity metrics for {len(texts)} texts...")
    
    metrics = {}
    
    # Self-BLEU (lower = more diverse)
    logger.info("  Computing Self-BLEU...")
    metrics['self_bleu'] = compute_self_bleu(texts)
    
    # Distinct-n (higher = more diverse)
    logger.info("  Computing Distinct-n...")
    metrics['distinct_1'] = compute_distinct_n(texts, n=1)
    metrics['distinct_2'] = compute_distinct_n(texts, n=2)
    metrics['distinct_3'] = compute_distinct_n(texts, n=3)
    
    # Semantic diversity (higher = more diverse)
    logger.info("  Computing semantic diversity...")
    metrics['semantic_diversity'] = compute_semantic_diversity(texts)
    
    # Vocabulary richness
    logger.info("  Computing vocabulary richness...")
    vocab_metrics = compute_vocabulary_richness(texts)
    metrics.update(vocab_metrics)
    
    logger.info("Diversity metrics computed!")
    return metrics

