"""
Pass@k Evaluation

Key metric for polychromic training:
- Generate k replies
- Pick the best one
- Polychromic should outperform baseline at higher k values

This simulates real-world usage where you generate multiple
options and select the best.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
import logging

logger = logging.getLogger(__name__)


def compute_passk(
    prompts: List[str],
    model_generate_func: Callable,
    quality_checker: Callable[[str], bool],
    k_values: List[int] = [1, 5, 10],
    n_total: int = 10,
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute Pass@k metric.
    
    Pass@k = Probability that at least one of k generations passes quality check
    
    Args:
        prompts: List of input prompts
        model_generate_func: Function that generates n replies for a prompt
                             Signature: (prompt: str, n: int) -> List[str]
        quality_checker: Function that checks if a reply is "good"
                        Signature: (reply: str) -> bool
        k_values: List of k values to compute (e.g., [1, 5, 10])
        n_total: Total number of generations per prompt
        verbose: Print progress
        
    Returns:
        Dictionary mapping k -> pass@k score
    """
    logger.info(f"Computing Pass@k for k in {k_values}")
    logger.info(f"  Total prompts: {len(prompts)}")
    logger.info(f"  Generations per prompt: {n_total}")
    
    results = {k: [] for k in k_values}
    
    for i, prompt in enumerate(prompts):
        if verbose and i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(prompts)}")
        
        # Generate n_total replies
        replies = model_generate_func(prompt, n_total)
        
        # Check which ones pass
        passed_indices = [
            idx for idx, reply in enumerate(replies)
            if quality_checker(reply)
        ]
        
        # For each k, check if at least one in first k passes
        for k in k_values:
            if k > n_total:
                logger.warning(f"k={k} > n_total={n_total}, skipping")
                continue
            
            # Check if any of first k passed
            passed_in_k = any(idx < k for idx in passed_indices)
            results[k].append(1 if passed_in_k else 0)
    
    # Compute averages
    passk_scores = {
        k: np.mean(passes) if passes else 0.0
        for k, passes in results.items()
    }
    
    logger.info("\nPass@k Results:")
    for k, score in passk_scores.items():
        logger.info(f"  Pass@{k}: {score:.3f} ({score*100:.1f}%)")
    
    return passk_scores


def compare_passk(
    prompts: List[str],
    model_a_generate: Callable,
    model_b_generate: Callable,
    quality_checker: Callable,
    k_values: List[int] = [1, 5, 10],
    n_total: int = 10
) -> Dict:
    """
    Compare Pass@k between two models.
    
    Args:
        prompts: Test prompts
        model_a_generate: Generation function for model A
        model_b_generate: Generation function for model B
        quality_checker: Quality checking function
        k_values: List of k values
        n_total: Total generations
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("="*60)
    logger.info("Pass@k Comparison")
    logger.info("="*60)
    
    # Compute Pass@k for both models
    logger.info("\nModel A:")
    passk_a = compute_passk(
        prompts,
        model_a_generate,
        quality_checker,
        k_values,
        n_total
    )
    
    logger.info("\nModel B:")
    passk_b = compute_passk(
        prompts,
        model_b_generate,
        quality_checker,
        k_values,
        n_total
    )
    
    # Compute improvements
    logger.info("\n" + "="*60)
    logger.info("Improvements (B over A):")
    logger.info("="*60)
    
    improvements = {}
    for k in k_values:
        if k in passk_a and k in passk_b:
            abs_improvement = passk_b[k] - passk_a[k]
            rel_improvement = (abs_improvement / passk_a[k] * 100) if passk_a[k] > 0 else 0
            
            improvements[k] = {
                'absolute': abs_improvement,
                'relative_pct': rel_improvement
            }
            
            logger.info(f"Pass@{k}:")
            logger.info(f"  Model A: {passk_a[k]:.3f}")
            logger.info(f"  Model B: {passk_b[k]:.3f}")
            logger.info(f"  Improvement: {abs_improvement:+.3f} ({rel_improvement:+.1f}%)")
    
    logger.info("="*60 + "\n")
    
    return {
        'model_a': passk_a,
        'model_b': passk_b,
        'improvements': improvements
    }


class HeuristicQualityChecker:
    """
    Simple heuristic-based quality checker.
    
    Checks:
    - Minimum length
    - Maximum length
    - Not repetitive
    - Relevant to prompt
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 300,
        max_repetition_ratio: float = 0.5
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_repetition_ratio = max_repetition_ratio
    
    def __call__(self, reply: str, prompt: Optional[str] = None) -> bool:
        """Check if reply passes quality criteria."""
        
        # Length check
        if len(reply) < self.min_length or len(reply) > self.max_length:
            return False
        
        # Repetition check
        words = reply.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = 1 - (unique_words / len(words))
            if repetition_ratio > self.max_repetition_ratio:
                return False
        
        # Basic content checks
        if reply.strip() == "":
            return False
        
        if reply.startswith("[") or reply.startswith("Error"):
            return False
        
        return True


class LLMQualityChecker:
    """
    LLM-based quality checker.
    
    Uses a simple binary classification:
    "Is this a good Twitter reply? Yes/No"
    """
    
    def __init__(self, llm_judge):
        self.llm_judge = llm_judge
    
    def __call__(self, reply: str, prompt: str) -> bool:
        """Check if reply is high quality using LLM."""
        
        check_prompt = f"""Is this a good Twitter reply?

Original tweet: "{prompt}"
Reply: "{reply}"

Answer with just "Yes" or "No"."""
        
        try:
            response = self.llm_judge._call_llm(check_prompt)
            return "yes" in response.lower()
        except:
            # Fallback to heuristic
            return HeuristicQualityChecker()(reply, prompt)

