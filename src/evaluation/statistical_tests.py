"""
Statistical Significance Testing

Essential for Arxiv-quality research:
- Mann-Whitney U test (non-parametric)
- Paired t-test (parametric)
- Cohen's d (effect size)
- Cliff's Delta (ordinal effect size)
- Bootstrap confidence intervals

We need p < 0.05 and meaningful effect size to claim success.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
import logging
import math

logger = logging.getLogger(__name__)


def sanitize_for_json(value):
    """Convert numpy types and handle NaN/Inf for JSON serialization."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    
    return value


def mann_whitney_u_test(
    group_a: List[float],
    group_b: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Mann-Whitney U test (non-parametric).
    
    Use when:
    - Data not normally distributed
    - Different sample sizes
    - Ordinal data (ratings)
    
    Args:
        group_a: Scores from model A
        group_b: Scores from model B
        alternative: 'two-sided', 'less', 'greater'
        
    Returns:
        Dictionary with statistic and p-value
    """
    try:
        statistic, p_value = stats.mannwhitneyu(
            group_a,
            group_b,
            alternative=alternative
        )
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
    except Exception as e:
        logger.error(f"Error in Mann-Whitney U test: {e}")
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}


def paired_t_test(
    group_a: List[float],
    group_b: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Paired t-test (parametric).
    
    Use when:
    - Same examples evaluated by both models (paired)
    - Data approximately normal
    - Equal sample sizes
    
    Args:
        group_a: Scores from model A
        group_b: Scores from model B
        alternative: 'two-sided', 'less', 'greater'
        
    Returns:
        Dictionary with statistic and p-value
    """
    try:
        statistic, p_value = stats.ttest_rel(
            group_a,
            group_b,
            alternative=alternative
        )
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
    except Exception as e:
        logger.error(f"Error in paired t-test: {e}")
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}


def compute_cohens_d(
    group_a: List[float],
    group_b: List[float]
) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    Args:
        group_a: Scores from model A
        group_b: Scores from model B
        
    Returns:
        Cohen's d value
    """
    try:
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        n_a = len(group_a)
        n_b = len(group_b)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        # Cohen's d
        cohens_d = (mean_a - mean_b) / pooled_std
        
        return float(cohens_d)
        
    except Exception as e:
        logger.error(f"Error computing Cohen's d: {e}")
        return 0.0


def compute_cliffs_delta(
    group_a: List[float],
    group_b: List[float]
) -> float:
    """
    Compute Cliff's Delta effect size (ordinal).
    
    Interpretation:
    - |δ| < 0.147: negligible
    - 0.147 ≤ |δ| < 0.33: small
    - 0.33 ≤ |δ| < 0.474: medium
    - |δ| ≥ 0.474: large
    
    Args:
        group_a: Scores from model A
        group_b: Scores from model B
        
    Returns:
        Cliff's Delta value (-1 to 1)
    """
    try:
        n_a = len(group_a)
        n_b = len(group_b)
        
        # Count dominances
        dominance = 0
        for a in group_a:
            for b in group_b:
                if a > b:
                    dominance += 1
                elif a < b:
                    dominance -= 1
        
        # Cliff's delta
        delta = dominance / (n_a * n_b)
        
        return float(delta)
        
    except Exception as e:
        logger.error(f"Error computing Cliff's Delta: {e}")
        return 0.0


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func=np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: Data points
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        (statistic, lower_bound, upper_bound)
    """
    try:
        # Compute original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return float(original_stat), float(lower_bound), float(upper_bound)
        
    except Exception as e:
        logger.error(f"Error in bootstrap CI: {e}")
        return 0.0, 0.0, 0.0


def comprehensive_statistical_comparison(
    baseline_scores: List[float],
    polychromic_scores: List[float],
    metric_name: str = "score"
) -> Dict:
    """
    Perform comprehensive statistical comparison.
    
    Args:
        baseline_scores: Scores from baseline model
        polychromic_scores: Scores from polychromic model
        metric_name: Name of the metric being compared
        
    Returns:
        Dictionary with all statistical tests and effect sizes
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Statistical Comparison: {metric_name}")
    logger.info(f"{'='*60}")
    
    results = {
        'metric_name': metric_name,
        'baseline_mean': sanitize_for_json(np.mean(baseline_scores)),
        'baseline_std': sanitize_for_json(np.std(baseline_scores)),
        'polychromic_mean': sanitize_for_json(np.mean(polychromic_scores)),
        'polychromic_std': sanitize_for_json(np.std(polychromic_scores)),
        'improvement': sanitize_for_json(np.mean(polychromic_scores) - np.mean(baseline_scores)),
        'improvement_pct': sanitize_for_json(100 * (np.mean(polychromic_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores))
    }
    
    # Mann-Whitney U test
    logger.info("\nMann-Whitney U Test (non-parametric):")
    mw_test = mann_whitney_u_test(baseline_scores, polychromic_scores, alternative='less')
    results['mann_whitney'] = mw_test
    logger.info(f"  p-value: {mw_test['p_value']:.6f}")
    logger.info(f"  Significant: {mw_test['significant']}")
    
    # Paired t-test (if same length)
    if len(baseline_scores) == len(polychromic_scores):
        logger.info("\nPaired t-test:")
        t_test = paired_t_test(baseline_scores, polychromic_scores, alternative='less')
        results['paired_t_test'] = t_test
        logger.info(f"  p-value: {t_test['p_value']:.6f}")
        logger.info(f"  Significant: {t_test['significant']}")
    
    # Effect sizes
    logger.info("\nEffect Sizes:")
    cohens_d = compute_cohens_d(polychromic_scores, baseline_scores)
    cliffs_delta = compute_cliffs_delta(polychromic_scores, baseline_scores)
    results['cohens_d'] = sanitize_for_json(cohens_d)
    results['cliffs_delta'] = sanitize_for_json(cliffs_delta)
    
    logger.info(f"  Cohen's d: {cohens_d if cohens_d is not None else 'NaN'}")
    logger.info(f"  Cliff's Delta: {cliffs_delta:.4f}")
    
    # Interpret effect size
    if cohens_d is None or math.isnan(cohens_d):
        effect_interpretation = "unable_to_compute"
    elif abs(cohens_d) >= 0.8:
        effect_interpretation = "large"
    elif abs(cohens_d) >= 0.5:
        effect_interpretation = "medium"
    elif abs(cohens_d) >= 0.2:
        effect_interpretation = "small"
    else:
        effect_interpretation = "negligible"
    
    results['effect_size_interpretation'] = effect_interpretation
    logger.info(f"  Interpretation: {effect_interpretation}")
    
    # Bootstrap confidence intervals
    logger.info("\nBootstrap 95% Confidence Intervals:")
    baseline_ci = bootstrap_confidence_interval(baseline_scores)
    polychromic_ci = bootstrap_confidence_interval(polychromic_scores)
    
    results['baseline_ci'] = {
        'mean': sanitize_for_json(baseline_ci[0]),
        'lower': sanitize_for_json(baseline_ci[1]),
        'upper': sanitize_for_json(baseline_ci[2])
    }
    results['polychromic_ci'] = {
        'mean': sanitize_for_json(polychromic_ci[0]),
        'lower': sanitize_for_json(polychromic_ci[1]),
        'upper': sanitize_for_json(polychromic_ci[2])
    }
    
    logger.info(f"  Baseline: {baseline_ci[0]:.4f} [{baseline_ci[1]:.4f}, {baseline_ci[2]:.4f}]")
    logger.info(f"  Polychromic: {polychromic_ci[0]:.4f} [{polychromic_ci[1]:.4f}, {polychromic_ci[2]:.4f}]")
    
    # Overall conclusion
    logger.info("\nConclusion:")
    if mw_test['significant'] and abs(cohens_d) >= 0.3:
        conclusion = "Statistically significant improvement with meaningful effect size"
        results['conclusion'] = "significant"
    elif mw_test['significant']:
        conclusion = "Statistically significant but small effect size"
        results['conclusion'] = "significant_small"
    else:
        conclusion = "No statistically significant difference"
        results['conclusion'] = "not_significant"
    
    logger.info(f"  {conclusion}")
    logger.info(f"{'='*60}\n")
    
    return results

