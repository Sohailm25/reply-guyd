"""
Unified Metrics Module

Consolidates all evaluation metrics in one place:
- Diversity metrics (Self-BLEU, Distinct-n, semantic diversity)
- Quality metrics (ROUGE, BERTScore, perplexity)
- Pass@k metrics
- Statistical tests

This provides a single import point for all evaluation needs.
"""

# Re-export diversity metrics
from .diversity_metrics import (
    compute_self_bleu,
    compute_distinct_n,
    compute_semantic_diversity,
    compute_all_diversity_metrics
)

# Re-export quality metrics
from .quality_metrics import (
    compute_rouge_scores,
    compute_bertscore,
    compute_perplexity
)

# Re-export statistical tests
from .statistical_tests import (
    mann_whitney_u_test,
    paired_t_test,
    compute_cohens_d,
    compute_cliffs_delta,
    bootstrap_confidence_interval,
    comprehensive_statistical_comparison
)

# Re-export Pass@k
from .passk_evaluation import (
    compute_passk,
    HeuristicQualityChecker
)

# Re-export novel metrics
from .novel_metrics import (
    compute_diversity_efficiency_ratio,
    find_collapse_point,
    compute_user_selection_quality
)

__all__ = [
    # Diversity metrics
    'compute_self_bleu',
    'compute_distinct_n',
    'compute_semantic_diversity',
    'compute_all_diversity_metrics',
    
    # Quality metrics
    'compute_rouge_scores',
    'compute_bertscore',
    'compute_perplexity',
    
    # Pass@k metrics
    'compute_passk',
    'HeuristicQualityChecker',
    
    # Novel metrics
    'compute_diversity_efficiency_ratio',
    'find_collapse_point',
    'compute_user_selection_quality',
    
    # Statistical tests
    'mann_whitney_u_test',
    'paired_t_test',
    'compute_cohens_d',
    'compute_cliffs_delta',
    'bootstrap_confidence_interval',
    'comprehensive_statistical_comparison',
]

