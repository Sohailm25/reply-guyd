"""
Evaluation modules for polychromic training research.

This package contains:
- Diversity metrics (Self-BLEU, Distinct-n, semantic diversity)
- Quality metrics (ROUGE, BERTScore, perplexity)
- LLM-as-judge evaluation
- Statistical significance testing
- Pass@k evaluation
"""

from .diversity_metrics import (
    compute_self_bleu,
    compute_distinct_n,
    compute_semantic_diversity,
    compute_all_diversity_metrics
)

from .quality_metrics import (
    compute_rouge_scores,
    compute_bertscore,
    compute_perplexity
)

from .statistical_tests import (
    mann_whitney_u_test,
    paired_t_test,
    compute_cohens_d,
    compute_cliffs_delta,
    bootstrap_confidence_interval,
    comprehensive_statistical_comparison
)

__all__ = [
    # Diversity
    'compute_self_bleu',
    'compute_distinct_n',
    'compute_semantic_diversity',
    'compute_all_diversity_metrics',
    # Quality
    'compute_rouge_scores',
    'compute_bertscore',
    'compute_perplexity',
    # Statistics
    'mann_whitney_u_test',
    'paired_t_test',
    'compute_cohens_d',
    'compute_cliffs_delta',
    'bootstrap_confidence_interval',
]

