"""
Evaluation modules for polychromic training research.

This package contains:
- Unified metrics module (diversity, quality, Pass@k, statistical tests)
- Benchmark runner for comprehensive evaluation
- LLM-as-judge evaluation
- Novel metrics (DER, collapse points)

For new code, use:
    from src.evaluation.metrics import *
    from src.evaluation.benchmark import EvaluationBenchmark
"""

# Import from unified metrics module (recommended)
from .metrics import *

# Import benchmark runner
from .benchmark import EvaluationBenchmark, compare_models

# Legacy imports (for backward compatibility)
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
    # New unified interface
    'EvaluationBenchmark',
    'compare_models',
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
    'comprehensive_statistical_comparison',
]

