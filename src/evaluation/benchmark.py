"""
Unified Evaluation Benchmark Runner

Provides a clean interface for comprehensive model evaluation.
Integrates with the environment abstraction for clean separation of concerns.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import json
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluationBenchmark:
    """
    Run comprehensive evaluation on models.
    
    Combines:
    - Diversity metrics
    - Quality metrics  
    - Pass@k evaluation
    - LLM-as-judge (optional)
    - Statistical significance tests
    
    Example:
        >>> from src.environments import TwitterReplyEnvironment
        >>> from src.evaluation.benchmark import EvaluationBenchmark
        >>> 
        >>> env = TwitterReplyEnvironment(model, tokenizer)
        >>> benchmark = EvaluationBenchmark(env, test_data)
        >>> results = benchmark.run(n_generations=10)
    """
    
    def __init__(
        self,
        environment,
        test_data: List[Dict[str, Any]],
        llm_judge: Optional[Any] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize evaluation benchmark.
        
        Args:
            environment: Environment to evaluate (e.g., TwitterReplyEnvironment)
            test_data: List of test examples (dicts with 'tweet' and optionally 'reply')
            llm_judge: Optional LLM judge for pairwise comparison
            metrics: List of metrics to compute (None = all)
        """
        self.env = environment
        self.test_data = test_data
        self.llm_judge = llm_judge
        self.metrics = metrics or [
            'diversity',
            'quality', 
            'passk',
            'rewards',
            'statistics'
        ]
        
        logger.info(f"EvaluationBenchmark initialized with {len(test_data)} examples")
        logger.info(f"Metrics to compute: {', '.join(self.metrics)}")
    
    def run(
        self,
        n_generations: int = 10,
        max_examples: Optional[int] = None,
        save_generations: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite.
        
        Args:
            n_generations: Number of candidates to generate per prompt
            max_examples: Maximum number of examples to evaluate (None = all)
            save_generations: Whether to save all generated texts
            temperature: Generation temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict with all evaluation results
        """
        logger.info("="*60)
        logger.info("Starting Comprehensive Evaluation")
        logger.info("="*60)
        logger.info(f"  Examples: {min(len(self.test_data), max_examples or len(self.test_data))}")
        logger.info(f"  Generations per example: {n_generations}")
        logger.info(f"  Temperature: {temperature}")
        logger.info("="*60)
        
        results = {
            "config": {
                "n_generations": n_generations,
                "max_examples": max_examples,
                "temperature": temperature,
                "top_p": top_p,
                "num_test_examples": len(self.test_data)
            },
            "generations": [],
            "metrics": {}
        }
        
        # Limit number of examples if requested
        test_subset = self.test_data[:max_examples] if max_examples else self.test_data
        
        # Generate responses for all test examples
        logger.info("\nGenerating responses...")
        all_generations = []
        all_tweets = []
        all_references = []
        all_rewards = []
        
        for example in tqdm(test_subset, desc="Generating"):
            tweet = example['tweet']
            reference = example.get('reply', '')
            
            # Reset environment
            self.env.reset(tweet)
            
            # Generate candidates
            try:
                candidates = self.env.generate_candidates(
                    n=n_generations,
                    temperature=temperature,
                    top_p=top_p
                )
            except Exception as e:
                logger.error(f"Generation failed for tweet: {tweet[:50]}... Error: {e}")
                candidates = [""] * n_generations  # Fallback
            
            # Compute rewards for candidates
            rewards = [self.env.compute_reward(tweet, c) for c in candidates]
            
            # Store
            all_generations.append(candidates)
            all_tweets.append(tweet)
            all_references.append(reference)
            all_rewards.append(rewards)
            
            if save_generations:
                results["generations"].append({
                    "tweet": tweet,
                    "reference": reference,
                    "candidates": candidates,
                    "rewards": rewards
                })
        
        # Compute metrics
        logger.info("\nComputing metrics...")
        
        # 1. Diversity metrics
        if 'diversity' in self.metrics:
            logger.info("  Computing diversity metrics...")
            results["metrics"]["diversity"] = self._compute_diversity_metrics(all_generations)
        
        # 2. Quality metrics (if references available)
        if 'quality' in self.metrics and all_references[0]:
            logger.info("  Computing quality metrics...")
            results["metrics"]["quality"] = self._compute_quality_metrics(
                all_generations, all_references
            )
        
        # 3. Pass@k metrics (if references available)
        if 'passk' in self.metrics and all_references[0]:
            logger.info("  Computing Pass@k metrics...")
            results["metrics"]["passk"] = self._compute_passk_metrics(
                all_generations, all_references
            )
        
        # 4. Reward statistics
        if 'rewards' in self.metrics:
            logger.info("  Computing reward statistics...")
            results["metrics"]["rewards"] = self._compute_reward_statistics(all_rewards)
        
        # 5. LLM judge (if available)
        if self.llm_judge and 'llm_judge' in self.metrics:
            logger.info("  Running LLM judge...")
            results["metrics"]["llm_judge"] = self._run_llm_judge(
                all_tweets, all_generations
            )
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Complete!")
        logger.info("="*60)
        
        return results
    
    def _compute_diversity_metrics(self, all_generations: List[List[str]]) -> Dict[str, Any]:
        """Compute diversity metrics across all generations."""
        from .metrics import compute_all_diversity_metrics
        
        # Flatten all generations
        all_texts = [text for gen_list in all_generations for text in gen_list]
        
        # Compute metrics
        diversity = compute_all_diversity_metrics(all_texts)
        
        # Also compute per-example diversity
        per_example_diversity = []
        for gen_list in all_generations:
            if len(gen_list) > 1:
                ex_div = compute_all_diversity_metrics(gen_list)
                per_example_diversity.append(ex_div.get('self_bleu', 0))
        
        diversity['avg_per_example_self_bleu'] = (
            sum(per_example_diversity) / len(per_example_diversity)
            if per_example_diversity else 0.0
        )
        
        return diversity
    
    def _compute_quality_metrics(
        self,
        all_generations: List[List[str]],
        all_references: List[str]
    ) -> Dict[str, Any]:
        """Compute quality metrics comparing to references."""
        from .metrics import compute_rouge_scores, compute_bertscore
        
        # Take best generation per example (highest reward or first)
        best_generations = [gen_list[0] for gen_list in all_generations]
        
        # ROUGE scores
        rouge = compute_rouge_scores(best_generations, all_references)
        
        # BERTScore (if available)
        try:
            bertscore = compute_bertscore(best_generations, all_references)
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            bertscore = {}
        
        return {
            "rouge": rouge,
            "bertscore": bertscore
        }
    
    def _compute_passk_metrics(
        self,
        all_generations: List[List[str]],
        all_references: List[str]
    ) -> Dict[str, Any]:
        """Compute Pass@k metrics."""
        from .metrics import compute_passk, HeuristicQualityChecker
        
        quality_checker = HeuristicQualityChecker(all_references)
        
        # Compute for different k values
        k_values = [1, 3, 5, 10]
        passk_results = {}
        
        for k in k_values:
            if k <= len(all_generations[0]):
                passk = compute_passk(
                    all_generations,
                    quality_checker,
                    k=k
                )
                passk_results[f"pass@{k}"] = passk
        
        return passk_results
    
    def _compute_reward_statistics(self, all_rewards: List[List[float]]) -> Dict[str, Any]:
        """Compute statistics on rewards."""
        import numpy as np
        
        # Flatten
        flat_rewards = [r for reward_list in all_rewards for r in reward_list]
        
        # Per-example statistics
        per_example_means = [np.mean(rewards) for rewards in all_rewards]
        per_example_maxs = [np.max(rewards) for rewards in all_rewards]
        per_example_stds = [np.std(rewards) for rewards in all_rewards]
        
        return {
            "overall": {
                "mean": float(np.mean(flat_rewards)),
                "std": float(np.std(flat_rewards)),
                "min": float(np.min(flat_rewards)),
                "max": float(np.max(flat_rewards)),
                "median": float(np.median(flat_rewards))
            },
            "per_example": {
                "mean_of_means": float(np.mean(per_example_means)),
                "mean_of_maxs": float(np.mean(per_example_maxs)),
                "mean_of_stds": float(np.mean(per_example_stds))
            }
        }
    
    def _run_llm_judge(
        self,
        all_tweets: List[str],
        all_generations: List[List[str]]
    ) -> Dict[str, Any]:
        """Run LLM-as-judge evaluation."""
        # Placeholder for LLM judge
        # In practice, would compare top-k candidates
        logger.warning("LLM judge not yet integrated with benchmark")
        return {}
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Results dict from run()
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print summary of evaluation results.
        
        Args:
            results: Results dict from run()
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        metrics = results.get("metrics", {})
        
        # Diversity
        if "diversity" in metrics:
            div = metrics["diversity"]
            print("\nDiversity Metrics:")
            print(f"  Self-BLEU:   {div.get('self_bleu', 0):.3f} (lower = more diverse)")
            print(f"  Distinct-1:  {div.get('distinct_1', 0):.3f}")
            print(f"  Distinct-2:  {div.get('distinct_2', 0):.3f}")
            if 'semantic_diversity' in div:
                print(f"  Semantic:    {div['semantic_diversity']:.3f}")
        
        # Quality
        if "quality" in metrics:
            qual = metrics["quality"]
            if "rouge" in qual:
                print("\nQuality Metrics (ROUGE):")
                print(f"  ROUGE-1: {qual['rouge'].get('rouge1', 0):.3f}")
                print(f"  ROUGE-2: {qual['rouge'].get('rouge2', 0):.3f}")
                print(f"  ROUGE-L: {qual['rouge'].get('rougeL', 0):.3f}")
        
        # Pass@k
        if "passk" in metrics:
            print("\nPass@k Metrics:")
            for k, score in metrics["passk"].items():
                print(f"  {k}: {score:.3f}")
        
        # Rewards
        if "rewards" in metrics:
            rew = metrics["rewards"]["overall"]
            print("\nReward Statistics:")
            print(f"  Mean:   {rew['mean']:.3f}")
            print(f"  Std:    {rew['std']:.3f}")
            print(f"  Median: {rew['median']:.3f}")
        
        print("\n" + "="*60)


def compare_models(
    environments: Dict[str, Any],
    test_data: List[Dict],
    n_generations: int = 10,
    max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare multiple models using the same test data.
    
    Args:
        environments: Dict mapping model_name -> environment
        test_data: Test examples
        n_generations: Number of candidates per example
        max_examples: Maximum examples to evaluate
        
    Returns:
        Dict with results for each model and comparisons
    """
    from .metrics import comprehensive_statistical_comparison
    
    logger.info(f"Comparing {len(environments)} models...")
    
    results = {"models": {}, "comparisons": {}}
    
    # Run benchmark for each model
    for name, env in environments.items():
        logger.info(f"\nEvaluating model: {name}")
        benchmark = EvaluationBenchmark(env, test_data)
        model_results = benchmark.run(n_generations, max_examples)
        results["models"][name] = model_results
    
    # Statistical comparisons
    logger.info("\nComputing statistical comparisons...")
    
    # Compare diversity
    if len(environments) == 2:
        names = list(environments.keys())
        model1_div = results["models"][names[0]]["metrics"].get("diversity", {})
        model2_div = results["models"][names[1]]["metrics"].get("diversity", {})
        
        # Compare Self-BLEU scores
        # (Would need individual scores for proper test, this is simplified)
        results["comparisons"]["diversity"] = {
            "model1": names[0],
            "model2": names[1],
            "model1_self_bleu": model1_div.get("self_bleu", 0),
            "model2_self_bleu": model2_div.get("self_bleu", 0)
        }
    
    return results

