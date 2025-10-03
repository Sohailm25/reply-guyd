"""
Novel Evaluation Metrics

Introduces NEW metrics for multi-candidate generation evaluation:
1. User Selection Quality (USQ) - More realistic than standard Pass@k
2. Diversity Efficiency Ratio (DER) - Quantifies diversity benefit
3. Collapse Point Analysis - When diversity stops helping

These are NOVEL contributions to evaluation methodology.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def compute_user_selection_quality(
    generations: Dict[int, List[str]],
    ground_truths: List[str],
    selector_function: Callable[[List[str]], str],
    quality_metric: Callable[[str, str], float]
) -> float:
    """
    User Selection Quality (USQ): Quality of selected candidates using realistic selection.
    
    This is MORE REALISTIC than standard Pass@k because:
    - Standard Pass@k: "Best of k according to ground truth"
    - USQ: "Best of k according to realistic selection mechanism"
    
    In deployment, users don't have ground truth - they use heuristics!
    
    Args:
        generations: Dict mapping k -> list of k generations per prompt
        ground_truths: Ground truth responses for each prompt
        selector_function: Function that selects best from k candidates
                          (e.g., reward model, length heuristic, diversity)
        quality_metric: Function that measures quality(selected, ground_truth)
    
    Returns:
        Mean quality of selected responses
        
    Example:
        # Reward model selection
        def reward_selector(candidates):
            return max(candidates, key=reward_model.score)
        
        usq = compute_user_selection_quality(
            generations={10: all_10_gens},
            ground_truths=test_refs,
            selector_function=reward_selector,
            quality_metric=rouge_l_score
        )
    """
    k = list(generations.keys())[0]
    all_k_gens = generations[k]
    
    if len(all_k_gens) != len(ground_truths):
        raise ValueError(f"Mismatch: {len(all_k_gens)} generation sets vs {len(ground_truths)} ground truths")
    
    qualities = []
    for gen_set, ground_truth in zip(all_k_gens, ground_truths):
        # Select best according to selector function
        try:
            selected = selector_function(gen_set)
        except Exception as e:
            logger.warning(f"Selector failed: {e}, using first candidate")
            selected = gen_set[0] if gen_set else ""
        
        # Measure quality of selected
        try:
            quality = quality_metric(selected, ground_truth)
            qualities.append(quality)
        except Exception as e:
            logger.warning(f"Quality metric failed: {e}")
            qualities.append(0.0)
    
    mean_usq = float(np.mean(qualities))
    
    logger.info(f"✅ USQ@{k} = {mean_usq:.4f} (n={len(qualities)})")
    return mean_usq


def compute_usq_for_multiple_k(
    model,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    k_values: List[int],
    selector_function: Callable[[List[str]], str],
    quality_metric: Callable[[str, str], float],
    max_new_tokens: int = 100,
    temperature: float = 0.9
) -> Dict[int, float]:
    """
    Compute USQ for multiple k values.
    
    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompts: Input prompts
        ground_truths: Ground truth responses
        k_values: List of k values to evaluate
        selector_function: Selection function
        quality_metric: Quality measurement function
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Dict mapping k -> USQ@k
    """
    logger.info(f"Computing USQ for k={k_values}...")
    
    usq_results = {}
    
    for k in k_values:
        logger.info(f"\n  Computing USQ@{k}...")
        
        # Generate k candidates for each prompt
        all_k_gens = []
        model.eval()
        
        for i, prompt in enumerate(prompts):
            # Format prompt
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Generate k candidates
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=k,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generations = [
                tokenizer.decode(
                    output[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                for output in outputs
            ]
            
            all_k_gens.append(generations)
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Generated for {i+1}/{len(prompts)} prompts")
        
        # Compute USQ for this k
        usq = compute_user_selection_quality(
            generations={k: all_k_gens},
            ground_truths=ground_truths,
            selector_function=selector_function,
            quality_metric=quality_metric
        )
        
        usq_results[k] = usq
    
    logger.info(f"\n✅ USQ computation complete!")
    for k, usq in usq_results.items():
        logger.info(f"  USQ@{k} = {usq:.4f}")
    
    return usq_results


def compute_diversity_efficiency_ratio(
    passk_curve: Dict[int, float]
) -> Dict[int, float]:
    """
    Diversity Efficiency Ratio (DER): How much does diversity help?
    
    DER = Pass@k / (k × Pass@1)
    
    Interpretation:
    - DER = 1.0: Perfect diversity (each candidate adds equal value)
    - DER → 0: Mode collapse (all candidates identical)
    - DER > 0.7: Good diversity
    - DER < 0.3: Poor diversity
    
    Args:
        passk_curve: Dict mapping k -> Pass@k score
        
    Returns:
        Dict mapping k -> DER value
        
    Example:
        passk = {1: 0.40, 5: 0.60, 10: 0.70}
        der = compute_diversity_efficiency_ratio(passk)
        # der = {1: 1.0, 5: 0.30, 10: 0.175}
    """
    if 1 not in passk_curve:
        raise ValueError("Pass@1 must be in passk_curve")
    
    pass_1 = passk_curve[1]
    
    if pass_1 == 0:
        logger.warning("Pass@1 = 0, DER undefined (returning 0)")
        return {k: 0.0 for k in passk_curve.keys()}
    
    der_values = {}
    for k, pass_k in passk_curve.items():
        if k == 1:
            der = 1.0  # By definition
        else:
            der = pass_k / (k * pass_1)
        der_values[k] = float(der)
    
    logger.info("✅ Diversity Efficiency Ratio:")
    for k, der in der_values.items():
        status = "🟢 Good" if der > 0.3 else "🟡 Moderate" if der > 0.15 else "🔴 Poor"
        logger.info(f"  DER@{k} = {der:.3f} {status}")
    
    return der_values


def find_collapse_point(
    passk_curve: Dict[int, float],
    threshold: float = 0.01
) -> int:
    """
    Collapse Point: Value of k where Pass@k stops improving significantly.
    
    Finds k where d(Pass@k)/dk < threshold.
    
    Practical insight: "Generate this many candidates before diminishing returns"
    
    Args:
        passk_curve: Dict mapping k -> Pass@k score
        threshold: Derivative threshold for "not improving"
        
    Returns:
        Collapse point k
        
    Example:
        passk = {1: 0.40, 3: 0.50, 5: 0.60, 10: 0.62, 20: 0.63}
        collapse = find_collapse_point(passk)
        # collapse = 10 (after k=10, gains are minimal)
    """
    k_values = sorted(passk_curve.keys())
    pass_values = [passk_curve[k] for k in k_values]
    
    if len(k_values) < 2:
        logger.warning("Need at least 2 k values for collapse point")
        return k_values[0] if k_values else 1
    
    # Compute derivatives (differences)
    derivatives = []
    for i in range(1, len(pass_values)):
        dk = k_values[i] - k_values[i-1]
        dpass = pass_values[i] - pass_values[i-1]
        derivative = dpass / dk if dk > 0 else 0
        derivatives.append(derivative)
    
    # Find where derivative drops below threshold
    for i, deriv in enumerate(derivatives):
        if deriv < threshold:
            collapse_k = k_values[i+1]
            logger.info(f"✅ Collapse Point: k={collapse_k}")
            logger.info(f"   Pass@{collapse_k} = {passk_curve[collapse_k]:.3f}")
            logger.info(f"   Derivative = {deriv:.4f} < {threshold}")
            return collapse_k
    
    # If never collapses, return last k
    collapse_k = k_values[-1]
    logger.info(f"✅ No collapse detected up to k={collapse_k}")
    logger.info(f"   Model maintains diversity throughout")
    return collapse_k


def analyze_diversity_efficiency(
    passk_curves: Dict[str, Dict[int, float]],
    save_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive diversity efficiency analysis for multiple models.
    
    Computes DER and collapse points, creates visualizations.
    
    Args:
        passk_curves: Dict mapping model_name -> {k: Pass@k}
        save_dir: Directory to save plots
        
    Returns:
        Dict with DER values and collapse points for all models
    """
    logger.info("="*60)
    logger.info("Diversity Efficiency Analysis")
    logger.info("="*60)
    
    results = {}
    
    for model_name, passk in passk_curves.items():
        logger.info(f"\n📊 Analyzing: {model_name}")
        
        # Compute DER
        der = compute_diversity_efficiency_ratio(passk)
        
        # Find collapse point
        collapse = find_collapse_point(passk)
        
        results[model_name] = {
            'passk': passk,
            'der': der,
            'collapse_point': collapse
        }
    
    # Create visualizations if save_dir provided
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: DER comparison
        plot_der_comparison(results, str(save_path / 'der_comparison.pdf'))
        
        # Plot 2: Collapse points
        plot_collapse_points(results, str(save_path / 'collapse_points.pdf'))
        
        # Save numerical results
        with open(save_path / 'diversity_efficiency_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {save_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("Analysis Complete!")
    logger.info("="*60)
    
    return results


def plot_der_comparison(
    results: Dict,
    save_path: str
):
    """Plot DER comparison across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, data) in enumerate(results.items()):
        der = data['der']
        k_values = sorted(der.keys())
        der_values = [der[k] for k in k_values]
        
        ax.plot(k_values, der_values, marker='o', linewidth=2.5,
               markersize=8, label=model_name, color=colors[i % len(colors)],
               alpha=0.8)
    
    # Add reference lines
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    ax.axhline(0.15, color='gray', linestyle=':', alpha=0.5, label='Moderate threshold')
    
    ax.set_xlabel('k (Number of Candidates)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Diversity Efficiency Ratio (DER)', fontsize=13, fontweight='bold')
    ax.set_title('Diversity Efficiency Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    try:
        plt.tight_layout()
    except:
        pass
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved DER comparison to {save_path}")
    plt.close()


def plot_collapse_points(
    results: Dict,
    save_path: str
):
    """Plot collapse points comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    collapse_points = [results[m]['collapse_point'] for m in model_names]
    
    colors = ['#2ca02c' if cp > 5 else '#ff7f0e' for cp in collapse_points]
    
    bars = ax.barh(model_names, collapse_points, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, cp) in enumerate(zip(bars, collapse_points)):
        ax.text(cp + 0.5, i, f'k={cp}', va='center', fontweight='bold')
    
    ax.set_xlabel('Collapse Point (k)', fontsize=13, fontweight='bold')
    ax.set_title('Diversity Collapse Points', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    try:
        plt.tight_layout()
    except:
        pass
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved collapse points to {save_path}")
    plt.close()


# Selection functions for USQ

def length_based_selector(candidates: List[str]) -> str:
    """Select longest candidate (simple heuristic)."""
    return max(candidates, key=len) if candidates else ""


def diversity_based_selector(candidates: List[str]) -> str:
    """Select most diverse candidate (simple heuristic)."""
    # Select candidate with most unique words
    def unique_word_count(text):
        return len(set(text.lower().split()))
    
    return max(candidates, key=unique_word_count) if candidates else ""


def first_candidate_selector(candidates: List[str]) -> str:
    """Select first candidate (random baseline)."""
    return candidates[0] if candidates else ""


# Test
if __name__ == "__main__":
    print("✅ Novel metrics module loaded successfully")
    print("")
    print("📊 NOVEL METRICS:")
    print("  1. User Selection Quality (USQ)")
    print("     • More realistic than Pass@k")
    print("     • Considers realistic selection mechanisms")
    print("")
    print("  2. Diversity Efficiency Ratio (DER)")
    print("     • Quantifies diversity benefit")
    print("     • Single number: How much does diversity help?")
    print("")
    print("  3. Collapse Point Analysis")
    print("     • When does diversity stop helping?")
    print("     • Practical: How many candidates to generate?")
    print("")
    print("These are NOVEL contributions to evaluation methodology!")

