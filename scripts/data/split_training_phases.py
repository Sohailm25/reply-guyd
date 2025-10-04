#!/usr/bin/env python3
"""
Split curated data for four experimental conditions.

Creates stratified splits for:
1. Test set (500 examples, held out)
2. Full SFT data (5,000 examples for models 1 & 2)
3. Two-phase data (2,500 SFT + 2,500 GRPO for models 3 & 4)

Stratification: By engagement quartiles to maintain distribution
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple


def load_data(input_file: Path) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def stratify_by_engagement(data: List[Dict], n_strata: int = 4) -> Dict[int, List[Dict]]:
    """
    Stratify data by engagement (reply likes) into quartiles.
    
    Returns:
        Dict mapping stratum_id -> list of examples
    """
    # Extract engagement values
    engagement_values = [d['reply_likes'] for d in data]
    
    # Compute quartile boundaries
    quartiles = np.percentile(engagement_values, [25, 50, 75])
    
    # Assign each example to a stratum
    strata = defaultdict(list)
    for example in data:
        likes = example['reply_likes']
        if likes <= quartiles[0]:
            stratum = 0
        elif likes <= quartiles[1]:
            stratum = 1
        elif likes <= quartiles[2]:
            stratum = 2
        else:
            stratum = 3
        strata[stratum].append(example)
    
    return strata


def stratified_split(
    strata: Dict[int, List[Dict]],
    test_size: int,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform stratified split into test and remaining data.
    
    Args:
        strata: Dict of stratum_id -> examples
        test_size: Total test examples
        seed: Random seed
    
    Returns:
        (test_data, remaining_data)
    """
    random.seed(seed)
    
    # Calculate test examples per stratum (proportional)
    total_examples = sum(len(examples) for examples in strata.values())
    test_per_stratum = {
        s: int(test_size * len(examples) / total_examples)
        for s, examples in strata.items()
    }
    
    # Adjust to ensure exactly test_size examples
    adjustment = test_size - sum(test_per_stratum.values())
    if adjustment > 0:
        # Add to largest stratum
        largest_stratum = max(strata.keys(), key=lambda s: len(strata[s]))
        test_per_stratum[largest_stratum] += adjustment
    
    test_data = []
    remaining_data = []
    
    for stratum_id, examples in strata.items():
        # Shuffle within stratum
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Split
        n_test = test_per_stratum[stratum_id]
        test_data.extend(shuffled[:n_test])
        remaining_data.extend(shuffled[n_test:])
    
    # Final shuffle
    random.shuffle(test_data)
    random.shuffle(remaining_data)
    
    return test_data, remaining_data


def split_two_phase(
    data: List[Dict],
    phase1_size: int,
    phase2_size: int,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into Phase 1 (SFT) and Phase 2 (GRPO).
    
    Uses stratified sampling to maintain engagement distribution.
    """
    # Re-stratify remaining data
    strata = stratify_by_engagement(data, n_strata=4)
    
    random.seed(seed)
    
    # Calculate phase1 examples per stratum
    total_examples = len(data)
    phase1_per_stratum = {
        s: int(phase1_size * len(examples) / total_examples)
        for s, examples in strata.items()
    }
    
    # Adjust to ensure exactly phase1_size examples
    adjustment = phase1_size - sum(phase1_per_stratum.values())
    if adjustment > 0:
        largest_stratum = max(strata.keys(), key=lambda s: len(strata[s]))
        phase1_per_stratum[largest_stratum] += adjustment
    
    phase1_data = []
    phase2_data = []
    
    for stratum_id, examples in strata.items():
        # Shuffle within stratum
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        # Split
        n_phase1 = phase1_per_stratum[stratum_id]
        phase1_data.extend(shuffled[:n_phase1])
        phase2_data.extend(shuffled[n_phase1:])
    
    # Final shuffle
    random.shuffle(phase1_data)
    random.shuffle(phase2_data)
    
    return phase1_data, phase2_data


def save_jsonl(data: List[Dict], output_file: Path):
    """Save data to JSONL."""
    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    print(f"✅ Saved {len(data)} examples to {output_file}")


def print_statistics(name: str, data: List[Dict]):
    """Print dataset statistics."""
    engagement = [d['reply_likes'] for d in data]
    print(f"\n{name}:")
    print(f"  Examples: {len(data)}")
    print(f"  Engagement - Mean: {np.mean(engagement):.1f}, "
          f"Median: {np.median(engagement):.1f}, "
          f"Min: {min(engagement)}, Max: {max(engagement)}")


def main():
    parser = argparse.ArgumentParser(description="Split data for four experimental conditions")
    parser.add_argument('--input', type=Path, required=True,
                        help='Input JSONL file (curated data)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'),
                        help='Output directory')
    parser.add_argument('--test-size', type=int, default=500,
                        help='Number of test examples')
    parser.add_argument('--phase1-size', type=int, default=2500,
                        help='Phase 1 (SFT) size for two-phase models')
    parser.add_argument('--phase2-size', type=int, default=2500,
                        help='Phase 2 (GRPO) size for two-phase models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DATA SPLITTING FOR FOUR EXPERIMENTAL CONDITIONS")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"✅ Loaded {len(data)} examples")
    
    # Step 1: Split into test and remaining
    print(f"\n📊 Step 1: Creating test set ({args.test_size} examples)...")
    strata = stratify_by_engagement(data, n_strata=4)
    test_data, remaining_data = stratified_split(strata, args.test_size, args.seed)
    
    print_statistics("Test Set", test_data)
    print_statistics("Remaining Data", remaining_data)
    
    # Save test set
    test_file = args.output_dir / 'test_data.jsonl'
    save_jsonl(test_data, test_file)
    
    # Step 2: Create full SFT data (for models 1 & 2)
    print(f"\n📊 Step 2: Creating full SFT dataset (models 1 & 2)...")
    # Use first 5000 examples from remaining
    full_sft_data = remaining_data[:5000]
    print_statistics("Full SFT Data", full_sft_data)
    
    full_sft_file = args.output_dir / 'train_full_sft.jsonl'
    save_jsonl(full_sft_data, full_sft_file)
    
    # Step 3: Create two-phase data (for models 3 & 4)
    print(f"\n📊 Step 3: Creating two-phase datasets (models 3 & 4)...")
    print(f"  Phase 1 (SFT): {args.phase1_size} examples")
    print(f"  Phase 2 (GRPO): {args.phase2_size} examples")
    
    # Use the same 5000 examples, but split differently
    phase1_data, phase2_data = split_two_phase(
        full_sft_data,
        args.phase1_size,
        args.phase2_size,
        args.seed
    )
    
    print_statistics("Phase 1 (SFT warm-start)", phase1_data)
    print_statistics("Phase 2 (GRPO refinement)", phase2_data)
    
    phase1_file = args.output_dir / 'train_phase1_sft.jsonl'
    phase2_file = args.output_dir / 'train_phase2_grpo.jsonl'
    
    save_jsonl(phase1_data, phase1_file)
    save_jsonl(phase2_data, phase2_file)
    
    # Summary
    print("\n" + "="*70)
    print("✅ DATA SPLITTING COMPLETE!")
    print("="*70)
    print(f"\nCreated files:")
    print(f"  1. {test_file} ({len(test_data)} examples)")
    print(f"  2. {full_sft_file} ({len(full_sft_data)} examples)")
    print(f"  3. {phase1_file} ({len(phase1_data)} examples)")
    print(f"  4. {phase2_file} ({len(phase2_data)} examples)")
    
    print(f"\nUsage by model:")
    print(f"  Model 1 (Baseline SFT):      {full_sft_file}")
    print(f"  Model 2 (Polychromic SFT):   {full_sft_file}")
    print(f"  Model 3 (Baseline→GRPO):     {phase1_file} → {phase2_file}")
    print(f"  Model 4 (Polychromic→GRPO):  {phase1_file} → {phase2_file}")
    print(f"  All models evaluate on:      {test_file}")
    
    print("\n" + "="*70)
    print("🚀 Ready for training!")
    print("="*70)


if __name__ == '__main__':
    main()
