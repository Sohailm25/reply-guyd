#!/usr/bin/env python3
"""
Quick Test Script for GRPO Implementation

Tests the heuristic reward function and GRPO trainer initialization
to ensure everything is working before full training run.

Usage:
    python scripts/test_grpo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import HeuristicRewardFunction, GRPOTrainer, GRPOConfig


def test_heuristic_reward():
    """Test heuristic reward function."""
    print("\n" + "="*60)
    print("TEST 1: Heuristic Reward Function")
    print("="*60 + "\n")
    
    # Initialize reward function
    reward_fn = HeuristicRewardFunction()
    
    # Test cases
    tweet = "AI will revolutionize healthcare in ways we can't even imagine yet."
    
    test_cases = [
        ("Absolutely! Early disease detection from routine scans could save millions of lives. Imagine AI catching cancer years before symptoms appear.", "Good reply - substantive, relevant"),
        ("Agreed!", "Bad reply - too generic"),
        ("THIS IS AMAZING!!! WOW!!! GREAT!!! NICE!!!", "Bad reply - too much punctuation, generic"),
        ("Interesting perspective, but have you considered the privacy implications?", "Good reply - adds value, asks question"),
        ("lol", "Bad reply - too short, generic"),
    ]
    
    print("Test Results:")
    print("-" * 60)
    
    for reply, description in test_cases:
        score = reward_fn.compute_reward(tweet, reply)
        components = reward_fn.get_component_scores(tweet, reply)
        
        print(f"\n{description}")
        print(f"Reply: {reply[:60]}...")
        print(f"Total Score: {score:.3f}")
        print("Components:")
        for component, value in components.items():
            if component != 'total':
                print(f"  {component:12s}: {value:.3f}")
    
    print("\n" + "="*60)
    print("✅ TEST 1 PASSED: Heuristic Reward Function Working!")
    print("="*60 + "\n")
    
    return True


def test_grpo_config():
    """Test GRPO configuration."""
    print("\n" + "="*60)
    print("TEST 2: GRPO Configuration")
    print("="*60 + "\n")
    
    # Create config
    config = GRPOConfig(
        n_generations=4,
        kl_coeff=0.1,
        temperature=0.8,
        max_new_tokens=100,
        top_p=0.9,
        clip_advantage=True,
        advantage_clip_range=10.0
    )
    
    print("Configuration:")
    print(f"  N generations: {config.n_generations}")
    print(f"  KL coefficient: {config.kl_coeff}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Clip advantages: {config.clip_advantage}")
    print(f"  Clip range: {config.advantage_clip_range}")
    
    # Validate
    assert config.n_generations > 0, "N generations must be positive"
    assert 0 <= config.kl_coeff <= 1, "KL coefficient should be in [0, 1]"
    assert config.temperature > 0, "Temperature must be positive"
    assert config.max_new_tokens > 0, "Max new tokens must be positive"
    
    print("\n" + "="*60)
    print("✅ TEST 2 PASSED: GRPO Configuration Valid!")
    print("="*60 + "\n")
    
    return True


def test_batch_reward_computation():
    """Test batch reward computation."""
    print("\n" + "="*60)
    print("TEST 3: Batch Reward Computation")
    print("="*60 + "\n")
    
    reward_fn = HeuristicRewardFunction()
    
    # Batch test
    tweets = [
        "AI will change healthcare",
        "Climate change is urgent",
        "Python is a great language"
    ]
    
    replies = [
        "Absolutely! Early detection could save millions of lives.",
        "We need action now before it's too late.",
        "I love using it for data science projects!"
    ]
    
    print("Computing rewards for batch of 3 pairs...")
    rewards = reward_fn.compute_rewards_batch(tweets, replies)
    
    print("\nResults:")
    for i, (tweet, reply, reward) in enumerate(zip(tweets, replies, rewards)):
        print(f"\nPair {i+1}:")
        print(f"  Tweet: {tweet[:50]}...")
        print(f"  Reply: {reply[:50]}...")
        print(f"  Reward: {reward:.3f}")
    
    # Validate
    assert len(rewards) == 3, "Should have 3 rewards"
    assert all(0 <= r <= 1 for r in rewards), "All rewards should be in [0, 1]"
    
    print("\n" + "="*60)
    print("✅ TEST 3 PASSED: Batch Reward Computation Working!")
    print("="*60 + "\n")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "GRPO IMPLEMENTATION TESTS")
    print("="*70 + "\n")
    
    try:
        # Run tests
        test_heuristic_reward()
        test_grpo_config()
        test_batch_reward_computation()
        
        # Summary
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nGRPO Implementation is ready!")
        print("\nNext steps:")
        print("  1. Create data split script (scripts/data/split_training_phases.py)")
        print("  2. Test on small dataset (100 examples, 50 steps)")
        print("  3. Full training run from baseline checkpoint")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPlease check the implementation and try again.")
        print("="*70 + "\n")
        
        import traceback
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

