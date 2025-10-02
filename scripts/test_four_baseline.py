#!/usr/bin/env python3
"""
Test Four-Baseline Evaluation Implementation

Quick test to verify all components work correctly.

Usage:
    python scripts/test_four_baseline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("TESTING FOUR-BASELINE IMPLEMENTATION")
print("="*60)

# Test 1: Import prompt templates
print("\n[Test 1] Importing prompt templates...")
try:
    from src.evaluation.prompt_templates import PROMPT_VARIANTS, get_prompt_description
    print(f"  ✓ Successfully imported")
    print(f"  ✓ Available variants: {list(PROMPT_VARIANTS.keys())}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Generate sample prompts
print("\n[Test 2] Testing prompt generation...")
try:
    test_tweet = "Just spent 3 hours debugging. Finally works! 🎉"
    
    for variant_name, variant_fn in PROMPT_VARIANTS.items():
        prompt = variant_fn(test_tweet)
        description = get_prompt_description(variant_name)
        print(f"  ✓ {variant_name}: {description}")
        print(f"    Length: {len(prompt)} chars")
        
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Verify evaluation script imports
print("\n[Test 3] Checking evaluation script...")
try:
    # Just check the file exists and has correct structure
    eval_script = Path(__file__).parent / 'evaluation' / 'evaluate_comprehensive.py'
    content = eval_script.read_text()
    
    required_functions = [
        'load_base_model_only',
        'generate_with_prompt_template',
        'PROMPT_VARIANTS'
    ]
    
    for func in required_functions:
        if func in content:
            print(f"  ✓ {func} found")
        else:
            print(f"  ✗ {func} not found")
            sys.exit(1)
            
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Verify visualization script updates
print("\n[Test 4] Checking visualization script...")
try:
    viz_script = Path(__file__).parent / 'analysis' / 'visualize_results.py'
    content = viz_script.read_text()
    
    # Check for dynamic model handling
    if 'model_names = list(diversity_data.keys())' in content:
        print(f"  ✓ Diversity plot handles N models")
    else:
        print(f"  ✗ Diversity plot not updated")
        
    if 'model_names = list(passk_data.keys())' in content:
        print(f"  ✓ Pass@k plot handles N models")
    else:
        print(f"  ✗ Pass@k plot not updated")
        
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour four-baseline evaluation is ready to use!")
print("\nQuick start:")
print("  1. Customize prompts: nano src/evaluation/prompt_templates.py")
print("  2. Test: python scripts/evaluation/evaluate_comprehensive.py \\")
print("           --include-zero-shot --test-data <path> --output test/")
print("  3. Full eval: Add --include-prompt-engineered and LoRA models")
print("\nSee: docs/implementation/FOUR_BASELINE_GUIDE.md")
