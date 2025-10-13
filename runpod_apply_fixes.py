#!/usr/bin/env python3
"""
Apply Bug Fixes to RunPod Files

Run this on RunPod to fix the JSON serialization bug.
This modifies src/evaluation/statistical_tests.py in place.

Usage (on RunPod):
    cd /workspace/Qwen3-8
    python runpod_apply_fixes.py
"""

import re

def apply_fixes():
    file_path = 'src/evaluation/statistical_tests.py'
    
    print("="*60)
    print("Applying Bug Fixes")
    print("="*60)
    print(f"\nFixing: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix 1: Add imports
        if 'import math' not in content:
            content = content.replace(
                'import logging\n',
                'import logging\nimport math\n'
            )
            print("  ✓ Added 'import math'")
        
        # Fix 2: Add sanitize_for_json function
        if 'def sanitize_for_json' not in content:
            sanitize_func = '''

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

'''
            content = content.replace(
                'logger = logging.getLogger(__name__)\n',
                'logger = logging.getLogger(__name__)\n' + sanitize_func
            )
            print("  ✓ Added sanitize_for_json() function")
        
        # Fix 3: Convert significant field to bool
        content = re.sub(
            r"'significant': p_value < 0\.05",
            "'significant': bool(p_value < 0.05)",
            content
        )
        print("  ✓ Fixed 'significant' field type")
        
        # Fix 4: Sanitize all values in comprehensive_statistical_comparison
        replacements = [
            ("'baseline_mean': float(np.mean(baseline_scores))", 
             "'baseline_mean': sanitize_for_json(np.mean(baseline_scores))"),
            ("'baseline_std': float(np.std(baseline_scores))",
             "'baseline_std': sanitize_for_json(np.std(baseline_scores))"),
            ("'polychromic_mean': float(np.mean(polychromic_scores))",
             "'polychromic_mean': sanitize_for_json(np.mean(polychromic_scores))"),
            ("'polychromic_std': float(np.std(polychromic_scores))",
             "'polychromic_std': sanitize_for_json(np.std(polychromic_scores))"),
            ("'improvement': float(np.mean(polychromic_scores) - np.mean(baseline_scores))",
             "'improvement': sanitize_for_json(np.mean(polychromic_scores) - np.mean(baseline_scores))"),
            ("results['cohens_d'] = cohens_d",
             "results['cohens_d'] = sanitize_for_json(cohens_d)"),
            ("results['cliffs_delta'] = cliffs_delta",
             "results['cliffs_delta'] = sanitize_for_json(cliffs_delta)"),
            ("if abs(cohens_d) >= 0.8:",
             "if cohens_d is None or math.isnan(cohens_d):\n        effect_interpretation = \"unable_to_compute\"\n    elif abs(cohens_d) >= 0.8:"),
        ]
        
        for old, new in replacements:
            if old in content and new not in content:
                content = content.replace(old, new)
        
        print("  ✓ Sanitized all numerical values")
        
        # Fix CI values
        ci_replacements = [
            ("'mean': baseline_ci[0]", "'mean': sanitize_for_json(baseline_ci[0])"),
            ("'lower': baseline_ci[1]", "'lower': sanitize_for_json(baseline_ci[1])"),
            ("'upper': baseline_ci[2]", "'upper': sanitize_for_json(baseline_ci[2])"),
            ("'mean': polychromic_ci[0]", "'mean': sanitize_for_json(polychromic_ci[0])"),
            ("'lower': polychromic_ci[1]", "'lower': sanitize_for_json(polychromic_ci[1])"),
            ("'upper': polychromic_ci[2]", "'upper': sanitize_for_json(polychromic_ci[2])"),
        ]
        
        for old, new in ci_replacements:
            if old in content:
                content = content.replace(old, new)
        
        print("  ✓ Sanitized confidence intervals")
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"\n✓ Successfully fixed {file_path}")
        print("\n" + "="*60)
        print("Fixes Applied!")
        print("="*60)
        print("\nYou can now re-run the evaluation:")
        print("  python scripts/evaluation/evaluate_comprehensive.py \\")
        print("    --baseline-lora output/experiments/baseline/seed_42 \\")
        print("    --polychromic-lora output/experiments/polychromic_0.3/seed_42 \\")
        print("    --test-data data/processed/test_data.jsonl \\")
        print("    --output output/evaluation/ \\")
        print("    --n-generations 10 \\")
        print("    --max-examples 500 \\")
        print("    --skip-llm-judge")
        print()
        
    except FileNotFoundError:
        print(f"❌ Error: {file_path} not found")
        print("Make sure you're in /workspace/Qwen3-8/")
        return False
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False
    
    return True


if __name__ == '__main__':
    import sys
    success = apply_fixes()
    sys.exit(0 if success else 1)


