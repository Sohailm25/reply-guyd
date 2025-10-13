#!/usr/bin/env python3
"""
Test NLTK Setup on RunPod

This script tests if NLTK is properly configured and downloads
missing data if needed.

Run this BEFORE evaluation to ensure diversity metrics will work!

Usage (on RunPod):
    cd /workspace/Qwen3-8
    python test_runpod_nltk.py
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_nltk():
    """Test and fix NLTK setup."""
    print("="*60)
    print("NLTK Configuration Test")
    print("="*60)
    print()
    
    # Test 1: Can we import nltk?
    print("1. Testing NLTK import...")
    try:
        import nltk
        print("  ✓ NLTK imported successfully")
    except ImportError:
        print("  ✗ NLTK not installed!")
        print("  Fix: pip install nltk")
        return False
    
    # Test 2: Check punkt tokenizer
    print("\n2. Testing punkt tokenizer...")
    try:
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence."
        tokens = word_tokenize(test_text)
        print(f"  ✓ Tokenization works: {tokens}")
    except LookupError as e:
        print(f"  ✗ punkt tokenizer missing!")
        print(f"  Error: {e}")
        print("  Downloading punkt...")
        nltk.download('punkt')
        print("  ✓ punkt downloaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 3: Check stopwords
    print("\n3. Testing stopwords...")
    try:
        from nltk.corpus import stopwords
        stops = stopwords.words('english')
        print(f"  ✓ Stopwords available ({len(stops)} words)")
    except LookupError:
        print("  ✗ Stopwords missing!")
        print("  Downloading stopwords...")
        nltk.download('stopwords')
        print("  ✓ Stopwords downloaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Test Self-BLEU computation
    print("\n4. Testing Self-BLEU computation...")
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        texts = [
            "This is a test sentence.",
            "This is another test sentence.",
            "Something completely different."
        ]
        
        # Try to compute Self-BLEU
        smoothing = SmoothingFunction().method1
        scores = []
        
        for i, text in enumerate(texts):
            references = [
                word_tokenize(t.lower())
                for j, t in enumerate(texts) if j != i
            ]
            candidate = word_tokenize(text.lower())
            
            if references and candidate:
                score = sentence_bleu(
                    references,
                    candidate,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothing
                )
                scores.append(score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  ✓ Self-BLEU computed: {avg_score:.4f}")
        else:
            print("  ✗ Self-BLEU failed to compute!")
            return False
            
    except Exception as e:
        print(f"  ✗ Self-BLEU failed: {e}")
        return False
    
    # Test 5: Test diversity metrics module
    print("\n5. Testing diversity metrics module...")
    try:
        import sys
        sys.path.insert(0, '.')
        from src.evaluation.diversity_metrics import (
            compute_self_bleu,
            compute_distinct_n,
            compute_all_diversity_metrics
        )
        
        test_texts = [
            "Hello world",
            "Hello there",
            "Goodbye world",
            "Something else entirely different"
        ]
        
        metrics = compute_all_diversity_metrics(test_texts)
        
        print(f"  Self-BLEU: {metrics['self_bleu']:.4f}")
        print(f"  Distinct-1: {metrics['distinct_1']:.4f}")
        print(f"  Distinct-2: {metrics['distinct_2']:.4f}")
        print(f"  Unique words: {metrics['unique_words']}")
        
        # Check if all are zero (would indicate failure)
        if metrics['self_bleu'] == 0.0 and metrics['distinct_1'] == 0.0:
            print("  ✗ All metrics are 0.0 - something is wrong!")
            return False
        else:
            print("  ✓ Diversity metrics working correctly!")
            
    except Exception as e:
        print(f"  ✗ Diversity metrics module failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("="*60)
    print("✓ ALL NLTK TESTS PASSED!")
    print("="*60)
    print()
    print("You're ready to run evaluation!")
    print()
    
    return True


if __name__ == '__main__':
    import sys
    success = test_nltk()
    
    if not success:
        print()
        print("="*60)
        print("⚠️  NLTK SETUP INCOMPLETE")
        print("="*60)
        print()
        print("Fix by running:")
        print("  python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        print()
        sys.exit(1)
    
    sys.exit(0)


