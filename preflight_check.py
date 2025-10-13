#!/usr/bin/env python3
"""
Pre-flight Check for Evaluation

Run this BEFORE starting evaluation to catch all potential issues.

Usage:
    python preflight_check.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_files():
    """Check all required files exist."""
    logger.info("="*60)
    logger.info("1. Checking Required Files")
    logger.info("="*60)
    
    required_files = [
        # Base model
        ("config.json", "Base model config"),
        ("model-00001-of-00005.safetensors", "Base model weights"),
        # Adapters
        ("output/experiments/baseline/seed_42/adapter_model.safetensors", "Baseline adapter"),
        ("output/experiments/polychromic_0.3/seed_42/adapter_model.safetensors", "Polychromic adapter"),
        # Data
        ("data/processed/test_data.jsonl", "Test data"),
        # Scripts
        ("scripts/evaluation/evaluate_comprehensive.py", "Evaluation script"),
        ("scripts/analysis/analyze_lora_parameters.py", "LoRA analysis script"),
        ("src/evaluation/diversity_metrics.py", "Diversity metrics module"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if Path(filepath).exists():
            logger.info(f"  ✓ {description}")
        else:
            logger.error(f"  ✗ {description} MISSING: {filepath}")
            all_exist = False
    
    return all_exist


def check_python_packages():
    """Check all required Python packages."""
    logger.info("\n" + "="*60)
    logger.info("2. Checking Python Packages")
    logger.info("="*60)
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT (LoRA)'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('nltk', 'NLTK'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('anthropic', 'Anthropic API'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    all_installed = True
    for package, name in packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} MISSING")
            all_installed = False
    
    return all_installed


def check_nltk_data():
    """Check NLTK data is downloaded."""
    logger.info("\n" + "="*60)
    logger.info("3. Checking NLTK Data (CRITICAL for diversity metrics)")
    logger.info("="*60)
    
    try:
        import nltk
    except ImportError:
        logger.error("  ✗ NLTK not installed!")
        return False
    
    # Test punkt
    logger.info("\n  Testing punkt tokenizer...")
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("This is a test.")
        logger.info(f"    ✓ punkt works: {tokens}")
    except LookupError:
        logger.error("    ✗ punkt tokenizer MISSING!")
        logger.info("    Downloading punkt...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)  # New NLTK version
            logger.info("    ✓ punkt downloaded")
        except Exception as e:
            logger.error(f"    ✗ Download failed: {e}")
            return False
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
        return False
    
    # Test stopwords
    logger.info("\n  Testing stopwords...")
    try:
        from nltk.corpus import stopwords
        stops = stopwords.words('english')
        logger.info(f"    ✓ stopwords works ({len(stops)} words)")
    except LookupError:
        logger.info("    ✗ stopwords MISSING!")
        logger.info("    Downloading stopwords...")
        try:
            nltk.download('stopwords', quiet=True)
            logger.info("    ✓ stopwords downloaded")
        except Exception as e:
            logger.error(f"    ✗ Download failed: {e}")
            return False
    except Exception as e:
        logger.error(f"    ✗ Error: {e}")
    
    return True


def test_diversity_computation():
    """Actually test diversity metrics work."""
    logger.info("\n" + "="*60)
    logger.info("4. Testing Diversity Metrics Computation")
    logger.info("="*60)
    
    try:
        sys.path.insert(0, '.')
        from src.evaluation.diversity_metrics import compute_all_diversity_metrics
        
        # Test with simple examples
        test_texts = [
            "Hello world this is a test",
            "Hello there this is another test",
            "Goodbye world this is different",
            "Something completely else entirely"
        ]
        
        logger.info(f"\n  Computing metrics for {len(test_texts)} test texts...")
        metrics = compute_all_diversity_metrics(test_texts)
        
        logger.info(f"    Self-BLEU: {metrics['self_bleu']:.4f}")
        logger.info(f"    Distinct-1: {metrics['distinct_1']:.4f}")
        logger.info(f"    Distinct-2: {metrics['distinct_2']:.4f}")
        logger.info(f"    Unique words: {metrics['unique_words']}")
        logger.info(f"    Total words: {metrics['total_words']}")
        
        # Verify metrics are not all zero
        if (metrics['self_bleu'] == 0.0 and 
            metrics['distinct_1'] == 0.0 and 
            metrics['distinct_2'] == 0.0 and
            metrics['unique_words'] == 0):
            logger.error("\n  ✗ ALL METRICS ARE ZERO - NLTK NOT WORKING!")
            logger.error("  This will cause diversity_metrics.json to be all zeros")
            return False
        
        logger.info("\n  ✓ Diversity metrics working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"\n  ✗ Diversity metrics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cuda():
    """Check if CUDA is available."""
    logger.info("\n" + "="*60)
    logger.info("5. Checking CUDA/GPU")
    logger.info("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"  ✓ CUDA available")
            logger.info(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            logger.warning("  ⚠️  CUDA not available - will use CPU (SLOW)")
            logger.warning("  Evaluation will take 12-24 hours instead of 1-2 hours")
            return True  # Not fatal, just slow
    except Exception as e:
        logger.error(f"  ✗ Error checking CUDA: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECK FOR EVALUATION")
    print("="*60)
    print()
    
    checks = [
        ("Files", check_files),
        ("Python Packages", check_python_packages),
        ("NLTK Data", check_nltk_data),
        ("Diversity Metrics", test_diversity_computation),
        ("CUDA/GPU", check_cuda),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"\n✗ {name} check crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("PRE-FLIGHT SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("="*60)
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print()
        print("You're ready to run evaluation:")
        print()
        print("  tmux new -s qwen-eval")
        print("  python scripts/evaluation/evaluate_comprehensive.py \\")
        print("    --baseline-lora output/experiments/baseline/seed_42 \\")
        print("    --polychromic-lora output/experiments/polychromic_0.3/seed_42 \\")
        print("    --test-data data/processed/test_data.jsonl \\")
        print("    --output output/evaluation/ \\")
        print("    --n-generations 10 \\")
        print("    --max-examples 500 \\")
        print("    --skip-llm-judge")
        print()
        return 0
    else:
        print("="*60)
        print("⚠️  SOME CHECKS FAILED!")
        print("="*60)
        print()
        print("Fix the issues above before running evaluation.")
        print()
        print("Common fixes:")
        print("  - Missing packages: pip install -r requirements-runpod.txt")
        print("  - Missing NLTK data: python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())


