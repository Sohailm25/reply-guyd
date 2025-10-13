#!/bin/bash
# Complete Setup Script for RunPod Evaluation
# This fixes all known issues and verifies everything works

set -e

cd /workspace/Qwen3-8

echo "========================================="
echo "RunPod Evaluation - Complete Setup"
echo "========================================="
echo ""

# Fix 1: JSON serialization bug
echo "1. Fixing JSON serialization bug..."
python << 'ENDPYTHON'
import re

# Fix statistical_tests.py
content = open('src/evaluation/statistical_tests.py').read()

# Add math import
if 'import math' not in content:
    content = content.replace('import logging\n', 'import logging\nimport math\n')
    print("  ✓ Added math import")

# Fix significant field
content = re.sub(r"'significant': p_value < 0\.05", "'significant': bool(p_value < 0.05)", content)
print("  ✓ Fixed bool serialization")

open('src/evaluation/statistical_tests.py', 'w').write(content)
print("  ✓ statistical_tests.py patched")

# Fix diversity_metrics.py - add NLTK auto-download
content = open('src/evaluation/diversity_metrics.py').read()

if 'ensure_nltk_data' not in content:
    ensure_func = '''

def ensure_nltk_data():
    """Ensure NLTK data is downloaded."""
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        word_tokenize("test")
    except LookupError:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

ensure_nltk_data()
'''
    content = content.replace('logger = logging.getLogger(__name__)\n', 'logger = logging.getLogger(__name__)\n' + ensure_func)
    open('src/evaluation/diversity_metrics.py', 'w').write(content)
    print("  ✓ diversity_metrics.py patched with auto-download")
ENDPYTHON

echo ""
echo "2. Downloading NLTK data..."
python << 'ENDPYTHON'
import nltk
import ssl

# Handle SSL issues on some systems
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

print("  ✓ punkt downloaded")
print("  ✓ punkt_tab downloaded")
print("  ✓ stopwords downloaded")
ENDPYTHON

echo ""
echo "3. Testing NLTK tokenization..."
python << 'ENDPYTHON'
from nltk.tokenize import word_tokenize

test_text = "This is a test sentence for Twitter replies."
tokens = word_tokenize(test_text)

if tokens and len(tokens) > 0:
    print(f"  ✓ Tokenization works: {tokens[:5]}...")
else:
    print("  ✗ Tokenization failed!")
    exit(1)
ENDPYTHON

echo ""
echo "4. Testing diversity metrics computation..."
python << 'ENDPYTHON'
import sys
sys.path.insert(0, '/workspace/Qwen3-8')

from src.evaluation.diversity_metrics import compute_all_diversity_metrics

test_texts = [
    "Hello world this is a test",
    "Hello there this is another test",
    "Goodbye world this is different",
    "Something completely else"
]

metrics = compute_all_diversity_metrics(test_texts)

print(f"  Self-BLEU: {metrics['self_bleu']:.4f}")
print(f"  Distinct-1: {metrics['distinct_1']:.4f}")
print(f"  Distinct-2: {metrics['distinct_2']:.4f}")
print(f"  Unique words: {metrics['unique_words']}")

# Verify not all zero
if metrics['distinct_2'] > 0 and metrics['unique_words'] > 0:
    print("  ✓ Diversity metrics WORKING correctly!")
else:
    print("  ✗ Diversity metrics returning zeros - STILL BROKEN!")
    exit(1)
ENDPYTHON

echo ""
echo "5. Checking CUDA..."
python << 'ENDPYTHON'
import torch

if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠️  No CUDA - will use CPU (slow)")
ENDPYTHON

echo ""
echo "6. Verifying model files..."
if [ -f "output/experiments/baseline/seed_42/adapter_model.safetensors" ]; then
    echo "  ✓ Baseline adapter found"
else
    echo "  ✗ Baseline adapter MISSING!"
    exit 1
fi

if [ -f "output/experiments/polychromic_0.3/seed_42/adapter_model.safetensors" ]; then
    echo "  ✓ Polychromic adapter found"
else
    echo "  ✗ Polychromic adapter MISSING!"
    exit 1
fi

if [ -f "data/processed/test_data.jsonl" ]; then
    LINES=$(wc -l < data/processed/test_data.jsonl)
    echo "  ✓ Test data found ($LINES examples)"
else
    echo "  ✗ Test data MISSING!"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ SETUP COMPLETE - ALL TESTS PASSED!"
echo "========================================="
echo ""
echo "You can now run evaluation:"
echo ""
echo "  tmux new -s qwen-eval"
echo "  cd /workspace/Qwen3-8"
echo "  python scripts/evaluation/evaluate_comprehensive.py \\"
echo "    --baseline-lora output/experiments/baseline/seed_42 \\"
echo "    --polychromic-lora output/experiments/polychromic_0.3/seed_42 \\"
echo "    --test-data data/processed/test_data.jsonl \\"
echo "    --output output/evaluation/ \\"
echo "    --n-generations 10 \\"
echo "    --max-examples 500 \\"
echo "    --skip-llm-judge 2>&1 | tee eval.log"
echo ""
echo "  # Detach with: Ctrl+b, d"
echo ""


