#!/usr/bin/env python3
"""
Test Installation Script

Verifies that all dependencies are installed correctly
and the codebase is ready for training.

Usage:
    python scripts/test_installation.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('nltk', 'NLTK'),
        ('wandb', 'Weights & Biases'),
        ('anthropic', 'Anthropic API'),
        ('yaml', 'PyYAML'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {name}")
        except ImportError as e:
            logger.error(f"  ✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        logger.error(f"\nFailed to import: {', '.join(failed)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All imports successful!")
    return True


def test_cuda():
    """Test CUDA availability."""
    logger.info("\nTesting CUDA...")
    
    try:
        import torch
        
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.warning("  CUDA not available - will use CPU (very slow)")
            logger.warning("  For training, use RunPod with GPU")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        logger.error(f"  Error testing CUDA: {e}")
        return False


def test_modules():
    """Test that custom modules can be imported."""
    logger.info("\nTesting custom modules...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    modules = [
        ('src.training', 'Training modules'),
        ('src.evaluation', 'Evaluation modules'),
        ('src.training.base_trainer', 'Base Trainer'),
        ('src.training.polychromic_trainer', 'Polychromic Trainer'),
        ('src.training.data_module', 'Data Module'),
        ('src.evaluation.diversity_metrics', 'Diversity Metrics'),
        ('src.evaluation.statistical_tests', 'Statistical Tests'),
    ]
    
    failed = []
    for module, name in modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError as e:
            logger.error(f"  ✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        logger.error(f"\nFailed to import custom modules: {', '.join(failed)}")
        return False
    
    logger.info("All custom modules loaded successfully!")
    return True


def test_nltk_data():
    """Test NLTK data is downloaded."""
    logger.info("\nTesting NLTK data...")
    
    try:
        import nltk
        
        required = ['punkt', 'stopwords']
        for dataset in required:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
                logger.info(f"  ✓ {dataset}")
            except LookupError:
                logger.warning(f"  ✗ {dataset} not found")
                logger.info(f"    Downloading {dataset}...")
                nltk.download(dataset)
        
        return True
        
    except Exception as e:
        logger.error(f"  Error testing NLTK: {e}")
        return False


def test_file_structure():
    """Test that required directories exist."""
    logger.info("\nTesting file structure...")
    
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        'config/experiments',
        'src/training',
        'src/evaluation',
        'scripts/training',
        'scripts/evaluation',
        'output/experiments',
        'data/processed',
    ]
    
    failed = []
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            logger.info(f"  ✓ {dir_path}")
        else:
            logger.error(f"  ✗ {dir_path} - creating...")
            full_path.mkdir(parents=True, exist_ok=True)
            failed.append(dir_path)
    
    if failed:
        logger.warning(f"Created missing directories: {', '.join(failed)}")
    
    return True


def test_configs():
    """Test that config files exist."""
    logger.info("\nTesting configuration files...")
    
    base_dir = Path(__file__).parent.parent
    
    configs = [
        'config/experiments/baseline.yaml',
        'config/experiments/polychromic_0.3.yaml',
    ]
    
    for config in configs:
        config_path = base_dir / config
        if config_path.exists():
            logger.info(f"  ✓ {config}")
        else:
            logger.error(f"  ✗ {config} not found")
            return False
    
    return True


def main():
    logger.info("="*60)
    logger.info("INSTALLATION TEST")
    logger.info("="*60)
    
    tests = [
        ("Dependencies", test_imports),
        ("CUDA", test_cuda),
        ("Custom Modules", test_modules),
        ("NLTK Data", test_nltk_data),
        ("File Structure", test_file_structure),
        ("Configuration Files", test_configs),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("="*60)
        logger.info("\nYou're ready to train!")
        logger.info("\nNext steps:")
        logger.info("  1. Ensure data is in data/processed/")
        logger.info("  2. Configure .env with API keys")
        logger.info("  3. Run training on RunPod:")
        logger.info("     python scripts/training/train_model.py \\")
        logger.info("       --config config/experiments/baseline.yaml \\")
        logger.info("       --data data/processed/training_data_*.jsonl")
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("✗ SOME TESTS FAILED")
        logger.error("="*60)
        logger.error("\nPlease fix the issues above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

