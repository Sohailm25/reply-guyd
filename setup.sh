#!/bin/bash
# Setup script for Qwen3-8B LoRA Fine-tuning Project

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-8B LoRA Fine-tuning Project Setup"
echo "=========================================="

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
if [ -d "qwen-lora-env" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv qwen-lora-env
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source qwen-lora-env/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo -e "\n[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "\n[5/6] Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

echo "✓ Core dependencies installed"

# Optional: Install Unsloth for 2x training speedup
read -p "Install Unsloth for faster training? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Unsloth..."
    echo "⚠️  Note: Unsloth requires bleeding-edge dependencies and may fail"
    if pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null; then
        echo "✓ Unsloth installed successfully"
    else
        echo "⚠️  Unsloth installation failed (dependency conflict)"
        echo "   This is OK - Unsloth is optional for 2x speedup"
        echo "   Your training will work fine without it"
    fi
fi

# Optional: Install twscrape for web scraping
read -p "Install twscrape for web scraping? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing twscrape..."
    pip install twscrape
    echo "✓ twscrape installed"
fi

# Setup environment file
echo -e "\n[6/6] Setting up environment..."
if [ -f ".env" ]; then
    echo ".env file already exists"
else
    echo "Creating .env from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ .env file created from template"
        echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
    else
        echo "⚠️  .env.example not found"
    fi
fi

# Verify installation
echo -e "\n=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import sys
print(f'Python: {sys.version}')

# Check core packages
packages = [
    'torch',
    'transformers',
    'peft',
    'bitsandbytes',
    'accelerate',
    'datasets',
    'wandb',
    'sentence_transformers',
]

print('\nInstalled packages:')
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {pkg} ({version})')
    except ImportError:
        print(f'  ✗ {pkg} - NOT INSTALLED')

# Check GPU
try:
    import torch
    print(f'\nGPU available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
except:
    print('\nGPU check failed')
"

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Review config/data_collection_config.yaml"
echo "3. Run data collection:"
echo "   python scripts/collect_data.py --method apify --target 1000"
echo ""
echo "For help:"
echo "   python scripts/collect_data.py --help"
echo ""
echo "To activate this environment in future sessions:"
echo "   source qwen-lora-env/bin/activate"
echo ""


