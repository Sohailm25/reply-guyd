#!/bin/bash
# Wrapper script to run commands with virtual environment activated

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "qwen-lora-env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run ./setup.sh first to create it."
    exit 1
fi

# Activate virtual environment
source qwen-lora-env/bin/activate

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <command>"
    echo ""
    echo "Examples:"
    echo "  ./run.sh python scripts/collect_data.py --method apify --target 1"
    echo "  ./run.sh python scripts/train_lora.py"
    echo "  ./run.sh python -c 'import torch; print(torch.__version__)'"
    echo ""
    echo "Or just activate the environment manually:"
    echo "  source qwen-lora-env/bin/activate"
    exit 0
fi

# Run the provided command
echo "🚀 Running in virtual environment: $*"
echo ""
exec "$@"

