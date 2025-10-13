#!/bin/bash
# RunPod Setup and Evaluation Launch Script

set -e

echo "========================================="
echo "RunPod Evaluation Setup"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "run_evaluation.sh" ]; then
    echo "Error: run_evaluation.sh not found. Are you in /workspace/Qwen3-8?"
    exit 1
fi

# Check for base model files
echo "1. Checking for base Qwen3-8B model..."
if [ ! -f "config.json" ] || [ ! -f "model-00001-of-00005.safetensors" ]; then
    echo ""
    echo "⚠️  Base model not found!"
    echo "You need the Qwen3-8B base model in this directory."
    echo ""
    echo "Options:"
    echo "  A) If you trained here before, the model should be in /workspace/Qwen3-8/"
    echo "  B) Download it:"
    echo "     huggingface-cli download Qwen/Qwen2.5-8B --local-dir ./ --local-dir-use-symlinks False"
    echo ""
    read -p "Do you want to download the model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading Qwen3-8B model (~16GB)..."
        huggingface-cli download Qwen/Qwen2.5-8B --local-dir ./ --local-dir-use-symlinks False
    else
        echo "Please download the model first, then run this script again."
        exit 1
    fi
else
    echo "✓ Base model found"
fi

echo ""
echo "2. Checking Python dependencies..."
if ! python -c "import transformers" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r requirements-runpod.txt
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "3. Verifying adapter files..."
if [ ! -f "output/experiments/baseline/seed_42/adapter_model.safetensors" ]; then
    echo "✗ Baseline adapter not found!"
    exit 1
fi
if [ ! -f "output/experiments/polychromic_0.3/seed_42/adapter_model.safetensors" ]; then
    echo "✗ Polychromic adapter not found!"
    exit 1
fi
echo "✓ Both adapters found"

echo ""
echo "4. Checking test data..."
if [ ! -f "data/processed/test_data.jsonl" ]; then
    echo "✗ Test data not found!"
    exit 1
fi
echo "✓ Test data found ($(wc -l < data/processed/test_data.jsonl) examples)"

echo ""
echo "5. Loading .env file..."
if [ ! -f ".env" ]; then
    echo "✗ .env file not found!"
    echo "Create .env with: ANTHROPIC_API_KEY=your_key_here"
    exit 1
fi
export $(grep -v '^#' .env | xargs)
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "✗ ANTHROPIC_API_KEY not set in .env"
    exit 1
fi
echo "✓ API key loaded"

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Starting evaluation in tmux session..."
echo ""

# Check if tmux session already exists
if tmux has-session -t qwen-eval 2>/dev/null; then
    echo "Tmux session 'qwen-eval' already exists."
    read -p "Kill it and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t qwen-eval
    else
        echo "Attaching to existing session..."
        tmux attach -t qwen-eval
        exit 0
    fi
fi

# Create new tmux session and start evaluation
tmux new-session -d -s qwen-eval -c /workspace/Qwen3-8
tmux send-keys -t qwen-eval './run_evaluation.sh 2>&1 | tee evaluation_run.log' C-m

echo ""
echo "========================================="
echo "🚀 Evaluation Started!"
echo "========================================="
echo ""
echo "Session: qwen-eval"
echo "Expected time: 1-2 hours on GPU"
echo "Expected cost: ~\$2-3 (compute) + \$80 (LLM-judge)"
echo ""
echo "Monitor progress:"
echo "  - View log: tail -f evaluation_run.log"
echo "  - Quick check: ./check_progress.sh"
echo "  - Attach to session: tmux attach -t qwen-eval"
echo "  - Detach: Ctrl+b, then d"
echo ""
echo "When complete:"
echo "  - Download results: See sync_from_runpod.sh on Mac"
echo ""


