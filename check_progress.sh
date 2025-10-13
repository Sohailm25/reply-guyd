#!/bin/bash
# Quick script to check evaluation progress

echo "==================================="
echo "Evaluation Progress Monitor"
echo "==================================="
echo ""

# Check if tmux session exists
if tmux has-session -t qwen-eval 2>/dev/null; then
    echo "✓ Tmux session 'qwen-eval' is running"
else
    echo "✗ Tmux session 'qwen-eval' not found"
    exit 1
fi

echo ""
echo "Latest log output (last 30 lines):"
echo "-----------------------------------"
tail -30 evaluation_run.log

echo ""
echo "==================================="
echo "Commands:"
echo "  - View live log: tail -f evaluation_run.log"
echo "  - Attach to session: tmux attach -t qwen-eval"
echo "  - Detach from session: Ctrl+b, then d"
echo "  - Check outputs: ls -lh output/evaluation/"
echo "==================================="


