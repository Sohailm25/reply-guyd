# Virtual Environment Guide

## ⚠️ Most Common Issue: "ModuleNotFoundError"

**Problem**: You ran a command but got:
```
ModuleNotFoundError: No module named 'apify_client'
```

**Cause**: The virtual environment isn't activated!

**Solution**: Activate it first.

---

## ✅ Solution: Always Activate First

### Method 1: Wrapper Script (Easiest)

Just prefix your commands with `./run.sh`:

```bash
# Instead of:
python scripts/collect_data.py --method apify --target 1

# Do this:
./run.sh python scripts/collect_data.py --method apify --target 1
```

The wrapper automatically activates the virtual environment for you!

### Method 2: Manual Activation (For Multiple Commands)

If you're running many commands, activate once:

```bash
# Activate the virtual environment
source qwen-lora-env/bin/activate

# Now run as many commands as you want
python scripts/collect_data.py --method apify --target 1
python scripts/train_lora.py
python -c "import torch; print(torch.__version__)"

# When done, deactivate
deactivate
```

---

## 🔍 How to Tell If You're in the Virtual Environment

### Check 1: Your Prompt

When activated, you'll see:
```bash
(qwen-lora-env) ➜ Qwen3-8 $
```

Notice the `(qwen-lora-env)` prefix!

### Check 2: Python Location

```bash
which python
```

**Correct** (in venv):
```
/Users/sohailmo/Repos/experiments/cuda/Qwen3-8/qwen-lora-env/bin/python
```

**Wrong** (not in venv):
```
/usr/bin/python
/opt/homebrew/bin/python3
```

If you see the wrong one, activate the venv!

---

## 🎯 Quick Reference

### Start a Session

```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
source qwen-lora-env/bin/activate
```

### Run Commands

```bash
# Now you can run anything:
python scripts/collect_data.py --method apify --target 1
```

### End Session

```bash
deactivate
```

---

## 🛠️ Common Commands with Virtual Environment

### Data Collection
```bash
source qwen-lora-env/bin/activate
python scripts/collect_data.py --method apify --target 1000
```

### Check Installation
```bash
source qwen-lora-env/bin/activate
python -c "
import transformers, peft, torch
print('✓ Everything installed')
"
```

### Use Wrapper (No activation needed)
```bash
./run.sh python scripts/collect_data.py --method apify --target 1
```

---

## 🐛 Troubleshooting

### "bash: ./run.sh: Permission denied"

```bash
chmod +x run.sh
./run.sh python scripts/collect_data.py --method apify --target 1
```

### "command not found: source"

You're probably in a different shell. Try:

```bash
. qwen-lora-env/bin/activate
```

Or just use the wrapper:
```bash
./run.sh python scripts/collect_data.py --method apify --target 1
```

### "Virtual environment not found"

Re-run setup:
```bash
./setup.sh
```

---

## 💡 Best Practice

**Always start your work session with:**

```bash
cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8
source qwen-lora-env/bin/activate
```

Then run all your commands.

**Or** just use `./run.sh` for every command!

---

## 📝 Add to Your Shell Profile (Optional)

To make activation easier, add an alias to your `~/.zshrc`:

```bash
echo "alias qwen='cd /Users/sohailmo/Repos/experiments/cuda/Qwen3-8 && source qwen-lora-env/bin/activate'" >> ~/.zshrc
source ~/.zshrc
```

Now you can just type:
```bash
qwen
# Automatically cd to project and activate venv!
```

---

## ✅ Summary

**The Fix**:
```bash
# Before running ANY Python command:
source qwen-lora-env/bin/activate

# Or use the wrapper:
./run.sh python <your-command>
```

That's it! 🚀

