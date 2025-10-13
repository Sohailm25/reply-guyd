#!/bin/bash
# Transfer only the small helper scripts to RunPod

RUNPOD_HOST="kz142xqyz00wbl-644111eb@ssh.runpod.io"
RUNPOD_KEY="$HOME/.ssh/id_ed25519"

echo "Creating small scripts package..."
tar -czf /tmp/helper_scripts.tar.gz \
  run_evaluation.sh \
  check_progress.sh \
  runpod_setup.sh \
  .env

echo "Package size: $(du -h /tmp/helper_scripts.tar.gz | cut -f1)"
echo ""
echo "Transferring to RunPod..."

# Try piping through SSH
cat /tmp/helper_scripts.tar.gz | ssh -i "$RUNPOD_KEY" "$RUNPOD_HOST" \
  "cd /workspace/Qwen3-8 && tar -xzf - && chmod +x *.sh && echo '✓ Scripts transferred and made executable'"

rm /tmp/helper_scripts.tar.gz

echo ""
echo "✓ Done! Now on RunPod run: ./runpod_setup.sh"


