#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Example: Auto-tune an ONNX model with ISAT
# ──────────────────────────────────────────────────────────────

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model.onnx>}"

echo "=== ISAT: Inference Stack Auto-Tuner ==="
echo ""

# Step 1: Quick hardware check
echo "[1] Hardware info:"
isat hwinfo

# Step 2: Inspect the model
echo "[2] Model inspection:"
isat inspect "$MODEL_PATH"

# Step 3: Dry-run to see the search plan
echo "[3] Tuning plan (dry-run):"
isat tune "$MODEL_PATH" \
  --warmup 3 \
  --runs 5 \
  --cooldown 60 \
  --dry-run

echo ""
read -rp "Proceed with benchmarking? [y/N] " confirm
if [[ "$confirm" != [yY]* ]]; then
  echo "Aborted."
  exit 0
fi

# Step 4: Full tune
echo "[4] Running full auto-tune..."
isat tune "$MODEL_PATH" \
  --warmup 3 \
  --runs 5 \
  --cooldown 60 \
  --output-dir isat_output \
  --verbose

echo ""
echo "Done! Check isat_output/ for reports."
echo "  - isat_output/isat_report.html   (open in browser)"
echo "  - isat_output/isat_report.json   (machine-readable)"
echo "  - isat_output/best_config.sh     (source this for best env vars)"
