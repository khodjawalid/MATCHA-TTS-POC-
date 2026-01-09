#!/usr/bin/env bash
set -euo pipefail

# =========================
# Run GPU pipeline
# =========================

# Activate conda env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate matcha310

# Go to repo
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: GPU check (won't fail if missing, but prints info)
echo "[GPU] Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (GPU may not be available in this environment)."
fi

# Paths
TEXTS="Data/eval_texts.txt"
OUTDIR="outputs_gpu"
RESULTS_DIR="results"

mkdir -p "$OUTDIR" "$RESULTS_DIR"

echo "[GPU] RTF eval..."
python scripts/rtf_eval_gpu.py \
  --texts "$TEXTS" \
  --output_dir "$OUTDIR" \
  --steps 2 4 10

echo "[GPU] RTF summary..."
python scripts/rtf_summary_gpu.py \
  --input_dir "$OUTDIR" \
  --out_csv "$RESULTS_DIR/rtf_gpu.csv"

echo "[GPU] WER eval..."
python scripts/wer_eval.py \
  --wav_dir "$OUTDIR" \
  --texts "$TEXTS" \
  --device cuda

echo "[GPU] WER summary..."
python scripts/wer_summary.py \
  --input_dir "$OUTDIR" \
  --out_csv "$RESULTS_DIR/wer_gpu.csv"

echo "✅ GPU done. Results:"
echo "  - $RESULTS_DIR/rtf_gpu.csv"
echo "  - $RESULTS_DIR/wer_gpu.csv"
