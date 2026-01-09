#!/usr/bin/env bash
set -euo pipefail

# =========================
# Run CPU pipeline
# =========================

# Activate conda env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate matcha310

# Go to repo
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Paths
TEXTS="Data/eval_texts.txt"
OUTDIR="outputs_cpu"
RESULTS_DIR="results"

mkdir -p "$OUTDIR" "$RESULTS_DIR"

echo "[CPU] RTF eval..."
python scripts/rtf_eval_cpu.py \
  --texts "$TEXTS" \
  --output_dir "$OUTDIR" \
  --steps 2 4 10

echo "[CPU] RTF summary..."
python scripts/rtf_summary_cpu.py \
  --input_dir "$OUTDIR" \
  --out_csv "$RESULTS_DIR/rtf_cpu.csv"

echo "[CPU] WER eval..."
python scripts/wer_eval.py \
  --wav_dir "$OUTDIR" \
  --texts "$TEXTS" \
  --device cpu

echo "[CPU] WER summary..."
python scripts/wer_summary.py \
  --input_dir "$OUTDIR" \
  --out_csv "$RESULTS_DIR/wer_cpu.csv"

echo "✅ CPU done. Results:"
echo "  - $RESULTS_DIR/rtf_cpu.csv"
echo "  - $RESULTS_DIR/wer_cpu.csv"
