#!/usr/bin/env bash
set -euo pipefail

# =========================
# Matcha-TTS install (WSL)
# =========================

echo "[1/6] System deps..."
sudo apt update
sudo apt install -y git ffmpeg espeak-ng build-essential curl

echo "[2/6] Miniconda (if missing)..."
if [ ! -d "$HOME/miniconda3" ]; then
  cd "$HOME"
  curl -L -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3.sh -b -p "$HOME/miniconda3"
fi

# Enable conda for this shell
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo "[3/6] Create/activate conda env matcha310..."
if conda env list | awk '{print $1}' | grep -qx "matcha310"; then
  conda activate matcha310
else
  conda create -n matcha310 python=3.10 -y
  conda activate matcha310
fi

echo "[4/6] Clone Matcha-TTS (if missing)..."
if [ ! -d "$HOME/Matcha-TTS" ]; then
  cd "$HOME"
  git clone https://github.com/shivammehta25/Matcha-TTS.git
fi

echo "[5/6] Install Matcha-TTS..."
cd "$HOME/Matcha-TTS"
pip install -U pip
pip install -r requirements.txt
pip install -e .

echo "[6/6] Patch torch/lightning weights_only issue (PyTorch 2.6+)..."
# Force weights_only=False in load_from_checkpoint call inside matcha/cli.py
perl -0777 -i -pe 's/load_from_checkpoint\((\s*)checkpoint_path,(\s*)map_location=device(\s*)\)/load_from_checkpoint($1checkpoint_path,$2map_location=device,$3weights_only=False)/g' matcha/cli.py

echo "✅ Done."
echo "Test:"
echo "  conda activate matcha310"
echo "  matcha-tts --text \"This is a test.\" --steps 4 --temperature 0.667"
