# Matcha-TTS Evaluation (RTF & WER)

This repository contains the evaluation pipeline used to benchmark **Matcha-TTS**
in terms of **Real-Time Factor (RTF)** and **Word Error Rate (WER)** on **CPU and GPU**.

The goal is to provide a **fully reproducible setup**:
clone → install → run → obtain metrics.

---

## Repository structure

matcha-eval/
├─ scripts/
│ ├─ rtf_eval_cpu.py
│ ├─ rtf_summary_cpu.py
│ ├─ rtf_eval_gpu.py
│ ├─ rtf_summary_gpu.py
│ ├─ wer_eval.py
│ └─ wer_summary.py
├─ commands/
│ ├─ install_wsl.sh
│ ├─ run_cpu.sh
│ └─ run_gpu.sh
├─ Data/
│ └─ eval_texts.txt
├─ results/
│ └─ summary.csv
├─ .gitignore
└─ README.md


---

## Requirements

- Ubuntu (tested on **WSL2 – Ubuntu 24.04**)
- Conda / Miniconda
- Python **3.10**
- CPU or NVIDIA GPU (for GPU evaluation)

System dependencies are installed automatically by the install script.

---

## Installation

Run once:

```bash
bash commands/install_wsl.sh
