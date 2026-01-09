import csv
import math
from collections import defaultdict
from pathlib import Path

# === Chemin vers les résultats RTF GPU ===
CSV_PATH = Path("matcha_rtf_mel_only/rtf_mel_only.csv")

if not CSV_PATH.exists():
    raise SystemExit(f"[FATAL] Fichier introuvable: {CSV_PATH}")

vals = defaultdict(list)

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps = int(row["steps"])
        rtf = float(row["rtf"])
        if math.isfinite(rtf):
            vals[steps].append(rtf)

def mean_std(x):
    m = sum(x) / len(x)
    v = sum((xi - m) ** 2 for xi in x) / (len(x) - 1) if len(x) > 1 else 0.0
    return m, v ** 0.5

print("\nRTF summary (mean ± std):\n")
for steps in sorted(vals):
    m, s = mean_std(vals[steps])
    print(f"steps={steps:>2} : {m:.4f} ± {s:.4f}   (n={len(vals[steps])})")