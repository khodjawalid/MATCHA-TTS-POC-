from pathlib import Path
import time

import pandas as pd
import jiwer
from jiwer import process_words
from jiwer.transformations import wer_default


# -----------------------
# Paths
# -----------------------
TEXTS_PATH = Path("../Data/eval_texts.txt")
TRANSCRIPTIONS = Path("../results/transcriptions_whisper_medium.csv")
OUT_SUMMARY = Path("../results/wer_summary.csv")

PRINT_EVERY = 50

print("[INIT] WER computation starting")
print(f"[INFO] jiwer version       : {getattr(jiwer, '__version__', 'unknown')}")
print(f"[INFO] Reference texts     : {TEXTS_PATH}")
print(f"[INFO] Transcriptions      : {TRANSCRIPTIONS}")
print(f"[INFO] Output summary      : {OUT_SUMMARY}\n")

# -----------------------
# Load reference texts
# -----------------------
refs = [
    l.strip()
    for l in TEXTS_PATH.read_text(encoding="utf-8").splitlines()
    if l.strip()
]
print(f"[INFO] Loaded {len(refs)} reference texts")

# -----------------------
# Load ASR transcriptions
# -----------------------
df = pd.read_csv(TRANSCRIPTIONS)
print(f"[INFO] Loaded {len(df)} ASR transcriptions\n")

# -----------------------
# Reference lookup
# -----------------------
def get_ref(i: int) -> str:
    if i < 0 or i >= len(refs):
        raise IndexError(f"id={i} hors limites (refs={len(refs)})")
    return refs[i]

df["ref_text"] = df["id"].apply(get_ref)

# -----------------------
# Compute WER
# -----------------------
wers = []
t_start = time.perf_counter()

print("[RUN] Computing WER per utterance...\n")

for idx, r in df.iterrows():
    ref = r["ref_text"]
    hyp = r["asr_text"] if isinstance(r["asr_text"], str) else ""

    # ✅ API récente: reference_transform / hypothesis_transform
    out = process_words(
        ref,
        hyp,
        reference_transform=wer_default,
        hypothesis_transform=wer_default,
    )
    wers.append(out.wer)

    if (idx + 1) % PRINT_EVERY == 0 or (idx + 1) == len(df):
        elapsed = time.perf_counter() - t_start
        avg = elapsed / (idx + 1)
        eta = avg * (len(df) - idx - 1)
        print(
            f"[PROGRESS] {idx+1:>4}/{len(df)} | avg={avg:.4f}s | ETA={eta:.1f}s",
            flush=True,
        )

df["wer"] = wers

# -----------------------
# Summary by steps
# -----------------------
summary = (
    df.groupby("steps")["wer"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("steps")
)

# -----------------------
# Save + final prints
# -----------------------
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_SUMMARY, index=False)

t_total = time.perf_counter() - t_start

print("\n[RESULT] WER summary by steps:")
print(summary)

print(
    f"\n[OK] WER computation finished\n"
    f"[TOTAL] Time = {t_total:.2f}s\n"
    f"[SAVED] {OUT_SUMMARY}"
)
