import csv
import time
from pathlib import Path

import pandas as pd
import whisper


# -----------------------
# Paths
# -----------------------
RTF_CSV = Path("../results/rtf_results.csv")
OUT_CSV = Path("../results/transcriptions_whisper_medium.csv")

# -----------------------
# Load model (exact paper setting)
# -----------------------
print("[INIT] Loading Whisper model: medium (CPU)")
t0_model = time.perf_counter()
model = whisper.load_model("medium")
t1_model = time.perf_counter()
print(f"[INIT] Whisper loaded in {(t1_model - t0_model)/60:.1f} min\n")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(RTF_CSV)
n_total = len(df)

print(f"[INFO] {n_total} wav files to transcribe")
print(f"[INFO] Output CSV: {OUT_CSV}\n")

rows = []

t_global_start = time.perf_counter()
last_steps = None

# -----------------------
# Main loop
# -----------------------
for idx, r in df.iterrows():
    wav_path = Path(r["wav_path"])
    steps = int(r["steps"])

    if not wav_path.exists():
        raise FileNotFoundError(f"Wav introuvable: {wav_path}")

    # Print quand on change de steps
    if steps != last_steps:
        print(f"\n[STEPS] Début transcription — steps = {steps}")
        last_steps = steps

    t0 = time.perf_counter()

    print(
        f"[{idx+1:>4}/{n_total}] "
        f"steps={steps} | "
        f"{wav_path.name}",
        flush=True
    )

    result = model.transcribe(
        str(wav_path),
        language="en",
        fp16=False  # IMPORTANT sur CPU
    )

    t1 = time.perf_counter()
    elapsed = t1 - t0

    rows.append({
        "id": int(r["id"]),
        "steps": steps,
        "wav_path": str(wav_path),
        "asr_text": result["text"].strip()
    })

    # Progress global + ETA
    done = idx + 1
    avg_per_file = (time.perf_counter() - t_global_start) / done
    remaining = n_total - done
    eta = remaining * avg_per_file

    print(
        f"      ↳ done in {elapsed:.1f}s | "
        f"avg={avg_per_file:.1f}s | "
        f"ETA={eta/60:.1f} min",
        flush=True
    )

# -----------------------
# Save CSV
# -----------------------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "steps", "wav_path", "asr_text"]
    )
    writer.writeheader()
    writer.writerows(rows)

t_end = time.perf_counter()
print(
    f"\n[OK] Transcriptions sauvegardées dans {OUT_CSV}\n"
    f"[TOTAL] Temps total = {(t_end - t_global_start)/60:.1f} min"
)
