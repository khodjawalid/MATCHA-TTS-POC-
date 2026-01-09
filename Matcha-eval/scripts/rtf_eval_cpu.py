import csv
import time
import wave
from pathlib import Path
import subprocess


TEXTS_PATH = "../Data/eval_texts.txt"      # <- ajuste si ton fichier est ailleurs
OUT_DIR = Path("../results/rtf_wavs")
CSV_PATH = Path("../results/rtf_results.csv")


if OUT_DIR.exists():
    for wav in OUT_DIR.glob("*.wav"):
        wav.unlink()

if CSV_PATH.exists():
    CSV_PATH.unlink()



STEPS_LIST = [2, 4, 10]
TEMPERATURE = "0.667"
SPEAKING_RATE = "0.95"

PRINT_EVERY = 10   # <-- affichage de progression tous les 10 fichiers

OUT_DIR.mkdir(parents=True, exist_ok=True)

def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        sr = w.getframerate()
        return frames / float(sr)

def list_wavs_sorted_by_mtime(folder: Path):
    wavs = list(folder.glob("*.wav"))
    return sorted(wavs, key=lambda p: p.stat().st_mtime)

def run_and_time(cmd: list) -> float:
    t0 = time.perf_counter()
    # On garde stdout/stderr pour debug si ça casse
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    if res.returncode != 0:
        print("\n[ERROR] Command failed:", flush=True)
        print(" ".join(cmd), flush=True)
        print("\n[STDOUT]\n", res.stdout[:2000], flush=True)
        print("\n[STDERR]\n", res.stderr[:2000], flush=True)
        raise SystemExit(res.returncode)
    return t1 - t0

# Lire les textes
texts = []
with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        t = line.strip()
        if t:
            texts.append(t)

if not texts:
    raise SystemExit(f"Aucun texte trouvé dans {TEXTS_PATH}")

total_jobs = len(STEPS_LIST) * len(texts)
job_count = 0
t_global_start = time.perf_counter()

print(f"[START] {len(texts)} textes, steps={STEPS_LIST} => total jobs = {total_jobs}", flush=True)
print(f"[INFO] Sortie wav: {OUT_DIR} | CSV: {CSV_PATH}", flush=True)

# CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as fcsv:
    writer = csv.DictWriter(
        fcsv,
        fieldnames=[
            "id", "steps", "text_len_chars", "text_len_words",
            "gen_time_s", "audio_dur_s", "rtf", "wav_path"
        ],
    )
    writer.writeheader()

    for steps in STEPS_LIST:
        print(f"\n[STEPS] Démarrage steps={steps}", flush=True)

        for i, text in enumerate(texts):
            target_name = OUT_DIR / f"s{i:04d}_steps{steps}.wav"

            # Snapshot des wavs existants avant génération
            before = set(OUT_DIR.glob("*.wav"))

            cmd = [
                "matcha-tts",
                "--text", text,
                "--steps", str(steps),
                "--temperature", TEMPERATURE,
                "--speaking_rate", SPEAKING_RATE,
                "--cpu",
                "--output_folder", str(OUT_DIR),
            ]

            gen_time = run_and_time(cmd)
            job_count += 1

            # Identifier le nouveau wav créé
            after = set(OUT_DIR.glob("*.wav"))
            new_wavs = list(after - before)

            if not new_wavs:
                # fallback: prendre le wav le plus récent
                wavs_sorted = list_wavs_sorted_by_mtime(OUT_DIR)
                if not wavs_sorted:
                    raise SystemExit("Aucun wav trouvé dans le dossier de sortie après génération.")
                produced = wavs_sorted[-1]
            elif len(new_wavs) == 1:
                produced = new_wavs[0]
            else:
                # si plusieurs wavs -> prendre le plus récent parmi les nouveaux
                produced = sorted(new_wavs, key=lambda p: p.stat().st_mtime)[-1]

            # Renommer en nom stable
            if produced.resolve() != target_name.resolve():
                if target_name.exists():
                    target_name.unlink()
                produced.rename(target_name)

            dur = wav_duration_seconds(target_name)
            rtf = gen_time / dur if dur > 0 else float("nan")

            writer.writerow({
                "id": i,
                "steps": steps,
                "text_len_chars": len(text),
                "text_len_words": len(text.split()),
                "gen_time_s": gen_time,
                "audio_dur_s": dur,
                "rtf": rtf,
                "wav_path": str(target_name),
            })

            # Progress toutes les PRINT_EVERY jobs (ou à la fin)
            if (job_count % PRINT_EVERY == 0) or (job_count == total_jobs):
                t_now = time.perf_counter()
                elapsed = t_now - t_global_start
                avg_per_job = elapsed / job_count
                remaining = total_jobs - job_count
                eta = remaining * avg_per_job

                print(
                    f"[PROGRESS] steps={steps} | text={i+1}/{len(texts)} | "
                    f"global={job_count}/{total_jobs} | "
                    f"elapsed={elapsed/60:.1f} min | "
                    f"avg/job={avg_per_job:.2f} s | "
                    f"ETA={eta/60:.1f} min",
                    flush=True
                )

print(f"\n[OK] Résultats sauvegardés dans: {CSV_PATH}", flush=True)
