#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matcha-TTS RTF (paper-like) — MEL ONLY (no vocoder)
- Single Python process
- Loads Matcha model once from local ckpt
- Measures ONLY mel inference time (GPU-synchronized)
- Computes audio duration from mel frames: dur = n_frames * hop_length / sampling_rate
- Writes CSV with per-utterance RTF + summary prints

Prereqs (your case):
- conda env matcha310 activated
- espeak-ng installed in ~/.local/bin and data in ~/.local/share/espeak-ng-data
- ckpt exists: ~/.local/share/matcha_tts/matcha_ljspeech.ckpt
"""

import os
import time
import csv
import math
import inspect
from pathlib import Path

# ============================================================
# 0) FIX ENV for phonemizer/espeak (MUST be set before imports)
# ============================================================
HOME = Path.home()
LOCAL_BIN = HOME / ".local" / "bin"
LOCAL_LIB = HOME / ".local" / "lib"
ESPEAK = LOCAL_BIN / "espeak"          # symlink -> espeak-ng
ESPEAK_NG = LOCAL_BIN / "espeak-ng"
ESPEAK_DATA = HOME / ".local" / "share" / "espeak-ng-data"

os.environ["PATH"] = f"{LOCAL_BIN}:{os.environ.get('PATH','')}"
os.environ["ESPEAK_DATA_PATH"] = str(ESPEAK_DATA)
os.environ["PHONEMIZER_ESPEAK_PATH"] = str(ESPEAK)
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(LOCAL_LIB / "libespeak-ng.so")
os.environ["LD_LIBRARY_PATH"] = f"{LOCAL_LIB}:{os.environ.get('LD_LIBRARY_PATH','')}"
os.environ["PYTHONNOUSERSITE"] = "1"

print("[ENV] espeak configured for phonemizer", flush=True)
print(f"[ENV] ESPEAK_DATA_PATH = {os.environ['ESPEAK_DATA_PATH']}", flush=True)

# ============================================================
# 1) Paths / Params
# ============================================================
TEXTS_PATH = Path("~/shared/Matcha_gr10/walid/eval_texts.txt").expanduser()

# fixed model paths (as you gave)
CKPT_PATH = Path("~/.local/share/matcha_tts/matcha_ljspeech.ckpt").expanduser()

OUT_DIR = Path("~/matcha_rtf_mel_only").expanduser()
OUT_CSV = OUT_DIR / "rtf_mel_only.csv"

STEPS_LIST = [2, 4, 10]
TEMPERATURE = 0.667
SPEAKING_RATE = 0.95
WARMUP_N = 3
PRINT_EVERY = 10

# audio params (LJSpeech defaults)
SAMPLING_RATE = 22050
HOP_LENGTH = 256

# ============================================================
# 2) Quick checks
# ============================================================
print("[INIT] Matcha RTF mel-only starting", flush=True)
print(f"[INFO] Texts    = {TEXTS_PATH}", flush=True)
print(f"[INFO] CKPT     = {CKPT_PATH}", flush=True)
print(f"[INFO] Out CSV  = {OUT_CSV}", flush=True)
print(f"[INFO] Steps    = {STEPS_LIST}", flush=True)

if not TEXTS_PATH.exists():
    raise SystemExit(f"[FATAL] Missing texts file: {TEXTS_PATH}")

if not CKPT_PATH.exists():
    raise SystemExit(f"[FATAL] Missing ckpt: {CKPT_PATH}")

if not ESPEAK.exists():
    raise SystemExit(f"[FATAL] Missing {ESPEAK}. Create symlink to espeak-ng.")
if not (ESPEAK_DATA / "phontab").exists():
    raise SystemExit(f"[FATAL] Missing phontab in {ESPEAK_DATA}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3) Imports after ENV is set
# ============================================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device   = {DEVICE}", flush=True)
print(f"[GPU] torch={torch.__version__}", flush=True)
print(f"[GPU] cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"[GPU] device_name={torch.cuda.get_device_name(0)}", flush=True)

# Load text utils AFTER env is set (phonemizer)
from matcha.text import text_to_sequence

# Model class
from matcha.models.matcha_tts import MatchaTTS

# ============================================================
# 4) Load texts
# ============================================================
texts = [l.strip() for l in TEXTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"[INFO] Loaded {len(texts)} texts", flush=True)
if not texts:
    raise SystemExit("[FATAL] No texts loaded")

# ============================================================
# 5) Load checkpoint + instantiate model properly
# ============================================================
print("[LOAD] Loading checkpoint...", flush=True)
ckpt = torch.load(str(CKPT_PATH), map_location="cpu")

# Try to locate model weights in common lightning formats
state_dict = None
for key in ["state_dict", "model", "model_state_dict"]:
    if key in ckpt and isinstance(ckpt[key], dict) and len(ckpt[key]) > 0:
        state_dict = ckpt[key]
        print(f"[LOAD] Found weights under ckpt['{key}'] (len={len(state_dict)})", flush=True)
        break
if state_dict is None and isinstance(ckpt, dict):
    # sometimes ckpt itself is the state_dict
    if all(isinstance(k, str) for k in ckpt.keys()) and any("." in k for k in ckpt.keys()):
        state_dict = ckpt
        print("[LOAD] Using checkpoint dict itself as state_dict", flush=True)

if state_dict is None:
    raise RuntimeError("[FATAL] Could not find model weights in checkpoint (state_dict not found).")

# Extract hyperparameters saved by Lightning
hparams = ckpt.get("hyper_parameters", None)
if hparams is None or not isinstance(hparams, dict):
    raise RuntimeError("[FATAL] No 'hyper_parameters' in ckpt. Can't rebuild MatchaTTS config.")

# Build model from hparams (MatchaTTS expects these kwargs)
print("[LOAD] Instantiating MatchaTTS from ckpt hyper_parameters...", flush=True)
model = MatchaTTS(**hparams).to(DEVICE)
model.eval()

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[LOAD] load_state_dict ok | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing:
    print("[WARN] missing keys (first 10):", missing[:10], flush=True)
if unexpected:
    print("[WARN] unexpected keys (first 10):", unexpected[:10], flush=True)

# ============================================================
# 6) Text -> token batch
# ============================================================
def _extract_int_sequence(obj):
    """
    Extract a list[int] from various possible text_to_sequence return formats.
    Supported:
      - list[int]
      - tuple/list containing a list[int]
      - dict containing a list[int] under common keys
      - string of space-separated ints ("12 53 9 ...")
    """
    # already list/tuple of ints
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, int) for x in obj):
        return list(obj)

    # dict case
    if isinstance(obj, dict):
        for k in ["sequence", "seq", "ids", "token_ids", "tokens"]:
            if k in obj:
                seq = _extract_int_sequence(obj[k])
                if seq is not None:
                    return seq
        return None

    # tuple/list container case (e.g., (seq, cleaned) or (cleaned, seq))
    if isinstance(obj, (list, tuple)):
        # try each element
        for it in obj:
            seq = _extract_int_sequence(it)
            if seq is not None:
                return seq
        return None

    # string case: maybe "1 23 456" (space-separated ints)
    if isinstance(obj, str):
        parts = obj.strip().split()
        if parts and all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
        return None

    return None


def text_to_ids(text: str, debug: bool = False):
    candidates = [
        ["english_cleaners2"],
        ["english_cleaners"],
        ["basic_cleaners"],
    ]
    last_err = None

    for cleaner_names in candidates:
        try:
            res = text_to_sequence(text, cleaner_names)

            seq = _extract_int_sequence(res)
            if seq is None:
                if debug:
                    print("[DEBUG] text_to_sequence return type:", type(res), flush=True)
                    print("[DEBUG] text_to_sequence return repr (first 300 chars):",
                          repr(res)[:300], flush=True)
                raise TypeError(
                    f"text_to_sequence returned {type(res)} but no int sequence could be extracted."
                )

            if debug:
                print(f"[DEBUG] cleaners={cleaner_names} | seq_len={len(seq)} | first10={seq[:10]}", flush=True)

            return seq, cleaner_names

        except Exception as e:
            last_err = e

    raise RuntimeError(f"text_to_sequence failed for all cleaners. Last error: {last_err}")


def to_batch_tokens(text: str, debug: bool = False):
    seq, used_cleaners = text_to_ids(text, debug=debug)

    # safety: ensure ints
    if not (isinstance(seq, list) and all(isinstance(x, int) for x in seq)):
        raise TypeError(f"Expected list[int] after extraction, got: {type(seq)}")

    x = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)    # [1, T]
    x_lengths = torch.LongTensor([x.size(1)]).to(DEVICE) # [1]
    return x, x_lengths, used_cleaners


# ============================================================
# 7) Run mel inference (robust call)
# ============================================================
def _call_synth(model, x, x_lengths, steps: int):
    """
    Matcha codebases differ a bit. We introspect the signature and call safely.
    We want MEL out (not vocoder).
    """
    # Prefer synthesize/synthesise if exists
    fn = None
    for name in ["synthesize", "synthesise"]:
        if hasattr(model, name):
            fn = getattr(model, name)
            fn_name = name
            break
    if fn is None:
        raise RuntimeError("MatchaTTS has no synthesize/synthesise method.")

    sig = inspect.signature(fn)
    kwargs = {}

    # common parameter names across versions
    if "n_timesteps" in sig.parameters:
        kwargs["n_timesteps"] = steps
    elif "n_steps" in sig.parameters:
        kwargs["n_steps"] = steps
    elif "steps" in sig.parameters:
        kwargs["steps"] = steps

    if "temperature" in sig.parameters:
        kwargs["temperature"] = TEMPERATURE

    if "speaking_rate" in sig.parameters:
        kwargs["speaking_rate"] = SPEAKING_RATE

    # ensure we don't try to run vocoder
    for k in ["vocoder", "use_vocoder", "run_vocoder"]:
        if k in sig.parameters:
            kwargs[k] = None if k == "vocoder" else False

    out = fn(x, x_lengths, **kwargs)
    return out, fn_name, sig, kwargs

@torch.no_grad()
def synthesize_mel(text: str, steps: int, debug_once: bool = False):
    x, x_lengths, used_cleaners = to_batch_tokens(text)

    out, fn_name, sig, kwargs = _call_synth(model, x, x_lengths, steps)

    if debug_once:
        print(f"[DEBUG] cleaners={used_cleaners}", flush=True)
        print(f"[DEBUG] called model.{fn_name}{sig}", flush=True)
        print(f"[DEBUG] kwargs={kwargs}", flush=True)
        print(f"[DEBUG] output type={type(out)}", flush=True)

    # Extract mel from common return formats:
    mel = None
    if isinstance(out, torch.Tensor):
        mel = out
    elif isinstance(out, (tuple, list)):
        # pick first tensor with 3 dims
        for item in out:
            if torch.is_tensor(item) and item.ndim >= 2:
                mel = item
                break
    elif isinstance(out, dict):
        for key in ["mel", "mel_out", "mels", "spectrogram", "spec", "y"]:
            if key in out and torch.is_tensor(out[key]):
                mel = out[key]
                break

    if mel is None:
        raise RuntimeError("Could not extract mel tensor from model output.")

    # mel shape can be [B, n_mels, T] or [B, T, n_mels]
    mel = mel.detach()
    if mel.ndim == 2:
        # [n_mels, T] or [T, n_mels] => add batch
        mel = mel.unsqueeze(0)

    # determine time frames T
    if mel.shape[1] in (80, 100, 128):  # likely [B, n_mels, T]
        n_frames = mel.shape[2]
    else:
        # assume [B, T, n_mels]
        n_frames = mel.shape[1]

    audio_dur = (n_frames * HOP_LENGTH) / float(SAMPLING_RATE)
    return audio_dur, n_frames

# ============================================================
# 8) Warmup
# ============================================================
print(f"\n[WARMUP] {WARMUP_N} warm-up runs (not measured)", flush=True)
for i in range(min(WARMUP_N, len(texts))):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    _dur, _frames = synthesize_mel(texts[i], steps=4, debug_once=(i == 0))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
print("[WARMUP] Done\n", flush=True)

# ============================================================
# 9) Main RTF loop
# ============================================================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

print("[RUN] Measuring RTF (mel-only)...", flush=True)

total_jobs = len(STEPS_LIST) * len(texts)
job = 0
t_global0 = time.perf_counter()

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "steps", "gen_time_s", "audio_dur_s", "n_frames", "rtf"]
    )
    writer.writeheader()

    for steps in STEPS_LIST:
        print(f"\n[STEPS] steps={steps}", flush=True)

        for i, text in enumerate(texts):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            audio_dur, n_frames = synthesize_mel(text, steps=steps)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            gen_time = t1 - t0
            rtf = gen_time / audio_dur if audio_dur > 0 else float("nan")

            writer.writerow({
                "id": i,
                "steps": steps,
                "gen_time_s": gen_time,
                "audio_dur_s": audio_dur,
                "n_frames": int(n_frames),
                "rtf": rtf
            })

            job += 1
            if (job % PRINT_EVERY == 0) or (job == total_jobs):
                elapsed = time.perf_counter() - t_global0
                avg = elapsed / job
                remaining = total_jobs - job
                eta = remaining * avg
                print(
                    f"[PROGRESS] {job}/{total_jobs} | "
                    f"elapsed={elapsed/60:.1f} min | avg/job={avg:.3f}s | ETA={eta/60:.1f} min",
                    flush=True
                )

print(f"\n[OK] CSV saved: {OUT_CSV}", flush=True)
print("[DONE] Mel-only RTF finished.", flush=True)
