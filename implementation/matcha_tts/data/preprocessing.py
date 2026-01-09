"""Fonctions de prétraitement texte et audio : normalisation, extraction des mél-spectrogrammes et masques."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio

# soundfile est optionnel, utilisé en repli si torchcodec/FFmpeg est absent
try:
    import soundfile as sf
except ImportError:  # pragma: no cover - dépendance optionnelle
    sf = None


def _fallback_load(audio_path: Path):
    """Essaie des backends alternatifs si torchcodec/FFmpeg échoue.

    Ordre :
    1) sox_io via torchaudio
    2) soundfile (libsndfile) si disponible
    """
    # 1) sox_io (intégré dans torchaudio)
    try:
        from torchaudio.backend import sox_io_backend

        return sox_io_backend.load(audio_path, normalize=True)
    except Exception:
        pass

    # 2) soundfile direct (sans passer par torchaudio)
    if sf is not None:
        try:
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return torch.tensor(data, dtype=torch.float32).unsqueeze(0), sr
        except Exception:
            pass

    raise RuntimeError(
        "torchaudio.load et les backends de secours ont échoué. "
        "Installez FFmpeg compatible ou le paquet soundfile (pip install soundfile)."
    )


def normalize_text(text: str) -> str:
    """Nettoie et normalise le texte pour la synthèse vocale."""
    # Ici on applique une normalisation minimale; on peut étendre avec NFKC, abréviations, etc.
    return text.strip()


def load_audio(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    """Charge un wav mono et le resample vers la fréquence souhaitée."""
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        waveform, sr = _fallback_load(audio_path)
    if waveform.ndim > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
    return waveform.squeeze(0)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> torch.Tensor:
    """Calcule un mél-spectrogramme à partir d'un signal brut.

    On suit la recette classique (STFT → projection mel) employée dans Grad-TTS :
    - fenêtre Hanning
    - puissance convertie en dB logarithmique pour une meilleure stabilité numérique
    """

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
        normalized=False,
    )
    mel = mel_transform(waveform.unsqueeze(0))  # (1, n_mels, T)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel)
    return mel_db.squeeze(0)  # (n_mels, T)


def normalize_mel(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Applique une normalisation type Grad-TTS : centrage et réduction par écart-type."""
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + eps
    return (mel - mean) / std


def load_metadata(metadata_path: Path) -> List[Tuple[str, str]]:
    """Charge les (id, transcription normalisée) depuis metadata.csv de LJSpeech."""
    pairs: List[Tuple[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            utt_id, _, normalized_text = parts[0], parts[1], parts[2]
            pairs.append((utt_id, normalize_text(normalized_text)))
    return pairs
