"""Utilitaires audio pour charger, sauvegarder et transformer des signaux en mél-spectrogrammes."""
from __future__ import annotations

from pathlib import Path

import torch


def load_wav(path: Path, sample_rate: int) -> torch.Tensor:
    """Charge un fichier wav en tenseur mono à l'échantillonnage désiré."""
    # TODO: utiliser torchaudio ou librosa pour le chargement et le resampling
    raise NotImplementedError("Chargement audio à implémenter.")


def save_wav(waveform: torch.Tensor, path: Path, sample_rate: int) -> None:
    """Sauvegarde un tenseur audio mono au format wav."""
    # TODO: écrire le wav sur disque
    raise NotImplementedError("Sauvegarde audio à implémenter.")


def mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int, win_length: int, n_mels: int, fmin: int, fmax: int) -> torch.Tensor:
    """Calcule un mél-spectrogramme à partir d'un signal audio brut."""
    # TODO: appeler torchaudio.transforms.MelSpectrogram ou équivalent
    raise NotImplementedError("Calcul du mél-spectrogramme à implémenter.")
