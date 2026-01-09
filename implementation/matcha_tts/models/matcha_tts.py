"""Module principal combinant encodeur de texte, prédicteur de durées, décodeur et flow matching."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Permet d'exécuter ce fichier en script direct pour les petits tests locaux.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.models.encoder.text_encoder import TextEncoder
from matcha_tts.models.duration.duration_predictor import DurationPredictor
from matcha_tts.models.decoder.unet_1d import UNet1D
from matcha_tts.inference.sampler import euler_sampler
from matcha_tts.text.symbols import SYMBOLS



class MatchaTTS(nn.Module):
    """Assemble les sous-modèles et expose les méthodes d'entraînement et d'inférence."""

    def __init__(self, config: dict) -> None:
        """Instancie chaque composant à partir de la configuration centralisée."""
        super().__init__()
        model_cfg = config["model"]
        audio_cfg = config["audio"]
        self.text_dim = model_cfg["text_encoder"]["embedding_dim"]
        self.n_mels = audio_cfg["n_mels"]

        vocab_size = len(SYMBOLS)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=self.text_dim,
            num_layers=model_cfg["text_encoder"]["num_layers"],
            num_heads=model_cfg["text_encoder"]["num_heads"],
            dropout=model_cfg["text_encoder"]["dropout"],
        )
        self.duration_predictor = DurationPredictor(
            channels=model_cfg["duration_predictor"]["channels"],
            kernel_size=model_cfg["duration_predictor"]["kernel_size"],
            dropout=model_cfg["duration_predictor"]["dropout"],
        )
        self.mu_proj = nn.Linear(self.text_dim, self.n_mels)
        dec_cfg = model_cfg["decoder"]
        dec_channels = dec_cfg.get("channels", self.n_mels)
        # Le décodeur opère sur les mél (C = n_mels), on force la cohérence si nécessaire.
        if dec_channels != self.n_mels:
            dec_channels = self.n_mels
        self.decoder = UNet1D(
            channels=dec_channels,
            attention_heads=dec_cfg["attention_heads"],
            time_embedding_dim=dec_cfg["time_embedding_dim"],
            hidden_dim=dec_cfg["hidden_dim"],
            num_layers_per_block=dec_cfg["num_layers_per_block"],
            num_down_blocks=dec_cfg["num_down_blocks"],
            num_mid_blocks=dec_cfg["num_mid_blocks"],
            num_up_blocks=dec_cfg["num_up_blocks"],
            dropout=dec_cfg.get("dropout", 0.0),
        )

    def upsample_text(self, encoded_text: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expanse les représentations texte selon les durées pour obtenir une séquence au pas frame.

        Chaque vecteur phonémique est répété `durations` fois, préservant la monotonie
        texte→temps. Le masque mél résultant suit la convention True = valide, False = padding.
        """

        batch, t_text, dim = encoded_text.shape
        durations = durations.long()

        upsampled_list = []
        mel_lengths = []
        for b in range(batch):
            reps = durations[b]
            pieces = []
            for t in range(t_text):
                d = reps[t].item()
                if d <= 0:
                    continue
                pieces.append(encoded_text[b, t].unsqueeze(0).repeat(d, 1))
            if len(pieces) == 0:
                upsampled = torch.zeros((1, dim), device=encoded_text.device)
            else:
                upsampled = torch.cat(pieces, dim=0)
            upsampled_list.append(upsampled)
            mel_lengths.append(upsampled.size(0))

        max_mel = max(mel_lengths)
        upsampled_text = torch.zeros((batch, max_mel, dim), device=encoded_text.device)
        mel_mask = torch.zeros((batch, max_mel), dtype=torch.bool, device=encoded_text.device)

        for b, up in enumerate(upsampled_list):
            L = up.size(0)
            upsampled_text[b, :L] = up
            mel_mask[b, :L] = True  # True = positions valides

        return upsampled_text, mel_mask

    def compute_mu(self, upsampled_text: torch.Tensor) -> torch.Tensor:
        """Projette les représentations upsamplées dans l'espace mél pour obtenir µ (condition)."""
        return self.mu_proj(upsampled_text)

    def forward(self, tokens: torch.Tensor, mel: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """Calcule les sorties intermédiaires nécessaires à la perte (encodage, durées, prédiction mél)."""
        # TODO: implémenter la passe avant complète (alignement, décodeur, flow)
        raise NotImplementedError("Passe avant du modèle à implémenter.")

    def inference(self, tokens: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Génère un mél-spectrogramme à partir d'une séquence de tokens via le sampler CFM."""
        # TODO: implémenter l'inférence complète avec phonémisation et duration predictor
        raise NotImplementedError("Utiliser la méthode infer(phoneme_ids, phoneme_mask, ...) pour l'inférence.")

    @torch.no_grad()
    def infer(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pipeline d'inférence texte→mél : encode, prédit durées, upsample, résout l'ODE.

        Args:
            phoneme_ids: (B, T_text) ids des phonèmes.
            phoneme_mask: (B, T_text) bool, True = valide.
            n_steps: nombre de pas Euler pour résoudre le flow.
            temperature: échelle du bruit initial x0.

        Returns:
            mel_pred: (B, n_mels, T_mel) mél généré.
            mel_mask: (B, T_mel) bool mask (True = valide).
        """

        encoded = self.text_encoder(phoneme_ids, phoneme_mask)
        log_dur = self.duration_predictor(encoded, phoneme_mask)
        durations = self.duration_predictor.infer_durations(log_dur, phoneme_mask)

        upsampled, mel_mask = self.upsample_text(encoded, durations)
        mu = self.compute_mu(upsampled)  # (B, T_mel, n_mels)

        # Prépare x0 et transpose µ pour matcher le format (B, C, T) attendu par le UNet.
        mu_cond = mu.transpose(1, 2)  # (B, n_mels, T_mel)
        x0 = torch.randn_like(mu_cond) * temperature

        mel_pred = euler_sampler(self.decoder, x0, mu_cond, mel_mask, n_steps)
        return mel_pred, mel_mask


if __name__ == "__main__":
    # Test rapide d'upsampling : vérifie T_mel = somme des durées et le masque associé.
    torch.manual_seed(0)
    config = {
        "model": {
            "vocab_size": 10,
            "text_encoder": {"embedding_dim": 8, "num_layers": 1, "num_heads": 2, "dropout": 0.1},
            "duration_predictor": {"channels": 8, "kernel_size": 3, "dropout": 0.1},
            "decoder": {
                "channels": 8,
                "attention_heads": 2,
                "time_embedding_dim": 8,
                "hidden_dim": 8,
                "num_layers_per_block": 1,
                "num_down_blocks": 1,
                "num_mid_blocks": 1,
                "num_up_blocks": 1,
                "dropout": 0.0,
            },
        },
        "audio": {"n_mels": 80},
    }
    model = MatchaTTS(config)
    encoded = torch.randn(2, 3, 8)
    durations = torch.tensor([[2, 1, 0], [1, 2, 1]])
    up, mask = model.upsample_text(encoded, durations)
    assert up.shape[1] == durations[0].sum().item() or up.shape[1] == durations[1].sum().item()
    assert mask.shape[:2] == up.shape[:2]
    mu = model.compute_mu(up)
    assert mu.shape[2] == config["audio"]["n_mels"]
    print("Upsample test ok. µ shape:", mu.shape)

    # Test rapide d'inférence (decoder non implémenté → zeros si placeholder)
    phoneme_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    phoneme_mask = phoneme_ids != 0
    mel_pred, mel_mask = model.infer(phoneme_ids, phoneme_mask, n_steps=4, temperature=1.0)
    assert mel_pred.shape[0] == phoneme_ids.shape[0]
    print("Infer test shape:", mel_pred.shape, "mask shape:", mel_mask.shape)
