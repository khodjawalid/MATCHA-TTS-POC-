"""Pertes et alignements (MAS) utilisés pour entraîner Matcha-TTS."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mel_reconstruction_loss(pred_mel: torch.Tensor, target_mel: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Calcule une perte L1 sur les mél-spectrogrammes, optionnellement masquée.

    mask: (B, T) bool, True = valide. Normalisation sur les positions valides.
    """

    if mask is None:
        return F.l1_loss(pred_mel, target_mel)

    valid = mask.float().unsqueeze(1)  # (B, 1, T)
    l1 = torch.abs(pred_mel - target_mel) * valid
    denom = torch.clamp(valid.sum() * pred_mel.shape[1], min=1.0)
    return l1.sum() / denom


def duration_loss(pred_durations: torch.Tensor, target_durations: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Calcule une perte MSE sur les durées prédites, en ignorant les positions padding.

    mask: (B, T_text) bool, True = valide.
    """

    if mask is None:
        return F.mse_loss(pred_durations, target_durations)

    valid = mask.float()
    mse = ((pred_durations - target_durations) ** 2) * valid
    denom = torch.clamp(valid.sum(), min=1.0)
    return mse.sum() / denom


def monotonic_alignment_search(log_probs: torch.Tensor, text_mask: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
    """Recherche d'alignement monotone (MAS) type Grad-TTS entre texte et mél.

    Args:
        log_probs: (B, T_text, T_mel) log-probabilités d'aligner chaque frame sur un phonème.
        text_mask: (B, T_text) bool, True = valide (phonème présent).
        mel_mask: (B, T_mel) bool, True = valide (frame présente).

    Returns:
        alignment: (B, T_text, T_mel) binaire, 1 si la frame est alignée au phonème.

    Principe (Viterbi monotone): on autorise soit rester sur le même phonème, soit avancer
    au phonème suivant en avançant d'une frame. On maximise la somme des log-probabilités
    et on rétro-propage le chemin optimal pour obtenir un alignement dur.
    """

    B, T_text, T_mel = log_probs.shape
    alignments = torch.zeros_like(log_probs, dtype=torch.float)

    for b in range(B):
        text_len = text_mask[b].sum().item()
        mel_len = mel_mask[b].sum().item()

        # Tronque aux longueurs valides pour accélérer le DP
        lp = log_probs[b, :text_len, :mel_len]

        # DP tables
        dp = torch.full((text_len, mel_len), float("-inf"), device=log_probs.device)
        backptr = torch.zeros((text_len, mel_len), dtype=torch.long, device=log_probs.device)

        # Initialisation : première frame forcée sur premier phonème
        dp[0, 0] = lp[0, 0]
        for j in range(1, mel_len):
            dp[0, j] = dp[0, j - 1] + lp[0, j]
            backptr[0, j] = 0  # reste sur même phonème

        # Remplissage DP : transitions (stay) ou (next phoneme + frame)
        for i in range(1, text_len):
            for j in range(i, mel_len):  # j>=i pour assurer monotonie
                stay = dp[i, j - 1]
                move = dp[i - 1, j - 1]
                if stay >= move:
                    dp[i, j] = stay + lp[i, j]
                    backptr[i, j] = i
                else:
                    dp[i, j] = move + lp[i, j]
                    backptr[i, j] = i - 1

        # Rétro-propagation du meilleur chemin
        alignment_b = torch.zeros((text_len, mel_len), device=log_probs.device)
        i, j = text_len - 1, mel_len - 1
        alignment_b[i, j] = 1.0
        while j > 0:
            prev_i = backptr[i, j].item()
            j -= 1
            if prev_i == i:
                # transition stay (même phonème)
                alignment_b[i, j] = 1.0
            else:
                # transition move (phonème précédent)
                i = prev_i
                alignment_b[i, j] = 1.0

        # Replace dans le tenseur batché
        alignments[b, :text_len, :mel_len] = alignment_b

    return alignments


def extract_durations(alignment: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    """Calcule les durées (frames par phonème) à partir d'un alignement dur."""
    durations = alignment.sum(dim=-1)  # (B, T_text)
    durations = durations * text_mask.float()
    return durations


def prior_loss(log_probs: torch.Tensor, alignment: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
    """Prior loss inspirée de Grad-TTS : -log p(z|x) via alignement dur.

    On suppose un bruit gaussien isotrope; log_probs sont les log-p de mapper chaque
    frame mél sur un phonème encodé. La loss agrège les log-probs alignés et les
    normalise par le nombre de frames valides.
    """

    aligned_logp = alignment * log_probs  # (B, T_text, T_mel)
    # On ne somme que sur les frames valides
    mel_valid = mel_mask.float().unsqueeze(1)
    aligned_logp = aligned_logp * mel_valid
    total_logp = aligned_logp.sum()
    total_frames = mel_valid.sum()
    # Minimiser -log p
    return -total_logp / torch.clamp(total_frames, min=1.0)


def conditional_flow_matching_loss(
    unet,
    mel: torch.Tensor,
    mu: torch.Tensor,
    mel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perte OT-CFM : le décodeur prédit le champ de vitesse v_t entre bruit et mél cible.

    Étapes (inspirées de Grad-TTS) :
    1. Échantillonner t ~ U(0, 1) pour chaque élément du batch.
    2. Générer x0 ~ N(0, I) (même shape que mel) et former la trajectoire linéaire
       x_t = (1 - t) x0 + t x1 où x1 = mel réel, µ est la condition.
    3. Cible de vitesse : v_target = x1 - x0 (champ OT constant le long de la ligne droite).
    4. Prédiction : v_pred = unet(x_t, t, conditioning=mu, mel_mask=mel_mask).
    5. Loss : MSE sur v_pred vs v_target, masquée sur les frames padding (mask False).
    """

    # t : (B, 1, 1) pour broadcast sur (B, C, T)
    batch = mel.shape[0]
    device = mel.device
    t = torch.rand(batch, 1, 1, device=device)

    x0 = torch.randn_like(mel)
    x1 = mel
    x_t = (1.0 - t) * x0 + t * x1
    v_target = x1 - x0

    # Ajuster µ en (B, C, T) si fourni en (B, T, C)
    if mu.dim() == 3 and mu.shape[1] == mel.shape[2]:
        mu_cond = mu.transpose(1, 2)
    else:
        mu_cond = mu

    v_pred = unet(x_t, t.squeeze(-1).squeeze(-1), conditioning=mu_cond, mel_mask=mel_mask)

    if __debug__:
        assert v_pred.shape == v_target.shape, "v_pred et v_target doivent avoir la même shape"

    if mel_mask is None:
        return F.mse_loss(v_pred, v_target)

    if __debug__:
        assert mel_mask.shape[-1] == v_target.shape[-1], "mel_mask doit correspondre à la longueur temporelle"

    valid = mel_mask.float().unsqueeze(1)  # (B, 1, T)
    mse = ((v_pred - v_target) ** 2) * valid
    denom = torch.clamp(valid.sum() * v_pred.shape[1], min=1.0)
    loss = mse.sum() / denom

    return loss


def compute_losses(batch: dict, model, config: dict, device: torch.device) -> dict:
    """Calcule les pertes principales (prior, durée, OT-CFM) pour un batch.

    Args:
        batch: dictionnaire issu du dataloader (phonemes, masks, mel...).
        model: instance de MatchaTTS (compose encodeur, duration predictor, décodeur).
        config: configuration expérimentale (inclut loss_weights).
        device: torch.device cible.

    Convention des masques : True = valide, False = padding.
    """

    loss_w = config.get("loss_weights", {"prior": 1.0, "duration": 1.0, "cfm": 1.0})

    phonemes = batch["phonemes"].to(device)
    phoneme_mask = batch["phoneme_mask"].to(device)
    mel = batch["mel"].to(device)  # (B, n_mels, T)
    mel_mask = batch.get("mel_mask")
    if mel_mask is not None:
        mel_mask = mel_mask.to(device)
    else:
        # Par défaut, tout est valide si le masque n'est pas fourni
        mel_mask = torch.ones(mel.shape[0], mel.shape[2], dtype=torch.bool, device=device)

    # 1) Encodage texte
    encoded_text = model.text_encoder(phonemes, phoneme_mask)  # (B, T_text, D)

    # 2) Log-probs phonème->frame pour MAS (score de similarité négatif L2)
    text_proj = model.mu_proj(encoded_text)  # (B, T_text, n_mels)
    mel_frames = mel.transpose(1, 2)  # (B, T_mel, n_mels)
    # log_probs ~ -||f(text) - mel||^2 (pas de softmax, sert de score pour MAS)
    log_probs = -((text_proj.unsqueeze(2) - mel_frames.unsqueeze(1)) ** 2).mean(dim=-1)

    # 3) MAS -> alignement dur et durées cibles
    alignment = monotonic_alignment_search(log_probs, phoneme_mask, mel_mask)
    durations_target = extract_durations(alignment, phoneme_mask)  # (B, T_text)
    durations_target_long = durations_target.long()

    # 4) Prior loss
    loss_prior = prior_loss(log_probs, alignment, mel_mask)

    # 5) Duration loss (sur log(d+1))
    log_dur_pred = model.duration_predictor(encoded_text, phoneme_mask)
    target_log_dur = torch.log(durations_target + 1.0 + 1e-5)
    loss_duration = duration_loss(log_dur_pred, target_log_dur, phoneme_mask)

    # 6) Upsampling supervision (durations_target) -> µ
    upsampled_text, mel_mask_from_dur = model.upsample_text(encoded_text, durations_target_long)
    mu = model.compute_mu(upsampled_text)  # (B, T_mel_hat, n_mels)
    mu_cond = mu.transpose(1, 2)  # (B, n_mels, T_mel_hat)

    target_len = mel.shape[-1]
    if mu_cond.shape[-1] != target_len:
        raise ValueError("mu_cond doit correspondre à la longueur du mél cible")

    mel_mask_cfm = mel_mask if mel_mask is not None else mel_mask_from_dur
    if mel_mask_cfm.shape[-1] != target_len:
        raise ValueError("mel_mask doit être aligné sur la longueur du mél")

    # 7) OT-CFM loss
    loss_cfm = conditional_flow_matching_loss(
        unet=model.decoder,
        mel=mel,
        mu=mu_cond,
        mel_mask=mel_mask_cfm,
    )

    loss_total = (
        loss_w.get("prior", 1.0) * loss_prior
        + loss_w.get("duration", 1.0) * loss_duration
        + loss_w.get("cfm", 1.0) * loss_cfm
    )

    return {
        "loss_total": loss_total,
        "loss_prior": loss_prior.detach(),
        "loss_duration": loss_duration.detach(),
        "loss_cfm": loss_cfm.detach(),
    }


if __name__ == "__main__":
    # Test rapide sur tenseurs factices pour vérifier formes et somme des durées.
    B, T_text, T_mel = 2, 4, 6
    torch.manual_seed(0)
    lp = torch.randn(B, T_text, T_mel)
    text_mask = torch.ones(B, T_text, dtype=torch.bool)
    mel_mask = torch.ones(B, T_mel, dtype=torch.bool)

    align = monotonic_alignment_search(lp, text_mask, mel_mask)
    durs = extract_durations(align, text_mask)
    assert torch.allclose(durs.sum(dim=1), torch.tensor([T_mel, T_mel], dtype=torch.float)), "Somme des durées incorrecte"
    pl = prior_loss(lp, align, mel_mask)
    print("Align shape", align.shape, "Durations", durs, "Prior loss", pl.item())

    # Test factice CFM : vérifie que la loss est finie
    mel = torch.randn(B, 80, T_mel)
    mu = torch.randn(B, T_mel, 80)

    class DummyUNet(torch.nn.Module):
        def forward(self, x, t, conditioning=None, mel_mask=None):
            return torch.zeros_like(x)

    dummy_unet = DummyUNet()
    cfm_loss = conditional_flow_matching_loss(dummy_unet, mel, mu, mel_mask)
    assert torch.isfinite(cfm_loss), "CFM loss non finie"
    print("CFM loss test ok", cfm_loss.item())
