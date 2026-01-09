"""Boucle d'entraînement principale : gestion des données, de l'optimisation et de la journalisation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.matcha_tts import MatchaTTS
from .losses import compute_losses
from .scheduler import build_scheduler
from matcha_tts.data.phonemizer import PhonemizerWrapper
from matcha_tts.text.symbols import PAD_ID
from matcha_tts.text.text_to_sequence import phonemes_to_ids


class _EMA:
    """Maintient une EMA des poids du décodeur pour stabiliser l'inférence."""

    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items() if v.is_floating_point()}
        self.backup: Dict[str, torch.Tensor] | None = None
        self.module = module

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.module.state_dict().items():
            if not param.is_floating_point():
                continue
            self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        self.backup = {k: v.detach().clone() for k, v in self.module.state_dict().items() if v.is_floating_point()}
        self.module.load_state_dict({**self.module.state_dict(), **self.shadow}, strict=False)

    def restore(self) -> None:
        if self.backup is None:
            return
        self.module.load_state_dict({**self.module.state_dict(), **self.backup}, strict=False)
        self.backup = None


class Trainer:
    """Encapsule la logique d'entraînement pour Matcha-TTS (pédagogique)."""

    def __init__(
        self,
        model: MatchaTTS,
        train_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        val_loader: Optional[DataLoader] = None,
        overfit_one_batch: bool = False,
        output_dir: Optional[Path] = None,
        phonemizer: Optional[PhonemizerWrapper] = None,
        validation_texts: Optional[List[str]] = None,
        val_every: Optional[int] = None,
        enable_validation: bool = True,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.overfit_one_batch = overfit_one_batch

        train_cfg = config.get("training", {})
        self.max_steps = int(train_cfg.get("max_steps", 100000))
        self.log_every = int(train_cfg.get("log_every", 100))
        self.ckpt_every = int(train_cfg.get("ckpt_every", 1000))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        # torch.cuda.amp est déprécié : on bascule sur torch.amp (device_type non supporté sur certaines versions)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.ema_decay = float(train_cfg.get("ema_decay", 0.0))
        self.overfit_one_batch = bool(train_cfg.get("overfit_one_batch", False)) or overfit_one_batch
        self.overfit_steps = int(train_cfg.get("overfit_steps", 200))

        lr = float(train_cfg.get("lr", 2e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-6))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, config)

        self.global_step = 0
        self.fixed_batch = None  # pour overfit_one_batch
        self._cached_batch = None

        val_cfg = config.get("validation", {})
        self.validation_texts = validation_texts if validation_texts is not None else val_cfg.get("texts", [])
        cfg_val_every = val_cfg.get("every_steps")
        self.val_every = int(val_every) if val_every is not None else (int(cfg_val_every) if cfg_val_every else None)
        self.enable_validation = enable_validation and bool(val_cfg.get("enabled", True)) and bool(self.validation_texts)

        infer_cfg = config.get("inference", {})
        self.infer_steps = int(infer_cfg.get("n_steps", 10))
        self.infer_temperature = float(infer_cfg.get("temperature", 0.8))

        self.output_dir = Path(output_dir) if output_dir is not None else Path("outputs")
        self.val_mel_dir = self.output_dir / "val_mels"
        self.val_mel_dir.mkdir(parents=True, exist_ok=True)

        self.phonemizer = phonemizer

        self.decoder_ema: Optional[_EMA] = None
        if self.ema_decay > 0.0:
            self.decoder_ema = _EMA(self.model.decoder, self.ema_decay)

    def _step(self, batch: dict) -> dict:
        """Effectue une passe avant/arrière sur un batch et retourne les pertes."""
        self.optimizer.zero_grad()

        mel_shape = tuple(batch["mel"].shape)

        # torch.amp.autocast remplace torch.cuda.amp.autocast (migration API)
        # device_type est requis sur certaines versions; on choisit selon le device courant
        autocast_device = self.device.type if hasattr(self.device, "type") else "cuda"
        with torch.amp.autocast(device_type=autocast_device, enabled=self.use_amp):
            losses = compute_losses(batch, self.model, self.config, self.device)
            loss_total = losses["loss_total"]

        if self.use_amp:
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.decoder_ema is not None:
            self.decoder_ema.update()

        # Norme de gradient pour le logging (non strictement exacte sous AMP)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        losses = {k: v.detach().item() for k, v in losses.items()}
        losses["grad_norm"] = grad_norm
        losses["mel_shape"] = mel_shape
        return losses

    def train_one_epoch(self) -> None:
        """Boucle une époque ou jusqu'à max_steps, avec logs/checkpoints périodiques."""
        self.model.train()

        if self.overfit_one_batch and self._cached_batch is None:
            first_batch = next(iter(self.train_loader))
            self._cached_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in first_batch.items()}
            print("OVERFIT MODE: using cached batch")

        for batch in self.train_loader:
            if self.overfit_one_batch and self._cached_batch is not None:
                batch = self._cached_batch

            losses = self._step(batch)
            self.global_step += 1

            if self.global_step % self.log_every == 0:
                self._log_step(losses)

            if self.enable_validation and self.val_every and self.global_step % self.val_every == 0:
                self._run_validation()

            if self.global_step % self.ckpt_every == 0:
                ckpt_path = self.output_dir / "checkpoints" / f"step_{self.global_step}.pt"
                self.save_checkpoint(ckpt_path, self.global_step)

            if self.global_step >= self.max_steps:
                break
            if self.overfit_one_batch and self.global_step >= self.overfit_steps:
                break

    def _log_step(self, losses: Dict[str, float]) -> None:
        """Log texte minimaliste (peut être étendu à TensorBoard)."""
        msg = [f"step {self.global_step}"]
        mel_shape = losses.pop("mel_shape", None)
        for k, v in losses.items():
            msg.append(f"{k}: {v:.4f}")
        if mel_shape is not None:
            msg.append(f"mel_shape: {mel_shape}")
        print(" | ".join(msg))

    def _infer_text_to_mel(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Phonémise un texte et exécute l'inférence pour générer un mél."""

        if self.phonemizer is None:
            raise ValueError("Phonemizer requis pour la validation")

        phoneme_ids = phonemes_to_ids(self.phonemizer(text))
        if len(phoneme_ids) == 0:
            phoneme_ids = [PAD_ID]

        ids = torch.tensor(phoneme_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.ones((1, ids.shape[1]), dtype=torch.bool, device=self.device)

        mel_pred, mel_mask = self.model.infer(
            ids,
            mask,
            n_steps=self.infer_steps,
            temperature=self.infer_temperature,
        )

        return mel_pred.squeeze(0), mel_mask.squeeze(0)

    def _run_validation(self) -> None:
        """Génère et sauvegarde des mels sur un petit ensemble de phrases fixes."""

        if not self.enable_validation or not self.validation_texts:
            return

        self.model.eval()
        decoder_ema_applied = False
        if self.decoder_ema is not None:
            self.decoder_ema.apply_shadow()
            decoder_ema_applied = True

        try:
            written = 0
            with torch.no_grad():
                for idx, text in enumerate(self.validation_texts):
                    mel_pred, mel_mask = self._infer_text_to_mel(text)
                    save_path = self.val_mel_dir / f"step_{self.global_step:07d}_{idx}.pt"
                    payload = {
                        "mel": mel_pred.cpu(),
                        "mel_mask": mel_mask.cpu(),
                        "text": text,
                        "step": self.global_step,
                    }
                    torch.save(payload, save_path)
                    written += 1

            print(f"[val] step {self.global_step}: wrote {written} mel previews to {self.val_mel_dir}")
        finally:
            if decoder_ema_applied:
                self.decoder_ema.restore()
            self.model.train()

    def save_checkpoint(self, path: Path, step: int) -> None:
        """Enregistre un checkpoint pour reprise ultérieure."""
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": step,
            "config": self.config,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path) -> int:
        """Charge un checkpoint et retourne le step associé."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state.get("optimizer", {}))
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state.get("step", 0)
        return self.global_step
