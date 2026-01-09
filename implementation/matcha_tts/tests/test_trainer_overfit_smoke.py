"""
Smoke test du mode overfit du Trainer : vérifie que le premier batch est réutilisé
et que le compteur de steps s'incrémente en conséquence.
"""

import sys
from pathlib import Path
import types

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcha_tts.training.trainer import Trainer


class DummyDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "mel": torch.randn(1, 4, 5),
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


def test_trainer_overfit_reuses_batch():
    device = torch.device("cpu")
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    config = {
        "training": {
            "batch_size": 1,
            "max_steps": 10,
            "log_every": 100,
            "ckpt_every": 1000,
            "grad_clip": 1.0,
            "amp": False,
            "num_workers": 0,
            "overfit_one_batch": True,
            "overfit_steps": 2,
        },
        "audio": {"n_mels": 4},
        "model": {
            "text_encoder": {"embedding_dim": 4, "num_layers": 1, "num_heads": 1, "dropout": 0.0},
            "duration_predictor": {"channels": 4, "kernel_size": 3, "dropout": 0.0},
            "decoder": {
                "channels": 4,
                "attention_heads": 1,
                "time_embedding_dim": 4,
                "hidden_dim": 4,
                "num_layers_per_block": 1,
                "num_down_blocks": 1,
                "num_mid_blocks": 1,
                "num_up_blocks": 1,
                "dropout": 0.0,
            },
        },
        "inference": {"n_steps": 1, "temperature": 1.0},
    }

    model = DummyModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=config,
        device=device,
        overfit_one_batch=True,
        output_dir=Path("outputs"),
    )

    seen = []

    def fake_step(self, batch):
        seen.append(id(batch["mel"]))
        return {"loss_total": torch.tensor(0.0)}

    trainer._step = types.MethodType(fake_step, trainer)

    trainer.train_one_epoch()

    assert trainer._cached_batch is not None, "Le batch n'a pas été mis en cache"
    assert len(set(seen)) == 1, "Le batch overfit doit être réutilisé à chaque step"
    assert trainer.global_step == trainer.overfit_steps, "Le compteur de steps doit s'arrêter sur overfit_steps"


if __name__ == "__main__":
    test_trainer_overfit_reuses_batch()
    print("✅ overfit smoke test passed")
