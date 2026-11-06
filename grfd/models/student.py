from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from grfd.config import StudentConfig


class GatedRandomForestDistillationModel(nn.Module):
    """Shared encoder, a softmax-gated mixture of expert heads, and a linear skip head."""

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 96, n_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(n_experts)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_experts),
            nn.Softmax(dim=-1),
        )
        self.skip = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        weights = self.gate(z)
        logits = torch.stack([expert(z) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * logits).sum(dim=1) + self.skip(z)


def build_student(input_dim: int, num_classes: int, cfg: StudentConfig, seed: int, device: torch.device) -> GatedRandomForestDistillationModel:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GatedRandomForestDistillationModel(input_dim, num_classes, cfg.hidden_dim, cfg.n_experts, cfg.dropout)
    return model.to(device)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = [
        model(torch.from_numpy(X[start:start + batch_size]).to(device)).softmax(-1).cpu().numpy()
        for start in range(0, len(X), batch_size)
    ]
    if not chunks:
        return np.empty((0, model.num_classes), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def save_checkpoint(
    model: GatedRandomForestDistillationModel,
    path: str | Path,
    *,
    feature_cols: list[str],
    class_names: list[str],
    fold: int,
    after_task: int,
    variant: str = "joint",
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "num_classes": model.num_classes,
            "hidden_dim": model.hidden_dim,
            "n_experts": model.n_experts,
            "fold": fold,
            "after_task": after_task,
            "variant": variant,
            "feature_cols": feature_cols,
            "class_names": class_names,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[GatedRandomForestDistillationModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GatedRandomForestDistillationModel(ckpt["input_dim"], ckpt["num_classes"], ckpt["hidden_dim"], ckpt["n_experts"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
