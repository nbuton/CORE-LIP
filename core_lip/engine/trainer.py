"""
core_lip/trainer.py
-------------------
Low-level training primitives and small helpers.

    - set_seed        : global reproducibility
    - get_config      : YAML → FullConfig
    - train_one_epoch : single epoch with gradient accumulation and clipping
"""

from __future__ import annotations

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from core_lip.config import FullConfig


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def get_config(yaml_path: str) -> FullConfig:
    """Parse a YAML config file and return a validated :class:`FullConfig`."""
    with open(yaml_path, "r") as fh:
        return FullConfig.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    accumulation_steps: int,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Run one full training epoch with gradient accumulation.

    Returns
    -------
    float
        Mean training loss over all samples in *loader*.
    """
    model.train()
    total_loss, total = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x_scalar, x_local, x_pairwise, seq, mask, y) in tqdm(
        enumerate(loader), total=len(loader)
    ):
        x_scalar = x_scalar.to(device)
        x_local = x_local.to(device)
        x_pairwise = x_pairwise.to(device)
        tokens = seq.long().to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(tokens, x_scalar, x_local, x_pairwise, mask)
        logits = logits.squeeze(-1)  # [batch, length]

        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits at batch {batch_idx}.")

        loss_raw = criterion(logits, y.float())
        loss = (loss_raw * mask).sum() / mask.sum() / accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}.")

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at batch {batch_idx}.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Undo the accumulation scaling to track the true loss magnitude
        total_loss += loss.item() * accumulation_steps * y.size(0)
        total += y.size(0)

    return total_loss / max(total, 1)
