"""
core_lip/metrics.py
-------------------
Model evaluation and diagnostic utilities.

    - evaluate              : loss + ROC-AUC + PR-AUC on a DataLoader
    - select_threshold_cv   : CV-MCC threshold search on training data
    - analyze_scalar_list   : distribution statistics for scalar features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate *model* on *loader*.

    Returns
    -------
    avg_loss : float
    roc_auc  : float  (NaN when only one class is present)
    pr_auc   : float  (NaN when only one class is present)
    """
    model.eval()
    total_loss, total_batch = 0.0, 0
    y_true_all, y_score_all = [], []

    for x_scalar, x_local, x_pairwise, seq, mask, y, plm_pad in loader:
        x_scalar, x_local, x_pairwise = (
            x_scalar.to(device),
            x_local.to(device),
            x_pairwise.to(device),
        )
        tokens = seq.long().to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(tokens, x_scalar, x_local, x_pairwise, mask, plm_pad)
        logits = logits.squeeze(-1)  # [batch, length]

        loss_raw = criterion(logits, y.float())
        loss = (loss_raw * mask).sum() / mask.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")

        total_loss += loss.item() * y.size(0)
        total_batch += y.size(0)

        # Collect valid (non-padded) residues only
        y_true_all.append(y[mask == 1].cpu())
        y_score_all.append(logits[mask == 1].cpu())

    avg_loss = total_loss / max(total_batch, 1)

    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()
    y_prob = torch.sigmoid(torch.from_numpy(y_score)).numpy()

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    except (ValueError, RuntimeError):
        roc_auc = float("nan")
        pr_auc = float("nan")

    return avg_loss, roc_auc, pr_auc


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def select_threshold_cv(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int = 0,
    n_splits: int = 5,
) -> float:
    """
    Collect residue-level positive-class probabilities over *loader*, then
    return the decision threshold that maximises mean MCC in n_splits-fold
    cross-validation.

    Parameters
    ----------
    model    : trained model (will be set to eval mode)
    loader   : DataLoader (typically the training set)
    device   : torch device
    seed     : random seed for KFold
    n_splits : number of CV folds

    Returns
    -------
    float
        Best probability threshold in [0, 1].
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for x_scalar, x_local, x_pairwise, seq, mask, y, plm_pad in tqdm(
            loader, desc="Collecting scores"
        ):
            x_scalar, x_local, x_pairwise = (
                x_scalar.to(device),
                x_local.to(device),
                x_pairwise.to(device),
            )
            tokens = seq.long().to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(tokens, x_scalar, x_local, x_pairwise, mask, plm_pad)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            for i in range(logits.size(0)):
                m = mask[i].bool()
                all_scores.append(probs[i][m].cpu().numpy())
                all_labels.append(y[i][m].cpu().numpy())

    protein_ids = np.arange(len(all_scores))
    n_splits_eff = min(n_splits, len(protein_ids))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

    flat_scores = np.concatenate(all_scores)
    candidates = np.sort(np.unique(flat_scores))
    if len(candidates) > 1000:
        candidates = np.linspace(flat_scores.min(), flat_scores.max(), 200)

    mean_mccs: list[float] = []
    for thr in tqdm(candidates, desc="Optimising threshold"):
        fold_mccs = []
        for _, val_idx in kf.split(protein_ids):
            y_true = np.concatenate([all_labels[i] for i in val_idx])
            y_pred = (np.concatenate([all_scores[i] for i in val_idx]) > thr).astype(
                np.int8
            )
            fold_mccs.append(matthews_corrcoef(y_true, y_pred))
        mean_mccs.append(float(np.mean(fold_mccs)))

    return float(candidates[int(np.argmax(mean_mccs))])


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def analyze_scalar_list(
    x_list: list[np.ndarray],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Convert a list of 1-D per-protein arrays into a summary statistics
    DataFrame (min, max, mean, std, range, CV).

    Useful for spotting features that might cause gradient explosion during
    training.

    Returns
    -------
    pd.DataFrame
        One row per feature, sorted by range descending.
    """
    data_matrix = np.stack(x_list)  # [N_proteins, N_features]
    df = pd.DataFrame(data_matrix, columns=feature_names)
    stats = df.agg(["min", "max", "mean", "std"]).transpose()
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / (stats["mean"].abs() + 1e-9)
    stats = stats.sort_values("range", ascending=False)

    print(
        f"Analyzed {data_matrix.shape[0]} proteins "
        f"with {data_matrix.shape[1]} features each."
    )
    return stats
