"""
core_lip/datasets.py
====================
Dataset classes and collation utilities for training, validation, and
inference.

Training / validation
---------------------
    dataset = ProteinDataset(..., labels=y_list)
    loader  = DataLoader(dataset, collate_fn=collate_proteins)
    # → x_scalar, x_local, x_pairwise, seq, mask, y_pad  (LongTensor)

Inference
---------
    dataset = ProteinDataset(..., ids=ids)
    loader  = DataLoader(dataset, collate_fn=collate_proteins)
    # → x_scalar, x_local, x_pairwise, seq, mask, protein_ids  (list[str])
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Amino-acid vocabulary (0 reserved for padding)
# ---------------------------------------------------------------------------

AA_TO_INT: dict[str, int] = {
    aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX", start=1)
}


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------


class ProteinDataset(Dataset):
    """
    Unified dataset for supervised training *and* inference.

    Pass exactly one of *labels* or *ids*:

    - ``labels`` (list of per-residue int arrays) → training / validation mode;
      ``__getitem__`` returns ``(features, y_tensor)``.
    - ``ids`` (list of str) → inference mode;
      ``__getitem__`` returns ``(features, protein_id_str)``.

    Parameters
    ----------
    X_scalar_list   : list of (nb_scalar,)        float arrays
    X_local_list    : list of (nb_local, L)        float arrays
    X_pairwise_list : list of (nb_pairwise, L, L)  float arrays
    seq_enc_list    : list of (L,) int arrays — encoded amino-acid sequences
    labels          : list of (L,) int arrays — per-residue LIP labels, or None
    ids             : list of str — protein identifiers, or None
    """

    def __init__(
        self,
        X_scalar_list: list[np.ndarray],
        X_local_list: list[np.ndarray],
        X_pairwise_list: list[np.ndarray],
        seq_enc_list: list[np.ndarray],
        labels: Optional[list[np.ndarray]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        if labels is None and ids is None:
            raise ValueError("Provide either `labels` (training) or `ids` (inference).")
        if labels is not None and ids is not None:
            raise ValueError(
                "Provide either `labels` (training) or `ids` (inference), not both."
            )

        self.X_scalar_list = X_scalar_list
        self.X_local_list = X_local_list
        self.X_pairwise_list = X_pairwise_list
        self.seq_enc_list = seq_enc_list
        self.labels = labels  # None  →  inference mode
        self.ids = ids  # None  →  training mode

    def __len__(self) -> int:
        return len(self.X_scalar_list)

    def __getitem__(self, idx: int):
        # Sanitise NaN / ±Inf that can come from MD feature extraction
        x_s = np.nan_to_num(self.X_scalar_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_l = np.nan_to_num(self.X_local_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_p = np.nan_to_num(self.X_pairwise_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        seq = self.seq_enc_list[idx]

        features = (
            torch.from_numpy(x_s).float(),
            torch.from_numpy(x_l).float(),
            torch.from_numpy(x_p).float(),
            torch.from_numpy(seq).long(),
        )

        if self.labels is not None:
            y = torch.from_numpy(np.asarray(self.labels[idx])).long()
            return features, y
        else:
            return features, self.ids[idx]  # str


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------


def ProteinInferenceDataset(
    X_scalar_list,
    X_local_list,
    X_pairwise_list,
    seq_enc_list,
    list_ids,
) -> ProteinDataset:
    """
    Deprecated — use ``ProteinDataset(..., ids=list_ids)`` directly.
    Kept so existing call sites continue to work without modification.
    """
    return ProteinDataset(
        X_scalar_list,
        X_local_list,
        X_pairwise_list,
        seq_enc_list,
        ids=list_ids,
    )


# ---------------------------------------------------------------------------
# Unified collate
# ---------------------------------------------------------------------------


def collate_proteins(batch: list) -> tuple:
    """
    Unified collate function for both training and inference batches.

    The type of the second element in each sample drives dispatch:

    - ``torch.Tensor``  → training / validation mode
    - ``str``           → inference mode

    Returns (training)
    ------------------
    x_scalar_pad   : [B, nb_scalar]
    x_local_pad    : [B, nb_local,    max_len]
    x_pairwise_pad : [B, nb_pairwise, max_len, max_len]
    seq_pad        : [B, max_len]
    mask           : [B, max_len]  — True for real residues, False for padding
    y_pad          : [B, max_len]  — per-residue labels; -100 at padding positions

    Returns (inference)
    -------------------
    x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask  — same as above
    protein_ids    : list[str]
    """
    inputs, targets = zip(*batch)
    xs_scalar, xs_local, xs_pairwise, seqs = zip(*inputs)

    is_inference = isinstance(targets[0], str)

    B = len(batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())

    nb_scalar = xs_scalar[0].shape[0]
    nb_local = xs_local[0].shape[0]
    nb_pairwise = xs_pairwise[0].shape[0]

    x_scalar_pad = torch.zeros((B, nb_scalar), dtype=torch.float32)
    x_local_pad = torch.zeros((B, nb_local, max_len), dtype=torch.float32)
    x_pairwise_pad = torch.zeros(
        (B, nb_pairwise, max_len, max_len), dtype=torch.float32
    )
    seq_pad = torch.zeros((B, max_len), dtype=torch.long)
    # -100 is ignored by CrossEntropyLoss; for BCE we mask explicitly via `mask`
    y_pad = torch.full((B, max_len), fill_value=-100, dtype=torch.long)

    for i in range(B):
        L = int(lengths[i])
        x_scalar_pad[i] = xs_scalar[i]
        x_local_pad[i, :, :L] = xs_local[i]
        x_pairwise_pad[i, :, :L, :L] = xs_pairwise[i]
        seq_pad[i, :L] = seqs[i]
        if not is_inference:
            y_pad[i, :L] = targets[i]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)  # [B, max_len]

    if is_inference:
        return x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, list(targets)
    else:
        return x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, y_pad
