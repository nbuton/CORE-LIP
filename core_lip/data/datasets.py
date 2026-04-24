"""
core_lip/datasets.py
====================
Dataset classes and collation utilities for training, validation, and
inference.

Training / validation
---------------------
    dataset = ProteinDataset(..., labels=y_list, ids=ids, plm_h5_path="...")
    loader  = DataLoader(dataset, collate_fn=collate_proteins)
    # → x_scalar, x_local, x_pairwise, seq, mask, plm_pad, y_pad

Inference
---------
    dataset = ProteinDataset(..., ids=ids, plm_h5_path="...")
    loader  = DataLoader(dataset, collate_fn=collate_proteins)
    # → x_scalar, x_local, x_pairwise, seq, mask, plm_pad, protein_ids
"""

from __future__ import annotations

from typing import Optional
import h5py

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
    """

    def __init__(
        self,
        X_scalar_list: list[np.ndarray],
        X_local_list: list[np.ndarray],
        X_pairwise_list: list[np.ndarray],
        seq_enc_list: list[np.ndarray],
        labels: Optional[list[np.ndarray]] = None,
        ids: Optional[list[str]] = None,
        plm_h5_path: Optional[str] = None,  # New option
    ) -> None:
        if labels is None and ids is None:
            raise ValueError(
                "Provide at least `labels` (training) or `ids` (inference)."
            )
        # IDs are required for PLM lookup even in training mode
        if plm_h5_path is not None and ids is None:
            raise ValueError("`ids` are required to fetch PLM embeddings from H5.")

        self.X_scalar_list = X_scalar_list
        self.X_local_list = X_local_list
        self.X_pairwise_list = X_pairwise_list
        self.seq_enc_list = seq_enc_list
        self.labels = labels
        self.ids = ids
        self.plm_h5_path = plm_h5_path

    def __len__(self) -> int:
        return len(self.X_scalar_list)

    def __getitem__(self, idx: int):
        x_s = np.nan_to_num(self.X_scalar_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_l = np.nan_to_num(self.X_local_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_p = np.nan_to_num(self.X_pairwise_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        seq = self.seq_enc_list[idx]

        # 1. Fetch PLM if path is provided
        plm_val = None
        if self.plm_h5_path:
            with h5py.File(self.plm_h5_path, "r") as f:
                plm_val = torch.from_numpy(f[self.ids[idx]][:]).float()

        features = (
            torch.from_numpy(x_s).float(),
            torch.from_numpy(x_l).float(),
            torch.from_numpy(x_p).float(),
            torch.from_numpy(seq).long(),
            plm_val,
        )

        if self.labels is not None:
            y = torch.from_numpy(np.asarray(self.labels[idx])).long()
            return features, y
        else:
            return features, self.ids[idx]


def collate_proteins(batch: list) -> tuple:
    inputs, targets = zip(*batch)
    xs_scalar, xs_local, xs_pairwise, seqs, plms = zip(*inputs)

    is_inference = isinstance(targets[0], str)
    B = len(batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())

    # Standard pads
    x_scalar_pad = torch.zeros((B, xs_scalar[0].shape[0]), dtype=torch.float32)
    x_local_pad = torch.zeros((B, xs_local[0].shape[0], max_len), dtype=torch.float32)
    x_pairwise_pad = torch.zeros(
        (B, xs_pairwise[0].shape[0], max_len, max_len), dtype=torch.float32
    )
    seq_pad = torch.zeros((B, max_len), dtype=torch.long)
    y_pad = torch.full((B, max_len), fill_value=-100, dtype=torch.long)

    # PLM pad setup
    plm_pad = None
    if plms[0] is not None:
        plm_pad = torch.zeros((B, max_len, plms[0].shape[-1]), dtype=torch.float32)

    for i in range(B):
        L = int(lengths[i])
        x_scalar_pad[i] = xs_scalar[i]
        x_local_pad[i, :, :L] = xs_local[i]
        x_pairwise_pad[i, :, :L, :L] = xs_pairwise[i]
        seq_pad[i, :L] = seqs[i]
        if plm_pad is not None:
            plm_pad[i, :L, :] = plms[i]
        if not is_inference:
            y_pad[i, :L] = targets[i]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    # Returns 1-5 (common), 6 (target/id), 7 (plm)
    if is_inference:
        return (
            x_scalar_pad,
            x_local_pad,
            x_pairwise_pad,
            seq_pad,
            mask,
            list(targets),
            plm_pad,
        )
    else:
        return x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, y_pad, plm_pad
