"""
CORE-LIP — Dataset and collation utilities
==========================================
Handles training, validation, and inference datasets for the
ProteinMultiScaleTransformer.
"""

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
# Training / validation dataset
# ---------------------------------------------------------------------------


class ProteinDataset(Dataset):
    """
    Dataset for supervised training.

    Parameters
    ----------
    X_scalar_list   : list of (nb_scalar,)         arrays — global conformational features
    X_local_list    : list of (nb_local, L)         arrays — per-residue features
    X_pairwise_list : list of (nb_pairwise, L, L)   arrays — pairwise features
    seq_enc_list    : list of (L,) int arrays        — encoded amino-acid sequences
    y_list          : list of (L,) int arrays        — per-residue LIP labels
    """

    def __init__(
        self,
        X_scalar_list,
        X_local_list,
        X_pairwise_list,
        seq_enc_list,
        y_list,
    ):
        self.X_scalar_list = X_scalar_list
        self.X_local_list = X_local_list
        self.X_pairwise_list = X_pairwise_list
        self.seq_enc_list = seq_enc_list
        self.y_list = y_list

    def __len__(self) -> int:
        return len(self.X_scalar_list)

    def __getitem__(self, idx: int):
        x_s = np.nan_to_num(self.X_scalar_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_l = np.nan_to_num(self.X_local_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_p = np.nan_to_num(self.X_pairwise_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        seq = self.seq_enc_list[idx]

        inputs = (
            torch.from_numpy(x_s).float(),
            torch.from_numpy(x_l).float(),
            torch.from_numpy(x_p).float(),
            torch.from_numpy(seq).long(),
        )
        y = torch.from_numpy(np.asarray(self.y_list[idx])).long()
        return inputs, y


# ---------------------------------------------------------------------------
# Inference dataset (no labels required)
# ---------------------------------------------------------------------------


class ProteinInferenceDataset(Dataset):
    """
    Dataset for inference / prediction (no labels).

    Parameters
    ----------
    X_scalar_list, X_local_list, X_pairwise_list, seq_enc_list — same as above
    list_ids : list of str — protein identifiers
    """

    def __init__(
        self,
        X_scalar_list,
        X_local_list,
        X_pairwise_list,
        seq_enc_list,
        list_ids,
    ):
        self.X_scalar_list = X_scalar_list
        self.X_local_list = X_local_list
        self.X_pairwise_list = X_pairwise_list
        self.seq_enc_list = seq_enc_list
        self.list_ids = list_ids

    def __len__(self) -> int:
        return len(self.X_scalar_list)

    def __getitem__(self, idx: int):
        x_s = np.nan_to_num(self.X_scalar_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_l = np.nan_to_num(self.X_local_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        x_p = np.nan_to_num(self.X_pairwise_list[idx], nan=0.0, posinf=0.0, neginf=0.0)
        seq = self.seq_enc_list[idx]

        inputs = (
            torch.from_numpy(x_s).float(),
            torch.from_numpy(x_l).float(),
            torch.from_numpy(x_p).float(),
            torch.from_numpy(seq).long(),
        )
        return inputs, self.list_ids[idx]


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_proteins(batch):
    """
    Collate function for ProteinDataset.

    Pads all tensors to the longest sequence in the batch and returns a
    boolean mask (True = real residue, False = padding).

    Padding positions in y are filled with -100 so they are ignored by
    CrossEntropyLoss by default.

    Returns
    -------
    x_scalar_pad   : [B, nb_scalar]
    x_local_pad    : [B, nb_local, max_len]
    x_pairwise_pad : [B, nb_pairwise, max_len, max_len]
    seq_pad        : [B, max_len]
    mask           : [B, max_len]
    y_pad          : [B, max_len]   — per-residue labels, -100 at padding positions
    """
    inputs, ys = zip(*batch)
    xs_scalar, xs_local, xs_pairwise, seqs = zip(*inputs)

    B = len(batch)
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
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
    y_pad = torch.full((B, max_len), fill_value=-100, dtype=torch.long)

    for i in range(B):
        L = int(lengths[i])
        x_scalar_pad[i] = xs_scalar[i]
        x_local_pad[i, :, :L] = xs_local[i]
        x_pairwise_pad[i, :, :L, :L] = xs_pairwise[i]
        seq_pad[i, :L] = seqs[i]
        y_pad[i, :L] = ys[i]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, y_pad


def collate_inference(batch):
    """
    Collate function for ProteinInferenceDataset.

    Same as collate_proteins but returns protein IDs instead of labels.

    Returns
    -------
    x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask : same as above
    protein_ids : list[str]
    """
    inputs, protein_ids = zip(*batch)
    xs_scalar, xs_local, xs_pairwise, seqs = zip(*inputs)

    B = len(batch)
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
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

    for i in range(B):
        L = int(lengths[i])
        x_scalar_pad[i] = xs_scalar[i]
        x_local_pad[i, :, :L] = xs_local[i]
        x_pairwise_pad[i, :, :L, :L] = xs_pairwise[i]
        seq_pad[i, :L] = seqs[i]

    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, list(protein_ids)
