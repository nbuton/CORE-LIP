"""
core_lip/predictor.py
---------------------
Reusable inference logic.

    load_checkpoint   : load a .pt file and reconstruct the model
    predict_dataset   : run per-residue inference on one dataset file
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core_lip.data.datasets import ProteinDataset, collate_proteins
from core_lip.data.io import prepare_data, read_protein_data
from core_lip.modeling.protein_multi_scale_transformer import (
    ProteinMultiScaleTransformer,
)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    model_path: str,
    device: torch.device,
) -> tuple[ProteinMultiScaleTransformer, dict]:
    """
    Load a CORE-LIP checkpoint and return a ready-to-use model + the raw dict.

    Parameters
    ----------
    model_path : path to the .pt checkpoint produced by train.py
    device     : target torch device

    Returns
    -------
    model      : ProteinMultiScaleTransformer in eval mode, moved to *device*
    checkpoint : raw dict (contains cfg, stats, feature lists, best_threshold…)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ProteinMultiScaleTransformer(checkpoint["cfg"], checkpoint["stats"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_dataset(
    dataset_path: str,
    h5_features: h5py.File,
    model: ProteinMultiScaleTransformer,
    checkpoint: dict,
    output_filepath: str,
    device: torch.device,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Run per-residue inference on *dataset_path* and write a CSV to
    *output_filepath*.

    Parameters
    ----------
    dataset_path     : path to a CLIP-format .txt dataset file
    h5_features      : open h5py.File containing MD features
    model            : model returned by :func:`load_checkpoint`
    checkpoint       : dict returned by :func:`load_checkpoint`
    output_filepath  : where to write the output CSV
    device           : torch device
    batch_size       : inference batch size

    Returns
    -------
    pd.DataFrame with columns:
        protein_id, length, predictions, binary_predictions
    """
    df = read_protein_data(dataset_path)
    X_scalar, X_local, X_pairwise, seqs, _labels, ids = prepare_data(
        df,
        h5_features,
        checkpoint["scalar_features"],
        checkpoint["local_features"],
        checkpoint["pairwise_features"],
    )

    dataset = ProteinDataset(X_scalar, X_local, X_pairwise, seqs, ids=ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_proteins,
    )

    if "best_threshold" not in checkpoint.keys():
        best_thr: float = float(
            input(
                "The best_threshold was not found in the checkpoint. Please enter a value (e.g., 0.5): "
            )
        )
    else:
        best_thr: float = checkpoint["best_threshold"]
    rows: list[dict] = []

    for (
        x_scalar_pad,
        x_local_pad,
        x_pairwise_pad,
        seq_pad,
        mask,
        protein_ids,
        plm_pad,
    ) in loader:
        x_scalar_pad = x_scalar_pad.to(device)
        x_local_pad = x_local_pad.to(device)
        x_pairwise_pad = x_pairwise_pad.to(device)
        tokens = seq_pad.long().to(device)
        mask = mask.to(device)
        if plm_pad is not None:
            plm_pad = plm_pad.to(device)

        logits = model(tokens, x_scalar_pad, x_local_pad, x_pairwise_pad, mask, plm_pad)

        # Normalise shape: [B, L, 1] → [B, L]
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if logits.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Expected residue-level output [B, L], got {tuple(logits.shape)}"
            )

        probs = torch.sigmoid(logits).cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        for i, prot_id in enumerate(protein_ids):
            valid_probs = probs[i][mask_np[i]]
            binary = (valid_probs >= best_thr).astype(int)
            rows.append(
                {
                    "protein_id": prot_id,
                    "length": len(valid_probs),
                    "predictions": ",".join(f"{p:.6f}" for p in valid_probs.tolist()),
                    "binary_predictions": ",".join(map(str, binary.tolist())),
                }
            )

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        rows, columns=["protein_id", "length", "predictions", "binary_predictions"]
    )
    df_out.to_csv(output_filepath, index=False)
    print(f"Saved {len(df_out)} proteins → {output_filepath}")
    return df_out
