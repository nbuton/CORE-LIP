"""
CORE-LIP — Step 4: Make predictions
=====================================
Loads a trained CORE-LIP checkpoint and outputs per-residue LIP scores
for one or more datasets.

Usage
-----
    python scripts/predict.py \
        --model      data/models/core_lip.pt \
        --h5         data/protein_MD_properties.h5 \
        --datasets   data/CLIP_dataset/TE440_reduced.txt data/CLIP_dataset/TR1000_reduced.txt \
        --output_dir data/predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core_lip import (
    ProteinInferenceDataset,
    ProteinMultiScaleTransformer,
    collate_inference,
    prepare_data,
    read_protein_data,
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_dataset(
    dataset_path: str,
    h5_features,
    checkpoint: dict,
    model: ProteinMultiScaleTransformer,
    output_filepath: str,
    device: torch.device,
) -> None:
    df = read_protein_data(dataset_path)
    X_scalar, X_local, X_pairwise, seqs, y_list, ids = prepare_data(
        df,
        h5_features,
        checkpoint["scalar_features"],
        checkpoint["local_features"],
        checkpoint["pairwise_features"],
    )

    dataset = ProteinInferenceDataset(X_scalar, X_local, X_pairwise, seqs, ids)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_inference,
    )

    best_thr = checkpoint["best_thr"]

    model.eval()
    rows = []

    for x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, protein_ids in loader:
        x_scalar_pad = x_scalar_pad.to(device)
        x_local_pad = x_local_pad.to(device)
        x_pairwise_pad = x_pairwise_pad.to(device)
        tokens = seq_pad.long().to(device)
        mask = mask.to(device)

        logits = model(tokens, x_scalar_pad, x_local_pad, x_pairwise_pad, mask)
        mask_np = mask.cpu().numpy()

        # Residue-wise model output: [B, L] or [B, L, 1]
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        if logits.dim() == 2 and logits.shape[1] == mask.shape[1]:
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, prot_id in enumerate(protein_ids):
                valid_probs = probs[i][mask_np[i].astype(bool)]
                binary = (valid_probs >= best_thr).astype(int)
                rows.append(
                    {
                        "protein_id": prot_id,
                        "length": int(len(valid_probs)),
                        "predictions": ",".join(map(str, valid_probs.tolist())),
                        "binary_predictions": ",".join(map(str, binary.tolist())),
                    }
                )

        # Protein-level classifier output: [B, C] — broadcast score to residues
        elif logits.dim() == 2:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for i, prot_id in enumerate(protein_ids):
                seq_len = int(mask_np[i].sum())
                score = (
                    float(probs[i, 1])
                    if probs.shape[1] == 2
                    else float(np.max(probs[i]))
                )
                binary_score = int(score >= best_thr)
                rows.append(
                    {
                        "protein_id": prot_id,
                        "length": seq_len,
                        "predictions": ",".join([str(score)] * seq_len),
                        "binary_predictions": ",".join([str(binary_score)] * seq_len),
                    }
                )

        else:
            raise ValueError(f"Unsupported model output shape: {tuple(logits.shape)}")

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        rows, columns=["protein_id", "length", "predictions", "binary_predictions"]
    )
    df_out.to_csv(output_filepath, index=False)
    print(f"Saved {len(df_out)} proteins → {output_filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run CORE-LIP predictions.")
    parser.add_argument(
        "--model",
        default="data/models/core_lip.pt",
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--h5",
        default="data/protein_MD_properties.h5",
        help="HDF5 file of precomputed MD properties.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/CLIP_dataset/TE440_reduced.txt"],
        help="One or more dataset files to run inference on.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/predictions/",
        help="Directory to write prediction CSV files.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    model = ProteinMultiScaleTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.model}")

    h5_features = h5py.File(args.h5, "r")

    for dataset_path in args.datasets:
        stem = Path(dataset_path).stem
        output_filepath = str(Path(args.output_dir) / f"core_lip_{stem}.csv")
        print(f"\nRunning inference on: {dataset_path}")
        predict_dataset(
            dataset_path=dataset_path,
            h5_features=h5_features,
            checkpoint=checkpoint,
            model=model,
            output_filepath=output_filepath,
            device=device,
        )

    h5_features.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
