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
from core_lip.utils import get_all_feature_stats
from train import (
    LOCAL_FEATURES,
    PAIRWISE_FEATURES,
    SCALAR_FEATURES,
    analyze_scalar_list,
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

    best_thr = checkpoint["best_threshold"]

    model.eval()
    rows = []

    for x_scalar_pad, x_local_pad, x_pairwise_pad, seq_pad, mask, protein_ids in loader:
        x_scalar_pad = x_scalar_pad.to(device)
        x_local_pad = x_local_pad.to(device)
        x_pairwise_pad = x_pairwise_pad.to(device)
        tokens = seq_pad.long().to(device)
        mask = mask.to(device)

        logits = model(tokens, x_scalar_pad, x_local_pad, x_pairwise_pad, mask)

        # Squeeze trailing singleton dim if present: [B, L, 1] -> [B, L]
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
                    "predictions": ",".join(map(str, valid_probs.tolist())),
                    "binary_predictions": ",".join(map(str, binary.tolist())),
                }
            )

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
    parser.add_argument("--model", default="data/models/core_lip.pt")
    parser.add_argument("--h5", default="data/protein_MD_properties.h5")
    parser.add_argument(
        "--datasets", nargs="+", default=["data/CLIP_dataset/TE440_reduced.txt"]
    )
    parser.add_argument("--output_dir", default="data/predictions/")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = ProteinMultiScaleTransformer(checkpoint["cfg"], checkpoint["stats"]).to(
        device
    )
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
