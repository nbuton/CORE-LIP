"""
CORE-LIP — Step 4: Make predictions
=====================================
Thin CLI wrapper around core_lip.predictor.

Usage
-----
    python scripts/predict.py \
        --model      data/models/core_lip.pt \
        --h5         data/protein_MD_properties.h5 \
        --datasets   data/CLIP_dataset/TE440_reduced.txt \
                     data/CLIP_dataset/TR1000_reduced.txt \
        --output_dir data/predictions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch

from core_lip.predictor import load_checkpoint, predict_dataset


def main() -> None:
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
    model, checkpoint = load_checkpoint(args.model, device)
    print(f"Loaded checkpoint: {args.model}")

    with h5py.File(args.h5, "r") as h5_features:
        for dataset_path in args.datasets:
            stem = Path(dataset_path).stem
            output_filepath = str(Path(args.output_dir) / f"core_lip_{stem}.csv")
            print(f"\nRunning inference on: {dataset_path}")
            predict_dataset(
                dataset_path=dataset_path,
                h5_features=h5_features,
                model=model,
                checkpoint=checkpoint,
                output_filepath=output_filepath,
                device=device,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
