"""
CORE-LIP — Step 3: Train the network
=====================================
Trains ProteinMultiScaleTransformer on the max_1024 CLIP dataset and saves the
best checkpoint (by validation loss).

Usage
-----
    python scripts/train.py \
        --dataset  data/CLIP_dataset/TR1000_max_1024.txt \
        --h5       data/protein_MD_properties.h5 \
        --model    data/models/core_lip.pt \
        --epochs   250 \
        --device   cpu
"""

from __future__ import annotations

import argparse
import math
import os
from matplotlib import pyplot as plt
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core_lip import (
    ProteinDataset,
    ProteinMultiScaleTransformer,
    collate_proteins,
    prepare_data,
    read_protein_data,
)

from core_lip.features import LOCAL_FEATURES, PAIRWISE_FEATURES, SCALAR_FEATURES
from core_lip.loss import FocalLoss
from core_lip.metrics import analyze_scalar_list, evaluate, select_threshold_cv
from core_lip.trainer import get_config, set_seed, train_one_epoch
from core_lip.utils import cluster_sequences_mmseqs2, get_all_feature_stats


def main():
    parser = argparse.ArgumentParser(description="Train CORE-LIP.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml",
        default="data/models/CORE_LIP_STARLING/config.yaml",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # ── Define the saving path for the model ───────────────────────────────────
    config_dir = os.path.dirname(os.path.abspath(args.config))
    model_save_path = os.path.join(config_dir, "core_lip.pt")

    # ── Load YAML (Now returns a FullConfig object) ───────────────────────────
    cfg = get_config(args.config)
    train_cfg = cfg.training
    model_cfg = cfg.model

    # ── Reproducibility (Changed to dot notation) ──────────────────────────────
    set_seed(train_cfg.seed)
    device = torch.device(args.device)

    print(f"Config:  {args.config}")
    print(f"Device:  {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    with h5py.File(train_cfg.h5_properties, "r") as h5:
        df = read_protein_data(train_cfg.training_dataset)
        X_scalar, X_local, X_pairwise, seqs, y_list, ids = prepare_data(
            df, h5, SCALAR_FEATURES, LOCAL_FEATURES, PAIRWISE_FEATURES
        )
    print(analyze_scalar_list(X_scalar, SCALAR_FEATURES))
    dataset = ProteinDataset(X_scalar, X_local, X_pairwise, seqs, y_list)

    # ── OOD Validation split via MMseqs2 clustering ────────────────────────────
    seq_df = pd.DataFrame({"id": ids, "sequence": seqs})
    cluster_df = cluster_sequences_mmseqs2(
        seq_df, output_file="data/TR1000_cluster.csv"
    )

    all_clusters = cluster_df["cluster"].unique()
    rng = np.random.default_rng(train_cfg.seed)
    rng.shuffle(all_clusters)

    n_val_clusters = max(1, int(0.2 * len(all_clusters)))
    val_clusters = set(all_clusters[:n_val_clusters])
    train_clusters = set(all_clusters[n_val_clusters:])

    val_ids = set(cluster_df[cluster_df["cluster"].isin(val_clusters)]["id"])
    train_ids = set(cluster_df[cluster_df["cluster"].isin(train_clusters)]["id"])

    print(
        f"[split] {n_val_clusters}/{len(all_clusters)} clusters used for val "
        f"→ {len(val_ids)} proteins ({len(val_ids)/len(ids)*100:.1f}%)"
    )
    print(f"[split] Train proteins: {len(train_ids)}")

    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    train_indices = [id_to_idx[pid] for pid in train_ids if pid in id_to_idx]
    val_indices = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # ── Loaders (Changed to dot notation) ──────────────────────────────────────
    train_loader = DataLoader(
        train_subset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,  # 0
        # persistent_workers=True,
        pin_memory=False,
        collate_fn=collate_proteins,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,  # 0
        # persistent_workers=True,
        pin_memory=False,
        collate_fn=collate_proteins,
    )

    # ── Model ──────────────────────────────────────────────────────────────────

    # ── Dynamic Config Update ──────────────────────────────────────────────────
    # Update the Pydantic object field directly
    model_cfg.num_classes = 1  # proba of LIP class only
    model_cfg.nb_scalar = len(SCALAR_FEATURES)
    model_cfg.nb_local = len(LOCAL_FEATURES)
    model_cfg.nb_pairwise = len(PAIRWISE_FEATURES)

    # Pass the validated object to the model
    stats = get_all_feature_stats(X_scalar, X_local, X_pairwise)
    model = ProteinMultiScaleTransformer(model_cfg, stats).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f} M")

    total_pos = sum(y.sum() for y in y_list)
    total_neg = sum((1 - np.array(y)).sum() for y in y_list)
    pos_weight_value = total_neg / total_pos
    print(f"pos_weight: {pos_weight_value:.1f}  (pos={total_pos}, neg={total_neg})")

    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    criterion = (
        FocalLoss(
            gamma=train_cfg.focal_gamma, alpha=train_cfg.focal_alpha, reduction="none"
        )
        if train_cfg.use_focal_loss
        else nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    )
    # Changed to dot notation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        epochs=train_cfg.epochs,
        steps_per_epoch=math.ceil(len(train_loader) / train_cfg.accumulation),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    best_val_auc = float("-inf")
    all_train_loss = []
    all_validation_loss = []
    all_val_auc = []
    # ── Training Loop (Changed to dot notation) ────────────────────────────────
    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            train_cfg.accumulation,
            device,
        )
        all_train_loss.append(train_loss)
        val_loss, val_roc_auc, val_pr_auc = evaluate(
            model, val_loader, criterion, device
        )
        all_validation_loss.append(val_loss)
        all_val_auc.append(val_roc_auc)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_roc_AUC={val_roc_auc:.4f} | val_pr_AUC={val_pr_auc:.4f}"
        )

        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": model_cfg,
                    "stats": stats,
                    "scalar_features": SCALAR_FEATURES,
                    "local_features": LOCAL_FEATURES,
                    "pairwise_features": PAIRWISE_FEATURES,
                    "best_val_auc": best_val_auc,
                },
                model_save_path,
            )
            print(f"  ✓ Checkpoint saved → {model_save_path}")
        else:
            print(f"  x Validation AUC doesn't increase so we didn't save the model")

    # ── Threshold selection ───────────────────────────────────────────────────
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Changed to dot notation
    best_thr = select_threshold_cv(model, train_loader, device, seed=train_cfg.seed)
    print(f"Best threshold (CV-MCC): {best_thr:.6f}")

    checkpoint["best_threshold"] = best_thr
    torch.save(checkpoint, model_save_path)
    print(f"  ✓ Checkpoint updated with threshold → {model_save_path}")

    print(f"\nTraining complete. Best val auc: {best_val_auc:.6f}")

    # Show the loss and metric plot
    plt.plot(all_train_loss, label="Training loss")
    plt.plot(all_validation_loss, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss evolution during training")
    plt.legend()
    plt.show()

    plt.plot(all_val_auc)
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC")
    plt.title("Validation auc evolution during training")
    plt.show()


if __name__ == "__main__":
    main()
