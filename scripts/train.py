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
import os
from matplotlib import pyplot as plt
import yaml
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from core_lip import (
    ProteinDataset,
    ProteinMultiScaleTransformer,
    collate_proteins,
    prepare_data,
    read_protein_data,
)
from core_lip.config import FullConfig

# ---------------------------------------------------------------------------
# Feature configuration (edit here to change the feature set)
# ---------------------------------------------------------------------------

SCALAR_FEATURES = [
    "asphericity_mean",
    "asphericity_std",
    "avg_maximum_diameter",
    "avg_squared_Ree",
    "gyration_eigenvalues_l1_mean",
    "gyration_eigenvalues_l1_std",
    "gyration_eigenvalues_l2_mean",
    "gyration_eigenvalues_l2_std",
    "gyration_eigenvalues_l3_mean",
    "gyration_eigenvalues_l3_std",
    "gyration_l1_per_l2_mean",
    "gyration_l1_per_l2_std",
    "gyration_l1_per_l3_mean",
    "gyration_l1_per_l3_std",
    "gyration_l2_per_l3_mean",
    "gyration_l2_per_l3_std",
    "normalized_acylindricity_mean",
    "normalized_acylindricity_std",
    "prolateness_mean",
    "prolateness_std",
    "radius_of_gyration_mean",
    "radius_of_gyration_std",
    "rel_shape_anisotropy_mean",
    "rel_shape_anisotropy_std",
    "scaling_exponent",
    "std_maximum_diameter",
    "std_squared_Ree",
]

LOCAL_FEATURES = [
    "phi_entropy",
    "psi_entropy",
    "sasa_abs_mean",
    "sasa_abs_std",
    "sasa_rel_mean",
    "sasa_rel_std",
    "ss_propensity_B",
    "ss_propensity_C",
    "ss_propensity_E",
    "ss_propensity_G",
    "ss_propensity_H",
    "ss_propensity_I",
    "ss_propensity_S",
    "ss_propensity_T",
]

PAIRWISE_FEATURES = ["dccm", "contact_map", "distance_fluctuations"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    # 1. Use BCEWithLogitsLoss with reduction='none' for masking
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_loss, total_batch = 0.0, 0
    y_true_all, y_score_all = [], []

    for x_scalar, x_local, x_pairwise, seq, mask, y in loader:
        x_scalar, x_local, x_pairwise = (
            x_scalar.to(device),
            x_local.to(device),
            x_pairwise.to(device),
        )
        tokens, mask, y = seq.long().to(device), mask.to(device), y.to(device)

        logits = model(tokens, x_scalar, x_local, x_pairwise, mask)
        logits = logits.squeeze(-1)  # Shape: [batch, length]

        # 2. Loss Calculation with masking
        loss_raw = criterion(logits, y.float())
        loss = (loss_raw * mask).sum() / mask.sum()

        total_loss += loss.item() * y.size(0)
        total_batch += y.size(0)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")

        # 5. Store values for ROC AUC (filtering out padding)
        # We only take the values where mask == 1
        y_true_all.append(y[mask == 1].cpu())
        y_score_all.append(logits[mask == 1].cpu())

    # Final Metrics
    avg_loss = total_loss / max(total_batch, 1)

    # 6. ROC AUC Calculation
    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()

    try:
        # For binary, y_score is the logit. sigmoid converts it to probability.
        y_prob = torch.sigmoid(torch.from_numpy(y_score)).numpy()
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, RuntimeError):
        roc_auc = float("nan")

    return avg_loss, roc_auc


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    accumulation_step,
    device,
    grad_clip=1.0,
):
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
        logits = logits.squeeze(-1)

        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits at batch {batch_idx}.")

        loss_raw = criterion(logits, y.float())
        loss = (loss_raw * mask).sum() / mask.sum() / accumulation_step
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}.")
        loss.backward()

        if (batch_idx + 1) % accumulation_step == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at batch {batch_idx}.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * y.size(0) * accumulation_step
        total += y.size(0)

    return total_loss / max(total, 1)


def select_threshold_cv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int = 0,
    n_splits: int = 5,
) -> float:
    """
    Collect residue-level positive-class probabilities over *loader*, then
    pick the decision threshold that maximises mean MCC in n_splits-fold CV.
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for x_scalar, x_local, x_pairwise, seq, mask, y in tqdm(loader):
            x_scalar, x_local, x_pairwise = (
                x_scalar.to(device),
                x_local.to(device),
                x_pairwise.to(device),
            )
            tokens, mask, y = seq.long().to(device), mask.to(device), y.to(device)

            logits = model(tokens, x_scalar, x_local, x_pairwise, mask)
            logits = logits.squeeze(-1)  # Ensure shape [batch, length]

            # 1. Correct probability calculation for Binary BCE setup
            probs = torch.sigmoid(logits)

            # 2. Extract valid residues using mask for each protein in batch
            # This is cleaner than safe_probs view logic
            for i in range(logits.size(0)):
                m = mask[i].bool()
                all_scores.append(probs[i][m].cpu().numpy())
                all_labels.append(y[i][m].cpu().numpy())

    protein_ids = np.arange(len(all_scores))
    n_splits_eff = min(n_splits, len(protein_ids))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

    # Concatenate all scores to find unique thresholds
    flat_scores = np.concatenate(all_scores)

    # Optimization: If you have millions of residues, unique thresholds might be too many.
    # We can subsample or use a fixed range if this is too slow.
    candidate_thresholds = np.sort(np.unique(flat_scores))

    # Optional: if unique thresholds are > 1000, consider np.linspace(0, 1, 100)
    if len(candidate_thresholds) > 1000:
        candidate_thresholds = np.linspace(flat_scores.min(), flat_scores.max(), 200)

    mean_mccs = []
    for thr in tqdm(candidate_thresholds, desc="Optimizing Threshold"):
        fold_mccs = []
        for _, val_idx in kf.split(protein_ids):
            # Evaluate threshold on this fold
            y_true = np.concatenate([all_labels[i] for i in val_idx])
            y_score = np.concatenate([all_scores[i] for i in val_idx])

            # Calculate MCC
            mcc = matthews_corrcoef(y_true, (y_score > thr).astype(np.int8))
            fold_mccs.append(mcc)

        mean_mccs.append(np.mean(fold_mccs))

    best_threshold = float(candidate_thresholds[np.argmax(mean_mccs)])
    return best_threshold


def get_config(yaml_path: str) -> FullConfig:
    """Parses YAML and returns a validated FullConfig object."""
    with open(yaml_path, "r") as f:
        # Pydantic handles the nested dict conversion and type casting automatically
        return FullConfig.model_validate(yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    dataset = ProteinDataset(X_scalar, X_local, X_pairwise, seqs, y_list)

    n = len(dataset)
    indices = np.random.permutation(n)
    split = max(1, int(0.9 * n))

    train_subset = torch.utils.data.Subset(dataset, indices[:split].tolist())
    val_subset = torch.utils.data.Subset(
        dataset, indices[split:].tolist() or indices[:1].tolist()
    )

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
    model = ProteinMultiScaleTransformer(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f} M")

    total_pos = sum(y.sum() for y in y_list)
    total_neg = sum((1 - np.array(y)).sum() for y in y_list)
    pos_weight_value = total_neg / total_pos
    print(f"pos_weight: {pos_weight_value:.1f}  (pos={total_pos}, neg={total_neg})")

    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    # Changed to dot notation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weigth_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg.lr,
        epochs=train_cfg.epochs,
        steps_per_epoch=int(len(train_loader) / train_cfg.accumulation),
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
        val_loss, val_auc = evaluate(model, val_loader, device)
        all_validation_loss.append(val_loss)
        all_val_auc.append(val_auc)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_AUC={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": model_cfg,
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
