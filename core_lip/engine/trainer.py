"""
core_lip/trainer.py
-------------------
Low-level training primitives and small helpers.

    - set_seed        : global reproducibility
    - get_config      : YAML → FullConfig
    - train_one_epoch : single epoch with gradient accumulation and clipping
"""

from __future__ import annotations

import math
import os
import random

import h5py
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Subset

from core_lip.config import FullConfig
from core_lip.data.datasets import ProteinDataset, collate_proteins
from core_lip.data.features import LOCAL_FEATURES, PAIRWISE_FEATURES, SCALAR_FEATURES
from core_lip.data.io import (
    cluster_sequences_mmseqs2,
    get_all_feature_stats,
    prepare_data,
    read_protein_data,
)
from core_lip.eval.metrics import evaluate, select_threshold_cv
from core_lip.modeling.loss import AUCMarginLoss, FocalLoss, LDAMLoss
from core_lip.modeling.protein_multi_scale_transformer import (
    ProteinMultiScaleTransformer,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def get_config(yaml_path: str) -> FullConfig:
    """Parse a YAML config file and return a validated :class:`FullConfig`."""
    with open(yaml_path, "r") as fh:
        return FullConfig.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class CORE_LIP_Trainer:
    def __init__(self, cfg, config_path, device="cpu"):
        self.cfg = cfg
        self.train_cfg = cfg.training
        self.model_cfg = cfg.model
        self.device = torch.device(device)

        # Paths
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.model_save_path = os.path.join(self.config_dir, "core_lip.pt")

        set_seed(self.train_cfg.seed)

        # Placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.stats = None
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def prepare_loaders(self):
        """Handles data loading and OOD splitting logic."""
        with h5py.File(self.train_cfg.h5_properties, "r") as h5:
            df = read_protein_data(self.train_cfg.training_dataset)
            X_scalar, X_local, X_pairwise, seqs, y_list, ids = prepare_data(
                df, h5, SCALAR_FEATURES, LOCAL_FEATURES, PAIRWISE_FEATURES
            )

        self.dataset = ProteinDataset(X_scalar, X_local, X_pairwise, seqs, y_list)
        self.stats = get_all_feature_stats(X_scalar, X_local, X_pairwise)
        self.y_list = y_list  # Keep for loss weight calculation

        # Handle split
        val_prop = self.train_cfg.val_prop

        if val_prop <= 0:
            print("[split] val_prop is 0. Using full dataset for training.")
            train_indices = list(range(len(ids)))
            val_indices = []
        else:
            seq_df = pd.DataFrame({"id": ids, "sequence": seqs})
            cluster_df = cluster_sequences_mmseqs2(
                seq_df, output_file="data/TR1000_cluster.csv"
            )

            all_clusters = cluster_df["cluster"].unique()
            rng = np.random.default_rng(self.train_cfg.seed)
            rng.shuffle(all_clusters)

            n_val_clusters = max(1, int(val_prop * len(all_clusters)))
            val_clusters = set(all_clusters[:n_val_clusters])

            val_ids = set(cluster_df[cluster_df["cluster"].isin(val_clusters)]["id"])
            id_to_idx = {pid: i for i, pid in enumerate(ids)}

            val_indices = [id_to_idx[pid] for pid in val_ids if pid in id_to_idx]
            train_indices = [i for i in range(len(ids)) if i not in set(val_indices)]

            print(
                f"[split] OOD split: {len(train_indices)} train, {len(val_indices)} val proteins."
            )

        loader_kwargs = dict(
            batch_size=self.train_cfg.batch_size,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_proteins,
        )

        self.train_loader = DataLoader(
            Subset(self.dataset, train_indices), shuffle=True, **loader_kwargs
        )
        self.val_loader = (
            DataLoader(
                Subset(self.dataset, val_indices), shuffle=False, **loader_kwargs
            )
            if val_indices
            else None
        )

    def build_model(self):
        self.model_cfg.num_classes = 1
        self.model_cfg.nb_scalar = len(SCALAR_FEATURES)
        self.model_cfg.nb_local = len(LOCAL_FEATURES)
        self.model_cfg.nb_pairwise = len(PAIRWISE_FEATURES)

        self.model = ProteinMultiScaleTransformer(self.model_cfg, self.stats).to(
            self.device
        )
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f} M"
        )

    def build_criterion(self):
        total_pos = sum(y.sum() for y in self.y_list)
        total_neg = sum((1 - np.array(y)).sum() for y in self.y_list)

        loss_type = self.train_cfg.loss_type
        params = self.train_cfg.loss_params

        if loss_type == "focal":
            self.criterion = FocalLoss(reduction="none", **params)
        elif loss_type == "ldam":
            self.criterion = LDAMLoss(
                n_pos=total_pos, n_neg=total_neg, reduction="none", **params
            )
        elif loss_type == "auc_margin":
            self.criterion = AUCMarginLoss(
                n_pos=total_pos, n_neg=total_neg, reduction="none", **params
            )
        elif loss_type == "bce_with_logits":
            pos_weight = torch.tensor(
                [total_neg / total_pos], device=self.device, dtype=torch.float32
            )
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            )
        else:
            raise ValueError(f"Unknown loss: {loss_type}")

        print(
            f"Initialized {loss_type} for imbalanced ranking (pos_ratio: {total_pos/(total_pos+total_neg):.2%})"
        )

    def save_checkpoint(self, auc=None, is_final=False):
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "cfg": self.model_cfg,
            "stats": self.stats,
            "scalar_features": SCALAR_FEATURES,
            "local_features": LOCAL_FEATURES,
            "pairwise_features": PAIRWISE_FEATURES,
            "best_val_auc": auc,
        }
        torch.save(save_dict, self.model_save_path)
        suffix = "(Final)" if is_final else f"(AUC: {auc:.4f})"
        print(f"  ✓ Checkpoint saved {suffix} → {self.model_save_path}")

    def run(self):
        self.prepare_loaders()
        self.build_model()
        self.build_criterion()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.train_cfg.lr,
            epochs=self.train_cfg.epochs,
            steps_per_epoch=math.ceil(
                len(self.train_loader) / self.train_cfg.accumulation
            ),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        best_auc = float("-inf")

        for epoch in range(1, self.train_cfg.epochs + 1):
            t_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.scheduler,
                self.criterion,
                self.train_cfg.accumulation,
                self.device,
            )
            self.history["train_loss"].append(t_loss)

            log_str = f"Epoch {epoch:03d} | train_loss={t_loss:.4f}"

            # Only evaluate if val_loader exists
            if self.val_loader:
                v_loss, v_auc, v_pr = evaluate(
                    self.model, self.val_loader, self.criterion, self.device
                )
                self.history["val_loss"].append(v_loss)
                self.history["val_auc"].append(v_auc)

                # Restoration of ROC-AUC and PR-AUC labels
                log_str += f" | val_loss={v_loss:.4f} | val_ROC-AUC={v_auc:.4f} | val_PR-AUC={v_pr:.4f}"
                print(log_str)

                if v_auc > best_auc:
                    best_auc = v_auc
                    self.save_checkpoint(auc=best_auc)
                else:
                    # Restoration of the "did not improve" notification
                    print(
                        f"  - Validation AUC did not improve, checkpoint not updated."
                    )
            else:
                print(log_str)
                # If no validation, save the model at the last epoch
                if epoch == self.train_cfg.epochs:
                    self.save_checkpoint(is_final=True)

        # Post-training: Threshold selection
        checkpoint = torch.load(
            self.model_save_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        best_thr = select_threshold_cv(
            self.model, self.train_loader, self.device, seed=self.train_cfg.seed
        )
        checkpoint["best_threshold"] = best_thr
        torch.save(checkpoint, self.model_save_path)
        print(f"Final threshold (CV-MCC): {best_thr:.6f}")

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history["train_loss"], label="Train")
        if self.history["val_loss"]:
            ax1.plot(self.history["val_loss"], label="Val")
        ax1.set_title("Loss")
        ax1.legend()

        if self.history["val_auc"]:
            ax2.plot(self.history["val_auc"])
            ax2.set_title("Validation ROC-AUC")

        plt.tight_layout()
        plt.show()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    accumulation_steps: int,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Run one full training epoch with gradient accumulation.

    Returns
    -------
    float
        Mean training loss over all samples in *loader*.
    """
    model.train()
    total_loss, total = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x_scalar, x_local, x_pairwise, seq, mask, y, plm_pad) in tqdm(
        enumerate(loader), total=len(loader)
    ):
        x_scalar = x_scalar.to(device)
        x_local = x_local.to(device)
        x_pairwise = x_pairwise.to(device)
        tokens = seq.long().to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(tokens, x_scalar, x_local, x_pairwise, mask, plm_pad)
        logits = logits.squeeze(-1)  # [batch, length]

        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits at batch {batch_idx}.")

        loss_raw = criterion(logits, y.float())
        loss = (loss_raw * mask).sum() / mask.sum() / accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}.")

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at batch {batch_idx}.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Undo the accumulation scaling to track the true loss magnitude
        total_loss += loss.item() * accumulation_steps * y.size(0)
        total += y.size(0)

    return total_loss / max(total, 1)
