"""
CORE-LIP — Step 5: Evaluate predictions
=========================================
Evaluates one or more models on a held-out test set at the residue level.
All prediction files must follow the CSV format produced by predict.py:
    protein_id, length, predictions, binary_predictions [, extra columns ignored]

Usage
-----
    python scripts/evaluate.py \
        --test_truth  data/CLIP_dataset/TE440_max_1024.txt \
        --pred_files  data/predictions/core_lip_TE440.csv \
                      data/predictions/clip_TE440.csv \
                      data/predictions/idplip_TE440.csv \
        --names       CORE-LIP CLIP IDP-LIP \
        --output_dir  results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ResidueExample:
    def __init__(self, protein_id: str, sequence: str, y_true: np.ndarray):
        self.protein_id = protein_id
        self.sequence = sequence
        self.y_true = y_true  # per-residue ground truth (int8)
        self.scores: Dict[str, np.ndarray] = {}  # per-residue continuous scores
        self.binary: Dict[str, np.ndarray] = {}  # per-residue binary predictions

    def add_prediction(
        self, model_name: str, scores: np.ndarray, binary: np.ndarray
    ) -> None:
        n = len(self.sequence)
        for arr, label in [(scores, "scores"), (binary, "binary")]:
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch for {self.protein_id} / {model_name} "
                    f"({label}): seq={n}, pred={len(arr)}"
                )
        self.scores[model_name] = scores
        self.binary[model_name] = binary


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _read_blocks(path: str | Path) -> List[List[str]]:
    blocks, current = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_binary_string(s: str) -> np.ndarray:
    return np.fromiter((1 if c == "1" else 0 for c in s.strip()), dtype=np.int8)


def _parse_prob_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in s.split(",") if x], dtype=np.float64)


def _parse_binary_csv_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.int8)
    return np.array([int(x) for x in s.split(",") if x], dtype=np.int8)


def parse_truth_file(path: str | Path) -> Dict[str, ResidueExample]:
    records: Dict[str, ResidueExample] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed truth block: {block}")
        protein_id = block[0][1:].strip()
        sequence = block[1].strip()
        y_true = _parse_binary_string("".join(block[2:]).strip())
        if len(sequence) != len(y_true):
            raise ValueError(
                f"Length mismatch for {protein_id}: "
                f"seq={len(sequence)}, labels={len(y_true)}"
            )
        records[protein_id] = ResidueExample(protein_id, sequence, y_true)
    return records


def parse_prediction_csv(
    path: str | Path,
    records: Dict[str, ResidueExample],
    model_name: str,
) -> None:
    """
    Parse a prediction CSV (protein_id, length, predictions, binary_predictions).
    Extra columns are silently ignored.
    Only proteins present in *records* are loaded; others are silently skipped.
    """
    required = {"protein_id", "length", "predictions", "binary_predictions"}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            pid = row["protein_id"].strip()
            if pid not in records:
                continue
            expected_len = int(row["length"])
            scores = _parse_prob_string(row["predictions"])
            binary = _parse_binary_csv_string(row["binary_predictions"])
            if len(scores) != expected_len or len(binary) != expected_len:
                raise ValueError(
                    f"Length mismatch for {pid} in {path}: "
                    f"expected {expected_len}, got scores={len(scores)}, binary={len(binary)}"
                )
            records[pid].add_prediction(model_name, scores, binary)

    missing = [pid for pid in records if model_name not in records[pid].scores]
    if missing:
        raise ValueError(
            f"Model '{model_name}' is missing predictions for "
            f"{len(missing)} protein(s): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Residue-level metrics
# ---------------------------------------------------------------------------


def compute_residue_metrics(
    records: Dict[str, ResidueExample],
    model_name: str,
) -> Dict:
    """
    Compute residue-level metrics, including several that are robust to class
    imbalance — important for LIP/non-LIP prediction where positive residues
    are typically a small minority.

    Metrics returned
    ----------------
    n_residues      : total residues evaluated
    pos_rate        : fraction of positive (LIP) residues  ← shows imbalance
    mcc             : Matthews Correlation Coefficient      ← balanced, threshold-dependent
    f1              : F1 score at chosen threshold          ← threshold-dependent
    precision       : Precision at chosen threshold
    recall          : Recall at chosen threshold
    auc_roc         : Area under ROC curve                 ← threshold-independent
    avg_precision   : Average Precision (PR-AUC)           ← best single metric for imbalance
    brier_score     : Brier score (calibration quality)    ← lower is better
    """
    missing = [pid for pid, r in records.items() if model_name not in r.scores]
    if missing:
        raise ValueError(
            f"Model '{model_name}' is missing predictions for "
            f"{len(missing)} protein(s): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    y_true = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    y_score = np.concatenate(
        [r.scores[model_name].astype(np.float64) for r in records.values()]
    )
    y_pred = np.concatenate(
        [r.binary[model_name].astype(np.int8) for r in records.values()]
    )

    if len(y_true) == 0:
        raise ValueError(f"No residue-level predictions found for model '{model_name}'")

    out: Dict = {
        "model": model_name,
        "n_residues": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }

    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc_roc"] = float("nan")

    try:
        out["avg_precision"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["avg_precision"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Ordered columns for the printed table
_TABLE_COLS = [
    ("model", "Model"),
    ("n_residues", "N res."),
    ("pos_rate", "LIP%"),
    ("mcc", "MCC"),
    ("f1", "F1"),
    ("precision", "Prec."),
    ("recall", "Rec."),
    ("auc_roc", "AUC-ROC"),
    ("avg_precision", "AP (PR-AUC)"),
    ("brier_score", "Brier↓"),
]


def _fmt(key: str, val) -> str:
    if key == "model":
        return str(val)
    if key == "n_residues":
        return str(int(val))
    if key == "pos_rate":
        return f"{val * 100:.1f}%"
    if key == "brier_score":
        return f"{val:.4f}"
    if np.isfinite(val):
        return f"{val:.4f}"
    return "nan"


def print_results_table(results: List[Dict]) -> None:
    headers = [label for _, label in _TABLE_COLS]
    keys = [k for k, _ in _TABLE_COLS]

    rows = [[_fmt(k, r[k]) for k in keys] for r in results]

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    sep = "-+-".join("-" * w for w in widths)
    fmt = lambda row: " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


_PALETTE = [
    "#e63946",
    "#2176ae",
    "#06d6a0",
    "#f4a261",
    "#9b5de5",
    "#f15bb5",
    "#118ab2",
    "#ffd166",
    "#06a77d",
    "#ef476f",
]


# ── ROC curves ──────────────────────────────────────────────────────────────


def plot_roc_curves(
    records: Dict[str, ResidueExample],
    model_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    handles = []

    for i, name in enumerate(model_names):
        color = _PALETTE[i % len(_PALETTE)]
        lw = 2.2 if i == 0 else 1.6
        try:
            y_true = np.concatenate(
                [r.y_true.astype(np.int8) for r in records.values()]
            )
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, color=color, lw=lw, alpha=0.92)
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AUC = {auc:.3f})")
            )
        except ValueError as e:
            print(f"[plot_roc_curves] Skipping {name}: {e}")
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AUC = n/a)")
            )

    ax.plot([0, 1], [0, 1], color="#aaa", lw=1.0, ls="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Precision-Recall curves ─────────────────────────────────────────────────


def plot_pr_curves(
    records: Dict[str, ResidueExample],
    model_names: List[str],
    title: str = "Precision-Recall Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Precision-Recall curves are the recommended visualisation for imbalanced
    datasets.  The no-skill baseline is shown as a horizontal dashed line at
    the positive class prevalence (LIP residue fraction).

    The Average Precision (AP) summarises the curve as a weighted mean of
    precisions at each threshold, which is equivalent to the area under the
    PR curve.  Unlike AUC-ROC, AP is sensitive to class imbalance and does not
    give credit for true-negative performance — making it the primary metric
    for LIP/non-LIP prediction.
    """
    # Compute global positive rate for the no-skill baseline
    all_true = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    pos_rate = float(all_true.mean())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    handles = []

    for i, name in enumerate(model_names):
        color = _PALETTE[i % len(_PALETTE)]
        lw = 2.2 if i == 0 else 1.6
        try:
            y_true = np.concatenate(
                [r.y_true.astype(np.int8) for r in records.values()]
            )
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            # sklearn returns arrays in descending recall order; reverse for a
            # left-to-right plot
            ax.plot(recall, precision, color=color, lw=lw, alpha=0.92)
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AP = {ap:.3f})")
            )
        except ValueError as e:
            print(f"[plot_pr_curves] Skipping {name}: {e}")
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AP = n/a)")
            )

    # No-skill baseline
    ax.axhline(
        pos_rate,
        color="#aaa",
        lw=1.0,
        ls="--",
        label=f"No skill  (prevalence = {pos_rate:.3f})",
    )
    handles.append(
        Line2D(
            [],
            [],
            color="#aaa",
            lw=1.0,
            ls="--",
            label=f"No skill  (prevalence = {pos_rate:.3f})",
        )
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", color="#e5e5e5", linewidth=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── Metrics bar chart ────────────────────────────────────────────────────────


def plot_metrics_bar(
    results: List[Dict],
    title: str = "Model Performance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing MCC, F1, Average Precision, and Brier Score across
    models.  Average Precision and Brier Score are particularly informative
    for imbalanced LIP prediction.
    """
    model_names = [r["model"] for r in results]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(results))]
    x = np.arange(len(model_names))

    panels = [
        ("mcc", "MCC", False),
        ("f1", "F1 Score", False),
        ("avg_precision", "Avg. Precision (PR-AUC)", False),
        ("brier_score", "Brier Score  (↓ better)", True),
    ]

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(8.0, len(model_names) * 1.6) * ncols / 2, 4.2 * nrows),
    )
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, (key, ylabel, invert) in zip(axes, panels):
        values = [r[key] for r in results]
        bars = ax.bar(
            x,
            values,
            width=0.55,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.012 if val >= 0 else -0.012),
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8.5,
                fontweight="bold",
                color="#333",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        y_min = min(0.0, min(values)) - 0.08
        y_max = max(values) + 0.12
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="#888", linewidth=0.8, zorder=2)
        ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if invert:
            # Visually invert so "shorter bar = better" for Brier score
            ax.invert_yaxis()

    # Hide any unused subplot panels
    for ax in axes[len(panels) :]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction CSV files at the residue level."
    )
    parser.add_argument(
        "--test_truth",
        required=True,
        help="Ground-truth file in FASTA-like format (header / sequence / binary label).",
    )
    parser.add_argument(
        "--pred_files",
        nargs="+",
        required=True,
        help="Prediction CSV files (one per model), in the same order as --names.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Display name for each prediction file, in the same order as --pred_files.",
    )
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    if len(args.pred_files) != len(args.names):
        parser.error(
            f"--pred_files ({len(args.pred_files)}) and --names ({len(args.names)}) "
            "must have the same number of entries."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    test_records = parse_truth_file(args.test_truth)
    print(f"Loaded {len(test_records)} proteins from {args.test_truth}")

    # Load predictions and compute residue-level metrics
    all_results = []
    for pred_file, name in zip(args.pred_files, args.names):
        print(f"Loading predictions for '{name}' from {pred_file} …")
        parse_prediction_csv(pred_file, test_records, name)
        all_results.append(compute_residue_metrics(test_records, name))

    # Print table
    print("\n=== Residue-level evaluation ===")
    print_results_table(all_results)

    # ── Figures ────────────────────────────────────────────────────────────
    model_names = [r["model"] for r in all_results]

    plot_roc_curves(
        test_records,
        model_names,
        title="ROC Curves – LIP Test Set (residue level)",
        save_path=output_dir / "roc_curves.pdf",
    )
    plot_pr_curves(
        test_records,
        model_names,
        title="Precision-Recall Curves – LIP Test Set (residue level)",
        save_path=output_dir / "pr_curves.pdf",
    )
    plot_metrics_bar(
        all_results,
        title="Model Performance – LIP Test Set (residue level)",
        save_path=output_dir / "metrics_bar.pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()
