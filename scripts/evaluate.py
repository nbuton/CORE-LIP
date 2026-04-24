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
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core_lip.evaluation import ResidueExample, parse_prediction_csv, parse_truth_file
from core_lip.eval.plotting import plot_metrics_bar, plot_pr_curves, plot_roc_curves

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
        title="ROC Curves - LIP Test Set (residue level)",
        save_path=output_dir / "roc_curves.pdf",
    )
    plot_pr_curves(
        test_records,
        model_names,
        title="Precision-Recall Curves - LIP Test Set (residue level)",
        save_path=output_dir / "pr_curves.pdf",
    )
    plot_metrics_bar(
        all_results,
        title="Model Performance - LIP Test Set (residue level)",
        save_path=output_dir / "metrics_bar.pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()
