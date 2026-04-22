"""
CORE-LIP — Step 5: Evaluate predictions
=========================================
Evaluates one or more models on a held-out test set.
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
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
import csv
import sys

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ResidueExample:
    def __init__(self, protein_id: str, sequence: str, y_true: np.ndarray):
        self.protein_id = protein_id
        self.sequence = sequence
        self.y_true = y_true
        self.predictions: Dict[str, np.ndarray] = {}  # continuous scores
        self.binary_preds: Dict[str, np.ndarray] = {}  # binary labels

    def add_prediction(
        self, model_name: str, scores: np.ndarray, binary: np.ndarray
    ) -> None:
        for arr, label in [(scores, "scores"), (binary, "binary")]:
            if len(arr) != len(self.sequence):
                raise ValueError(
                    f"Length mismatch for {self.protein_id} / {model_name} "
                    f"({label}): seq={len(self.sequence)}, pred={len(arr)}"
                )
        self.predictions[model_name] = scores
        self.binary_preds[model_name] = binary


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
    return np.array([float(x) for x in s.split(",") if x != ""], dtype=np.float64)


def _parse_binary_csv_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.int8)
    return np.array([int(x) for x in s.split(",") if x != ""], dtype=np.int8)


def parse_truth_file(path: str | Path) -> Dict[str, ResidueExample]:
    records: Dict[str, ResidueExample] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed truth block: {block}")
        protein_id = block[0][1:].strip()
        sequence = block[1].strip()
        y_true = _parse_binary_string("".join(block[2:]).strip())
        if len(sequence) != len(y_true):
            raise ValueError(f"Length mismatch for {protein_id}")
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
        missing = []
        for row in reader:
            pid = row["protein_id"].strip()
            if pid not in records:
                continue  # extra protein — ignore
            expected_len = int(row["length"])
            scores = _parse_prob_string(row["predictions"])
            binary = _parse_binary_csv_string(row["binary_predictions"])
            if len(scores) != expected_len or len(binary) != expected_len:
                raise ValueError(
                    f"Length mismatch for {pid} in {path}: "
                    f"expected {expected_len}, got scores={len(scores)}, binary={len(binary)}"
                )
            records[pid].add_prediction(model_name, scores, binary)

    # Report proteins in truth file that had no prediction
    missing = [pid for pid in records if model_name not in records[pid].predictions]
    if missing:
        raise ValueError(
            f"Model '{model_name}' is missing predictions for "
            f"{len(missing)} protein(s): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    records: Dict[str, ResidueExample],
    model_name: str,
) -> Dict:
    y_true_all, y_score_all, y_pred_all = [], [], []
    for rec in records.values():
        if model_name in rec.predictions:
            y_true_all.append(rec.y_true.astype(np.int8))
            y_score_all.append(rec.predictions[model_name].astype(np.float64))
            y_pred_all.append(rec.binary_preds[model_name].astype(np.int8))

    if not y_true_all:
        raise ValueError(f"No predictions found for model '{model_name}'")

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    y_pred = np.concatenate(y_pred_all)

    out: Dict = {
        "model": model_name,
        "n_residues": int(len(y_true)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc_roc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: List[Dict]) -> None:
    headers = ["model", "n_residues", "mcc", "f1", "auc_roc"]
    rows = [
        [
            str(r["model"]),
            str(int(r["n_residues"])),
            f'{r["mcc"]:.4f}',
            f'{r["f1"]:.4f}',
            f'{r["auc_roc"]:.4f}' if np.isfinite(r["auc_roc"]) else "nan",
        ]
        for r in results
    ]

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
            yt = np.concatenate(
                [
                    r.y_true.astype(np.int8)
                    for r in records.values()
                    if name in r.predictions
                ]
            )
            ys = np.concatenate(
                [r.predictions[name] for r in records.values() if name in r.predictions]
            )
            fpr, tpr, _ = roc_curve(yt, ys)
            auc = roc_auc_score(yt, ys)
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


def plot_metrics_bar(
    results: List[Dict],
    title: str = "Model Performance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    model_names = [r["model"] for r in results]
    mcc_vals = [r["mcc"] for r in results]
    f1_vals = [r["f1"] for r in results]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(results))]
    x = np.arange(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, len(model_names) * 1.4), 4.8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, values, ylabel in [
        (axes[0], mcc_vals, "Matthews Correlation Coefficient"),
        (axes[1], f1_vals, "F1 Score"),
    ]:
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
        ax.set_ylim(min(0.0, min(values)) - 0.08, max(values) + 0.12)
        ax.axhline(0, color="#888", linewidth=0.8, zorder=2)
        ax.grid(axis="y", which="major", color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV files.")
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

    # Load predictions and compute metrics
    all_results = []
    for pred_file, name in zip(args.pred_files, args.names):
        print(f"Loading predictions for '{name}' from {pred_file} …")
        parse_prediction_csv(pred_file, test_records, name)
        all_results.append(compute_metrics(test_records, name))

    # Print table
    print("\n=== Test-set evaluation ===")
    print_results_table(all_results)

    # Figures
    model_names = [r["model"] for r in all_results]
    plot_roc_curves(
        test_records,
        model_names,
        title="ROC Curves – LIP Test Set",
        save_path=output_dir / "roc_curves.pdf",
    )
    plot_metrics_bar(
        all_results,
        title="Model Performance – LIP Test Set",
        save_path=output_dir / "metrics_bar.pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()
