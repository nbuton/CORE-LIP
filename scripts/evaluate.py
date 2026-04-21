"""
CORE-LIP — Step 5: Evaluate predictions
=========================================
Evaluates CORE-LIP against baseline models on a held-out test set.
Threshold selection uses 5-fold cross-validation on the training set (MCC),
exactly as in the CLIP benchmarking protocol.

Usage
-----
    python scripts/evaluate.py \
        --train_truth  data/CLIP_dataset/TR1000_reduced.txt \
        --train_preds  data/predictions/core_lip_TR1000_reduced.csv \
        --test_truth   data/CLIP_dataset/TE440_reduced.txt \
        --test_preds   data/predictions/core_lip_TE440_reduced.csv \
        --output_dir   results/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResidueExample:
    protein_id: str
    sequence: str
    y_true: np.ndarray
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_prediction(self, model_name: str, scores: np.ndarray) -> None:
        if len(scores) != len(self.sequence):
            raise ValueError(
                f"Length mismatch for {self.protein_id} / {model_name}: "
                f"seq={len(self.sequence)}, pred={len(scores)}"
            )
        self.predictions[model_name] = scores


@dataclass
class ComparisonModel:
    """
    Describes an external baseline model to include in the benchmark.

    Example
    -------
    >>> baselines = [
    ...     ComparisonModel("CLIP",    "data/predictions/CLIP_TE440.txt",    parse_clip_predictions, 0.20),
    ...     ComparisonModel("IDP-LIP", "data/predictions/IDPLIP_TE440.txt",  parse_clip_predictions, 0.35),
    ... ]
    """

    name: str
    prediction_file: str | Path
    parser: Callable[[str | Path], Dict[str, np.ndarray]]
    threshold: float


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


def parse_clip_predictions(path: str | Path) -> Dict[str, np.ndarray]:
    """Parse CLIP-format prediction files (FASTA-like with probability lines)."""
    preds: Dict[str, np.ndarray] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed CLIP prediction block: {block}")
        protein_id = block[0][1:].strip()
        prob_lines = [line for line in block[2:] if "," in line]
        if not prob_lines:
            raise ValueError(f"No probability line found for {protein_id}")
        preds[protein_id] = _parse_prob_string(",".join(prob_lines))
    return preds


def parse_core_lip_csv(path: str | Path) -> Dict[str, np.ndarray]:
    """Parse CORE-LIP CSV prediction files (columns: protein_id, length, predictions)."""
    preds: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"protein_id", "length", "predictions"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            protein_id = row["protein_id"].strip()
            expected_len = int(row["length"])
            scores = _parse_prob_string(row["predictions"])
            if len(scores) != expected_len:
                raise ValueError(f"Length mismatch for {protein_id}")
            preds[protein_id] = scores
    return preds


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def attach_predictions(
    records: Dict[str, ResidueExample],
    preds: Dict[str, np.ndarray],
    model_name: str,
) -> None:
    for pid in records:
        if pid not in preds:
            raise ValueError(
                f"Missing prediction for protein '{pid}' in model '{model_name}'"
            )
        records[pid].add_prediction(model_name, preds[pid])


def select_threshold_cv(
    records: Dict[str, ResidueExample],
    model_name: str,
    n_splits: int = 5,
    random_state: int = 0,
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Select the decision threshold that maximises mean MCC in 5-fold CV
    (protein-level folds), matching the CLIP benchmarking protocol.
    """
    protein_ids = np.array(
        [pid for pid, rec in records.items() if model_name in rec.predictions]
    )
    if len(protein_ids) < 2:
        raise ValueError(f"Need ≥ 2 proteins for CV; got {len(protein_ids)}")

    n_splits_eff = min(n_splits, len(protein_ids))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    all_scores = np.concatenate(
        [records[pid].predictions[model_name] for pid in protein_ids]
    )
    candidate_thresholds = np.sort(np.unique(all_scores))

    mean_mccs = []
    for thr in candidate_thresholds:
        fold_mccs = []
        for _, val_idx in kf.split(protein_ids):
            val_ids = protein_ids[val_idx]
            y_true = np.concatenate([records[pid].y_true for pid in val_ids])
            y_score = np.concatenate(
                [records[pid].predictions[model_name] for pid in val_ids]
            )
            y_pred = (y_score > thr).astype(np.int8)
            fold_mccs.append(matthews_corrcoef(y_true, y_pred))
        mean_mccs.append(float(np.mean(fold_mccs)))

    best_idx = int(np.argmax(mean_mccs))
    return float(candidate_thresholds[best_idx]), list(
        zip(candidate_thresholds.tolist(), mean_mccs)
    )


def compute_metrics(
    records: Dict[str, ResidueExample],
    model_name: str,
    threshold: float,
) -> Dict:
    y_true_all, y_score_all = [], []
    for rec in records.values():
        if model_name in rec.predictions:
            y_true_all.append(rec.y_true.astype(np.int8))
            y_score_all.append(rec.predictions[model_name].astype(np.float64))

    if not y_true_all:
        raise ValueError(f"No predictions found for model '{model_name}'")

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    y_pred = (y_score > threshold).astype(np.int8)

    out = {
        "model": model_name,
        "threshold": float(threshold),
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
    headers = ["model", "threshold", "n_residues", "mcc", "f1", "auc_roc"]
    rows = [
        [
            str(r["model"]),
            f'{r["threshold"]:.6f}',
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
        lw = 2.2 if name == "CORE-LIP" else 1.6
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
    parser = argparse.ArgumentParser(description="Evaluate CORE-LIP predictions.")
    parser.add_argument("--train_truth", default="data/CLIP_dataset/TR1000_reduced.txt")
    parser.add_argument(
        "--train_preds", default="data/predictions/core_lip_TR1000_reduced.csv"
    )
    parser.add_argument("--test_truth", default="data/CLIP_dataset/TE440_reduced.txt")
    parser.add_argument(
        "--test_preds", default="data/predictions/core_lip_TE440_reduced.csv"
    )
    parser.add_argument(
        "--clip_preds",
        default=None,
        help="Optional: path to CLIP predictions for comparison.",
    )
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Tune threshold on training set
    print("=== Threshold tuning (5-fold CV on training set) ===")
    train_records = parse_truth_file(args.train_truth)
    attach_predictions(train_records, parse_core_lip_csv(args.train_preds), "CORE-LIP")
    best_thr, tuning = select_threshold_cv(train_records, "CORE-LIP")
    print(f"Best threshold: {best_thr:.6f}")

    # Step 2 — Declare baselines
    baselines: List[ComparisonModel] = []
    if args.clip_preds:
        baselines.append(
            ComparisonModel(
                name="CLIP",
                prediction_file=args.clip_preds,
                parser=parse_clip_predictions,
                threshold=0.20,
            )
        )
    # Add more baselines here as needed

    # Step 3 — Evaluate on test set
    print("\n=== Test-set evaluation ===")
    test_records = parse_truth_file(args.test_truth)
    all_results = []

    for bm in baselines:
        preds = bm.parser(bm.prediction_file)
        attach_predictions(test_records, preds, bm.name)
        all_results.append(compute_metrics(test_records, bm.name, bm.threshold))

    attach_predictions(test_records, parse_core_lip_csv(args.test_preds), "CORE-LIP")
    all_results.append(compute_metrics(test_records, "CORE-LIP", best_thr))

    print_results_table(all_results)

    # Step 4 — Figures
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
