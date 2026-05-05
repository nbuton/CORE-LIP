"""
core_lip/plotting.py
--------------------
Publication-ready matplotlib figures for residue-level LIP evaluation.

    plot_roc_curves   : ROC curves for one or more models
    plot_pr_curves    : Precision-Recall curves (recommended for imbalanced data)
    plot_metrics_bar  : bar chart comparing MCC, F1, AP, and Brier score
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from core_lip.eval.structures import ResidueExample

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

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


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _save(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")


def _style_ax(ax: plt.Axes) -> None:
    """Apply shared grid / spine style."""
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(which="major", color="#e5e5e5", linewidth=0.8)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------


def plot_roc_curves(
    records: Dict[str, ResidueExample],
    model_names: List[str],
    title: str = "ROC Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot ROC curves for each model in *model_names*.

    Parameters
    ----------
    records     : dict of ResidueExample (ground truth + predictions loaded)
    model_names : model keys to plot, in display order
    title       : figure title
    save_path   : if given, save the figure at this path (PDF/PNG/SVG)

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    handles: List[Line2D] = []

    y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    y_mask = y_true_global != -1
    y_true_global = y_true_global[y_mask]

    for i, name in enumerate(model_names):
        color = _color(i)
        lw = 2.2 if i == 0 else 1.6
        try:
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            y_score = y_score[y_mask]
            fpr, tpr, _ = roc_curve(y_true_global, y_score)
            auc = roc_auc_score(y_true_global, y_score)
            ax.plot(fpr, tpr, color=color, lw=lw, alpha=0.92)
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AUC = {auc:.3f})")
            )
        except ValueError as exc:
            print(f"[plot_roc_curves] Skipping {name}: {exc}")
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
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------


def plot_pr_curves(
    records: Dict[str, ResidueExample],
    model_names: List[str],
    title: str = "Precision-Recall Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for each model in *model_names*.

    PR curves are the recommended visualisation for imbalanced datasets.
    The no-skill baseline is shown as a horizontal dashed line at the positive
    class prevalence (LIP residue fraction).

    Average Precision (AP) summarises the curve as a weighted mean of
    precisions at each threshold — it is the primary metric for LIP prediction
    because, unlike AUC-ROC, it does not reward true-negative performance.

    Parameters
    ----------
    records     : dict of ResidueExample
    model_names : model keys to plot
    title       : figure title
    save_path   : optional save path

    Returns
    -------
    plt.Figure
    """
    y_true_global = np.concatenate([r.y_true.astype(np.int8) for r in records.values()])
    y_mask = y_true_global != -1
    y_true_global = y_true_global[y_mask]
    pos_rate = float(y_true_global.mean())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    handles: List[Line2D] = []

    for i, name in enumerate(model_names):
        color = _color(i)
        lw = 2.2 if i == 0 else 1.6
        try:
            y_score = np.concatenate([r.scores[name] for r in records.values()])
            y_score = y_score[y_mask]
            precision, recall, _ = precision_recall_curve(y_true_global, y_score)
            ap = average_precision_score(y_true_global, y_score)
            ax.plot(recall, precision, color=color, lw=lw, alpha=0.92)
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AP = {ap:.3f})")
            )
        except ValueError as exc:
            print(f"[plot_pr_curves] Skipping {name}: {exc}")
            handles.append(
                Line2D([], [], color=color, lw=lw, label=f"{name}  (AP = n/a)")
            )

    # No-skill baseline
    baseline_label = f"No skill  (prevalence = {pos_rate:.3f})"
    ax.axhline(pos_rate, color="#aaa", lw=1.0, ls="--")
    handles.append(Line2D([], [], color="#aaa", lw=1.0, ls="--", label=baseline_label))

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

# (key, y-axis label, invert_axis)
_BAR_PANELS = [
    ("mcc", "MCC", False),
    ("f1", "F1 Score", False),
    ("avg_precision", "Avg. Precision (PR-AUC)", False),
    ("brier_score", "Brier Score  (↓ better)", True),
]


def plot_metrics_bar(
    results: List[Dict],
    title: str = "Model Performance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing MCC, F1, Average Precision, and Brier Score across
    models.

    Average Precision and Brier Score are particularly informative for
    imbalanced LIP prediction.  The Brier Score axis is inverted so that a
    shorter bar always means better performance.

    Parameters
    ----------
    results   : list of metric dicts as returned by compute_residue_metrics()
    title     : figure title
    save_path : optional save path

    Returns
    -------
    plt.Figure
    """
    model_names = [r["model"] for r in results]
    colors = [_color(i) for i in range(len(results))]
    x = np.arange(len(model_names))

    ncols = 2
    nrows = (len(_BAR_PANELS) + ncols - 1) // ncols
    fig_w = max(8.0, len(model_names) * 1.6) * ncols / 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 4.2 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, (key, ylabel, invert) in zip(axes, _BAR_PANELS):
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
            ax.invert_yaxis()

    for ax in axes[len(_BAR_PANELS) :]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, save_path)
    return fig
