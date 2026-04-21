"""
CORE-LIP — Step 2: Visualise feature distributions
===================================================
Generates violin-plot comparisons of conformational features between
LIP and non-LIP residues, and saves a publication-ready PDF figure.

Usage
-----
    python scripts/visualize_features.py \
        --dataset  data/CLIP_dataset/TR1000_max_1024.txt \
        --h5       data/protein_MD_properties.h5 \
        --output   results/feature_comparison_violin.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from core_lip.utils import read_protein_data


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

LOCAL_FEATURES = [
    "phi_entropy",
    "psi_entropy",
    "sasa_abs_mean",
    "sasa_abs_std",
    "sasa_rel_mean",
    "sasa_rel_std",
    "ss_propensity_B",
    "ss_propensity_C",
]
PAIRWISE_FEATURES = ["dccm", "contact_map", "distance_fluctuations"]
ALL_FEATURES = LOCAL_FEATURES + PAIRWISE_FEATURES

# Visual constants
DOUBLE_COL_W, SINGLE_COL_W, PANEL_H = 7.0, 3.5, 2.5
DPI, ALPHA_VIOLIN, ALPHA_SCATTER, STRIP_JITTER = 300, 0.4, 0.3, 0.1
PALETTE = {"teal": "#008080", "orange": "#FF8C00"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohen_d(x, y) -> float:
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0.0


def format_pvalue(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def si_score(matrix, min_dist: int, max_dist: Optional[int] = None) -> np.ndarray:
    """Compute a sequence-local score from a pairwise matrix (e.g. DCCM)."""
    M = np.asarray(matrix, dtype=float)
    L = M.shape[0]
    if max_dist is None:
        max_dist = L
    scores = np.full(L, np.nan)
    indices = np.arange(L)
    for i in range(L):
        distances = np.abs(indices - i)
        mask = (distances >= min_dist) & (distances <= max_dist)
        row_sel = M[i, mask]
        if row_sel.size > 0:
            scores[i] = (row_sel**2).mean()
    return scores


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_feature_data(
    txt_path: str,
    h5_path: str,
    local_features: list[str],
    pairwise_features: list[str],
    min_dist_si: int = 0,
    max_dist_si: Optional[int] = None,
):
    df = read_protein_data(txt_path)
    X_features, y_list = [], []

    with h5py.File(h5_path, "r") as h5_data:
        for _, row in df.iterrows():
            pid = row["protein_id"]
            local_feats = np.stack(
                [h5_data[pid][f][()] for f in local_features], axis=0
            )
            if pairwise_features:
                pairwise_feats = np.stack(
                    [
                        si_score(h5_data[pid][f][()], min_dist_si, max_dist_si)
                        for f in pairwise_features
                    ],
                    axis=0,
                )
                all_feats = np.concatenate((local_feats, pairwise_feats), axis=0)
            else:
                all_feats = local_feats

            labels = np.array([int(c) for c in row["LIP_annotations"]])
            X_features.append(all_feats)
            y_list.append(labels)

    return X_features, y_list


def build_plot_data(
    x_list: list,
    y_list: list,
    features: list[str],
) -> dict:
    plot_data = {feat: {"LIP": [], "NON_LIP": []} for feat in features}
    for feats, mask in zip(x_list, y_list):
        is_lip = mask.astype(bool)
        for i, feat in enumerate(features):
            vals = feats[i]
            plot_data[feat]["LIP"].extend(vals[is_lip])
            plot_data[feat]["NON_LIP"].extend(vals[~is_lip])
    return plot_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def violin_subplot(
    ax, g1_vals, g2_vals, g1_label, g2_label, g1_color, g2_color, prop_name
):
    def _finite(v):
        v = np.asarray(v, dtype=float)
        return v[np.isfinite(v)]

    def _clip(v):
        v = _finite(v)
        if v.size == 0:
            return v
        lo, hi = np.nanpercentile(v, [1.0, 99.0])
        return v if np.isclose(lo, hi) else np.clip(v, lo, hi)

    raw = [_finite(g1_vals), _finite(g2_vals)]
    plot = [_clip(g) for g in raw]
    labels = [g1_label, g2_label]
    colors = [g1_color, g2_color]
    positions = [1, 2]

    can_violin = all(len(np.unique(g)) > 1 and len(g) > 5 for g in plot)

    if can_violin:
        parts = ax.violinplot(
            plot,
            positions=positions,
            showmedians=True,
            showextrema=False,
            widths=0.60,
            points=300,
        )
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_edgecolor("none")
            pc.set_alpha(ALPHA_VIOLIN)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.2)
    else:
        bp = ax.boxplot(
            plot,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.2),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(ALPHA_VIOLIN)

    all_plot = np.concatenate([g for g in plot if len(g) > 0])
    if all_plot.size > 0:
        y_lo, y_hi = np.nanpercentile(all_plot, [1, 99])
        if np.isclose(y_lo, y_hi):
            y_lo, y_hi = float(np.nanmin(all_plot)), float(np.nanmax(all_plot))
    else:
        y_lo, y_hi = 0.0, 1.0

    y_range = y_hi - y_lo
    y_pad = max(y_range * 0.12, 1e-6)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad * 3.2)

    rng = np.random.default_rng(42)
    for vals, pos, col in zip(raw, positions, colors):
        if not len(vals):
            continue
        pv = vals if len(vals) <= 500 else rng.choice(vals, 500, replace=False)
        jit = rng.uniform(-STRIP_JITTER, STRIP_JITTER, size=len(pv))
        ax.scatter(
            pos + jit,
            pv,
            color=col,
            alpha=ALPHA_SCATTER,
            s=2.5,
            linewidths=0,
            zorder=3,
            clip_on=True,
        )

    if len(raw[0]) >= 5 and len(raw[1]) >= 5:
        _, pval = stats.mannwhitneyu(raw[1], raw[0], alternative="two-sided")
        d = cohen_d(raw[1], raw[0])
        stars = (
            "***"
            if pval < 0.001
            else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        )
        y_bar = y_hi + y_pad * 0.5
        bar_h = y_pad * 0.30
        ax.plot(
            [1, 1, 2, 2],
            [y_bar, y_bar + bar_h, y_bar + bar_h, y_bar],
            color="#333",
            linewidth=0.8,
        )
        ax.text(
            1.5,
            y_bar + bar_h * 1.1,
            stars,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
        annot = (
            f"{format_pvalue(pval)}\n|d|={abs(d):.2f}"
            if not np.isnan(d)
            else format_pvalue(pval)
        )
        ax.text(
            0.97,
            0.99,
            annot,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
            color="#444",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#ccc",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([""] * 2)
    ax.tick_params(axis="x", which="both", length=0)
    for vals, pos, lbl in zip(raw, positions, labels):
        ax.text(
            pos,
            -0.04,
            f"{lbl}\n(n={len(vals):,})",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            clip_on=False,
            linespacing=1.3,
        )
    ax.set_title(prop_name.replace("_", " "), fontsize=6.5, fontweight="bold", pad=4)
    ax.set_xlim(0.4, 2.6)
    ax.tick_params(axis="y", which="both", length=2, width=0.6)


def make_figure(data: dict, output_path: str, ncols: int = 4) -> None:
    props = list(data.keys())
    n = len(props)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = DOUBLE_COL_W if ncols > 2 else SINGLE_COL_W
    fig_h = nrows * PANEL_H + 0.35

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), gridspec_kw=dict(hspace=0.72, wspace=0.45)
    )
    axes_flat = np.array(axes).flatten()

    for ax, prop in zip(axes_flat, props):
        g1 = np.asarray(data[prop]["NON_LIP"], dtype=float)
        g2 = np.asarray(data[prop]["LIP"], dtype=float)
        g1, g2 = g1[~np.isnan(g1)], g2[~np.isnan(g2)]
        if not len(g1) and not len(g2):
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#aaa",
                fontsize=6,
            )
            continue
        violin_subplot(
            ax, g1, g2, "Non-LIP", "LIP", PALETTE["teal"], PALETTE["orange"], prop
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", transparent=False)
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualise feature distributions.")
    parser.add_argument("--dataset", default="data/CLIP_dataset/TR1000_max_1024.txt")
    parser.add_argument("--h5", default="data/protein_MD_properties.h5")
    parser.add_argument("--output", default="results/feature_comparison_violin.pdf")
    parser.add_argument("--ncols", type=int, default=4)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    x_list, y_list = load_feature_data(
        args.dataset,
        args.h5,
        LOCAL_FEATURES,
        PAIRWISE_FEATURES,
        min_dist_si=0,
        max_dist_si=None,
    )
    plot_data = build_plot_data(x_list, y_list, ALL_FEATURES)

    print("Generating figure …")
    make_figure(plot_data, args.output, ncols=args.ncols)


if __name__ == "__main__":
    main()
