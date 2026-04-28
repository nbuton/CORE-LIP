"""
lip_interpretability.py
=======================
Per-residue attribution and feature-importance analysis for
ProteinMultiScaleTransformer LIP predictions.

Supported methods (each is a standalone Interpreter subclass):
  - IntegratedGradientsInterpreter   : gradient-based per-feature attribution
  - GradientSHAPInterpreter          : SHAP values over a baseline distribution
  - FeatureAblationInterpreter       : channel-level ablation (which input stream matters)
  - OcclusionInterpreter             : sliding-window perturbation on local / pairwise
  - AttentionRolloutInterpreter      : aggregate attention across transformer blocks
  - FeatureValueCorrelationAnalyzer  : correlation between raw feature values and LIP score
  - FeatureRangeProfiler             : bin feature values and compute LIP rate per bin

All interpreters share a common base class `BaseInterpreter` and return
`AttributionResult` dataclasses so downstream analysis / plotting is uniform.

Quick start
-----------
>>> from lip_interpretability import run_all, REGISTRY
>>> results = run_all(model, loader, device, scalar_names, local_names, pairwise_names)
>>> results["integrated_gradients"].to_dataframe().to_csv("ig_attributions.csv")

Adding a new method
-------------------
1. Subclass `BaseInterpreter` and implement `run()`.
2. Decorate it with `@register("your_method_name")`.
3. Done — it will automatically appear in `REGISTRY` and `run_all()`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Captum imports — install with: pip install captum
try:
    from captum.attr import (
        FeatureAblation,
        GradientShap,
        IntegratedGradients,
        Occlusion,
    )

    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False
    warnings.warn(
        "captum not found. Gradient-based interpreters will be unavailable. "
        "Install with: pip install captum"
    )

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, type] = {}


def register(name: str) -> Callable:
    """Class decorator to add an interpreter to the global registry."""

    def decorator(cls):
        REGISTRY[name] = cls
        cls.method_name = name
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    """
    Stores per-residue or per-feature attribution scores.

    Attributes
    ----------
    method        : interpreter name
    protein_ids   : list of protein identifiers (length B)
    attributions  : dict mapping stream name → np.ndarray of shape (B, L, F)
                    or (B, L) for scalar summaries
    metadata      : free-form dict for extra outputs (e.g., convergence deltas)
    feature_names : dict mapping stream → list of feature names
    """

    method: str
    protein_ids: List[str]
    attributions: Dict[str, np.ndarray]  # stream → array
    feature_names: Dict[str, List[str]]
    masks: np.ndarray  # (B, L) bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def mean_per_feature(self) -> pd.DataFrame:
        """
        Returns a DataFrame with mean |attribution| per feature across all
        residues that are in the valid mask.  Useful for global feature ranking.
        """
        rows = []
        for stream, arr in self.attributions.items():
            names = self.feature_names.get(
                stream, [f"f{i}" for i in range(arr.shape[-1])]
            )
            # arr shape: (B, L, F)
            for f_idx, fname in enumerate(names):
                vals = arr[:, :, f_idx][self.masks]  # flatten valid residues
                rows.append(
                    {
                        "stream": stream,
                        "feature": fname,
                        "mean_abs_attr": float(np.abs(vals).mean()),
                        "mean_attr": float(vals.mean()),
                        "std_attr": float(vals.std()),
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values("mean_abs_attr", ascending=False)
            .reset_index(drop=True)
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Long-format DataFrame: one row per (protein, residue, stream, feature).
        """
        rows = []
        B, L = self.masks.shape
        for b, prot in enumerate(self.protein_ids):
            for stream, arr in self.attributions.items():
                names = self.feature_names.get(
                    stream, [f"f{i}" for i in range(arr.shape[-1])]
                )
                for l_idx in range(L):
                    if not self.masks[b, l_idx]:
                        continue
                    for f_idx, fname in enumerate(names):
                        rows.append(
                            {
                                "protein_id": prot,
                                "residue": l_idx,
                                "stream": stream,
                                "feature": fname,
                                "attribution": float(arr[b, l_idx, f_idx]),
                            }
                        )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wrapper — makes model accept a single tuple for Captum
# ---------------------------------------------------------------------------


class _ModelWrapper(nn.Module):
    """
    Thin wrapper so Captum can call the model with a tuple of tensors.
    The wrapper also squeezes the [B, L, 1] output to [B, L] and
    averages over the sequence dimension to return a scalar per sample
    (needed for some Captum methods that require scalar output).
    """

    def __init__(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        plm_pad: Optional[torch.Tensor],
        reduce: str = "none",
    ):
        super().__init__()
        self.model = model
        self.mask = mask
        self.plm_pad = plm_pad
        self.reduce = reduce  # "none" | "mean" | "sum"

    def forward(self, tokens, x_scalar, x_local, x_pairwise):
        logits = self.model(
            tokens, x_scalar, x_local, x_pairwise, self.mask, self.plm_pad
        )
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # [B, L]
        if self.reduce == "mean":
            # mask out padding before averaging
            m = self.mask.float()
            return (logits * m).sum(-1) / m.sum(-1).clamp(min=1)
        return logits


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseInterpreter(ABC):
    """
    Abstract base for all LIP interpreters.

    Parameters
    ----------
    model           : trained ProteinMultiScaleTransformer (eval mode)
    scalar_names    : list of scalar feature names  (len = F_s)
    local_names     : list of local  feature names  (len = F_l)
    pairwise_names  : list of pairwise feature names (len = F_p)
    device          : torch device
    """

    method_name: str = "base"

    def __init__(
        self,
        model: nn.Module,
        scalar_names: List[str],
        local_names: List[str],
        pairwise_names: List[str],
        device: torch.device,
    ):
        self.model = model.eval()
        self.scalar_names = scalar_names
        self.local_names = local_names
        self.pairwise_names = pairwise_names
        self.device = device
        self.feature_names = {
            "scalar": scalar_names,
            "local": local_names,
            "pairwise": pairwise_names,
        }

    @abstractmethod
    def run(
        self,
        loader: DataLoader,
        **kwargs,
    ) -> AttributionResult:
        """Compute attributions for the full dataset in *loader*."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch, device):
        (
            x_scalar_pad,
            x_local_pad,
            x_pairwise_pad,
            seq_pad,
            mask,
            protein_ids,
            plm_pad,
        ) = batch
        return (
            x_scalar_pad.to(device),
            x_local_pad.to(device),
            x_pairwise_pad.to(device),
            seq_pad.long().to(device),
            mask.to(device),
            protein_ids,
            plm_pad.to(device) if plm_pad is not None else None,
        )

    @staticmethod
    def _expand_pairwise_to_residue(pairwise_attr: np.ndarray) -> np.ndarray:
        """
        pairwise_attr : (B, L, L, F_p) → average over one axis → (B, L, F_p)
        When Captum returns attributions for a 2-D contact map we reduce it
        so the result is per-residue like the other streams.
        """
        return pairwise_attr.mean(axis=2)


# ---------------------------------------------------------------------------
# Interpreter 1 — Integrated Gradients
# ---------------------------------------------------------------------------


@register("integrated_gradients")
class IntegratedGradientsInterpreter(BaseInterpreter):
    """
    Integrated Gradients (Sundararajan et al., 2017).

    Attributes each residue's LIP probability to individual feature dimensions.
    Baseline: zero tensors (disorder = no conformational signal).

    Biological reading
    ------------------
    Large positive attribution  → feature value *increases* LIP probability
    Large negative attribution  → feature value *decreases* LIP probability
    Near-zero                   → feature is ignored by the model for this residue
    """

    def run(
        self,
        loader: DataLoader,
        n_steps: int = 50,
        internal_batch_size: int = 4,
        **kwargs,
    ) -> AttributionResult:
        if not _CAPTUM_AVAILABLE:
            raise RuntimeError("captum is required for IntegratedGradientsInterpreter")

        all_scalar, all_local, all_pairwise = [], [], []
        all_masks, all_ids = [], []

        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = self._unpack_batch(
                batch, self.device
            )

            wrapper = _ModelWrapper(self.model, mask, plm_pad, reduce="none")
            ig = IntegratedGradients(wrapper)

            inputs = (tokens.float(), x_sc, x_lo, x_pw)
            baselines = tuple(torch.zeros_like(t) for t in inputs)

            # We attribute over all residues simultaneously;
            # target=None → attributions for logit vector [B, L]
            # We sum over residue dim to get a scalar then attribute — or
            # use target per-residue in a loop.  For efficiency we attribute
            # the *mean* logit (see wrapper reduce="mean").
            wrapper_mean = _ModelWrapper(self.model, mask, plm_pad, reduce="mean")
            ig_mean = IntegratedGradients(wrapper_mean)

            attrs = ig_mean.attribute(
                inputs=inputs,
                baselines=baselines,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )
            # attrs is a tuple: (tokens_attr, scalar_attr, local_attr, pairwise_attr)
            # shapes: (B, L, *) — we skip tokens (index 0)
            sc_attr = attrs[1].detach().cpu().numpy()  # (B, L, F_s)
            lo_attr = attrs[2].detach().cpu().numpy()  # (B, L, F_l)
            pw_attr = attrs[3].detach().cpu().numpy()  # (B, L, L, F_p) or (B, L, F_p)

            if pw_attr.ndim == 4:
                pw_attr = self._expand_pairwise_to_residue(pw_attr)

            all_scalar.append(sc_attr)
            all_local.append(lo_attr)
            all_pairwise.append(pw_attr)
            all_masks.append(mask.cpu().numpy().astype(bool))
            all_ids.extend(prot_ids)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "scalar": np.concatenate(all_scalar, axis=0),
                "local": np.concatenate(all_local, axis=0),
                "pairwise": np.concatenate(all_pairwise, axis=0),
            },
            feature_names=self.feature_names,
            masks=np.concatenate(all_masks, axis=0),
        )


# ---------------------------------------------------------------------------
# Interpreter 2 — GradientSHAP
# ---------------------------------------------------------------------------


@register("gradient_shap")
class GradientSHAPInterpreter(BaseInterpreter):
    """
    GradientSHAP (Lundberg & Lee, 2017 via Captum).

    Uses the conformational ensemble as a natural baseline distribution:
    pass `n_baselines` random samples from the dataset as the reference set.
    This is biologically meaningful — the baseline is "typical disordered
    protein behaviour" rather than an arbitrary zero.

    Parameters
    ----------
    n_baselines : number of baseline samples drawn at random per batch
    """

    def run(
        self,
        loader: DataLoader,
        n_baselines: int = 10,
        **kwargs,
    ) -> AttributionResult:
        if not _CAPTUM_AVAILABLE:
            raise RuntimeError("captum is required for GradientSHAPInterpreter")

        # Collect a pool of baseline tensors from the first pass
        baseline_sc, baseline_lo, baseline_pw, baseline_tok = [], [], [], []
        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, _, _) = self._unpack_batch(
                batch, self.device
            )
            baseline_sc.append(x_sc)
            baseline_lo.append(x_lo)
            baseline_pw.append(x_pw)
            baseline_tok.append(tokens.float())
            if sum(b.shape[0] for b in baseline_sc) >= n_baselines:
                break

        bl_sc = torch.cat(baseline_sc, dim=0)[:n_baselines]
        bl_lo = torch.cat(baseline_lo, dim=0)[:n_baselines]
        bl_pw = torch.cat(baseline_pw, dim=0)[:n_baselines]
        bl_tok = torch.cat(baseline_tok, dim=0)[:n_baselines]

        all_scalar, all_local, all_pairwise = [], [], []
        all_masks, all_ids = [], []

        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = self._unpack_batch(
                batch, self.device
            )
            B = x_sc.shape[0]

            wrapper = _ModelWrapper(self.model, mask, plm_pad, reduce="mean")
            gs = GradientShap(wrapper)

            inputs = (tokens.float(), x_sc, x_lo, x_pw)
            # Repeat baselines to match batch size (GradientShap stochastic sampling)
            baselines = (bl_tok[:B], bl_sc[:B], bl_lo[:B], bl_pw[:B])

            attrs = gs.attribute(inputs=inputs, baselines=baselines)

            sc_attr = attrs[1].detach().cpu().numpy()
            lo_attr = attrs[2].detach().cpu().numpy()
            pw_attr = attrs[3].detach().cpu().numpy()
            if pw_attr.ndim == 4:
                pw_attr = self._expand_pairwise_to_residue(pw_attr)

            all_scalar.append(sc_attr)
            all_local.append(lo_attr)
            all_pairwise.append(pw_attr)
            all_masks.append(mask.cpu().numpy().astype(bool))
            all_ids.extend(prot_ids)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "scalar": np.concatenate(all_scalar, axis=0),
                "local": np.concatenate(all_local, axis=0),
                "pairwise": np.concatenate(all_pairwise, axis=0),
            },
            feature_names=self.feature_names,
            masks=np.concatenate(all_masks, axis=0),
        )


# ---------------------------------------------------------------------------
# Interpreter 3 — Feature Ablation (channel-level)
# ---------------------------------------------------------------------------


@register("feature_ablation")
class FeatureAblationInterpreter(BaseInterpreter):
    """
    Feature Ablation — zeroes out one feature channel at a time and measures
    the drop in mean LIP probability.

    Biologically useful for answering: "Which input stream matters most?"
    and "Which individual descriptor drives LIP prediction?"

    Returns attributions shaped (B, L, F) where each value is the
    *drop in mean probability* when that feature channel is zeroed.
    """

    def run(self, loader: DataLoader, **kwargs) -> AttributionResult:
        if not _CAPTUM_AVAILABLE:
            raise RuntimeError("captum is required for FeatureAblationInterpreter")

        all_scalar, all_local, all_pairwise = [], [], []
        all_masks, all_ids = [], []

        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = self._unpack_batch(
                batch, self.device
            )

            wrapper = _ModelWrapper(self.model, mask, plm_pad, reduce="mean")
            abl = FeatureAblation(wrapper)

            inputs = (tokens.float(), x_sc, x_lo, x_pw)

            # feature_mask: same shape as input, each unique int = one feature group
            # We ablate one feature dimension (last axis) at a time
            def _make_feature_mask(t: torch.Tensor, offset: int):
                """Each slice along the last axis gets a unique index."""
                mask_t = torch.zeros_like(t, dtype=torch.long)
                for i in range(t.shape[-1]):
                    mask_t[..., i] = offset + i
                return mask_t

            F_tok = tokens.shape[-1] if tokens.dim() > 2 else 1
            fm_tok = torch.zeros_like(
                tokens.float(), dtype=torch.long
            )  # ablate whole token
            fm_sc = _make_feature_mask(x_sc, offset=1)
            fm_lo = _make_feature_mask(x_lo, offset=1 + x_sc.shape[-1])
            fm_pw = _make_feature_mask(x_pw, offset=1 + x_sc.shape[-1] + x_lo.shape[-1])

            attrs = abl.attribute(
                inputs=inputs,
                feature_mask=(fm_tok, fm_sc, fm_lo, fm_pw),
            )

            sc_attr = attrs[1].detach().cpu().numpy()
            lo_attr = attrs[2].detach().cpu().numpy()
            pw_attr = attrs[3].detach().cpu().numpy()
            if pw_attr.ndim == 4:
                pw_attr = self._expand_pairwise_to_residue(pw_attr)

            all_scalar.append(sc_attr)
            all_local.append(lo_attr)
            all_pairwise.append(pw_attr)
            all_masks.append(mask.cpu().numpy().astype(bool))
            all_ids.extend(prot_ids)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "scalar": np.concatenate(all_scalar, axis=0),
                "local": np.concatenate(all_local, axis=0),
                "pairwise": np.concatenate(all_pairwise, axis=0),
            },
            feature_names=self.feature_names,
            masks=np.concatenate(all_masks, axis=0),
        )


# ---------------------------------------------------------------------------
# Interpreter 4 — Occlusion (sliding window on local features)
# ---------------------------------------------------------------------------


@register("occlusion")
class OcclusionInterpreter(BaseInterpreter):
    """
    Occlusion with a sliding window over the *sequence* dimension of local
    features — answers: "Which stretch of residues is the model actually
    paying attention to?"

    Biological reading: high occlusion score at position i means that
    *masking that region* drops the LIP prediction for the protein as a whole.

    Parameters
    ----------
    window_size : residue window to occlude at once (default 5)
    """

    def run(
        self,
        loader: DataLoader,
        window_size: int = 5,
        **kwargs,
    ) -> AttributionResult:
        if not _CAPTUM_AVAILABLE:
            raise RuntimeError("captum is required for OcclusionInterpreter")

        all_local, all_masks, all_ids = [], [], []

        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = self._unpack_batch(
                batch, self.device
            )

            wrapper = _ModelWrapper(self.model, mask, plm_pad, reduce="mean")
            occ = Occlusion(wrapper)

            inputs = (tokens.float(), x_sc, x_lo, x_pw)

            # Only slide over local features; keep other inputs fixed
            sliding_window = (
                (1,),  # tokens: no window
                (1, x_sc.shape[-1]),  # scalar: full feature at once
                (window_size, 1),  # local: window over residues
                (1, 1, x_pw.shape[-1]),  # pairwise: single cell
            )

            attrs = occ.attribute(
                inputs=inputs,
                sliding_window_shapes=sliding_window,
            )

            lo_attr = attrs[2].detach().cpu().numpy()  # (B, L, F_l)
            all_local.append(lo_attr)
            all_masks.append(mask.cpu().numpy().astype(bool))
            all_ids.extend(prot_ids)

        # Return only local stream for occlusion
        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "local": np.concatenate(all_local, axis=0),
            },
            feature_names={"local": self.local_names},
            masks=np.concatenate(all_masks, axis=0),
        )


# ---------------------------------------------------------------------------
# Interpreter 5 — Attention Rollout (no Captum needed)
# ---------------------------------------------------------------------------


@register("attention_rollout")
class AttentionRolloutInterpreter(BaseInterpreter):
    """
    Aggregates raw attention weights across all transformer blocks to produce
    a per-residue importance score (Abnar & Zuidema, 2020).

    Requires the model to expose attention weights — expects a list
    `model.blocks` where each block has a `self_attn` module that returns
    (output, attn_weights) when called with `need_weights=True`, OR a
    `model.get_attention_maps()` hook.

    If attention hooks are not available, this falls back to a gradient-based
    attention proxy.

    Biological reading
    ------------------
    High rollout score at residue i means the model integrates information
    *from* many other positions when deciding whether i is a LIP residue.
    Long-range peaks suggest allosteric / context-driven LIP binding.
    """

    def _register_attn_hooks(self):
        """Register forward hooks to capture attention matrices."""
        self._attn_maps: List[torch.Tensor] = []

        def _hook(module, input, output):
            # output can be (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self._attn_maps.append(output[1].detach().cpu())

        handles = []
        for name, module in self.model.named_modules():
            # Adjust class name to match your actual attention module
            if "Attention" in type(module).__name__ or "attention" in name.lower():
                handles.append(module.register_forward_hook(_hook))
        return handles

    @staticmethod
    def _rollout(attn_maps: List[torch.Tensor]) -> np.ndarray:
        """
        attn_maps : list of (B, H, L, L) tensors, one per block
        Returns   : (B, L) rollout scores
        """
        if not attn_maps:
            return None
        # Average over heads, add residual
        result = None
        for attn in attn_maps:
            # attn: (B, H, L, L)
            avg = attn.mean(dim=1)  # (B, L, L)
            avg = avg + torch.eye(avg.shape[-1]).unsqueeze(0)
            avg = avg / avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            result = avg if result is None else torch.bmm(result, avg)
        # result: (B, L, L) — take CLS-like mean over rows
        return result.mean(dim=1).numpy()  # (B, L)

    def run(self, loader: DataLoader, **kwargs) -> AttributionResult:
        all_rollout, all_masks, all_ids = [], [], []

        for batch in loader:
            (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = self._unpack_batch(
                batch, self.device
            )

            self._attn_maps = []
            handles = self._register_attn_hooks()

            with torch.no_grad():
                _ = self.model(tokens, x_sc, x_lo, x_pw, mask, plm_pad)

            for h in handles:
                h.remove()

            rollout = self._rollout(self._attn_maps)  # (B, L) or None

            if rollout is None:
                warnings.warn(
                    "AttentionRollout: no attention weights captured. "
                    "Make sure your attention modules return weights."
                )
                # Fallback: all-ones (uninformative)
                rollout = np.ones((x_sc.shape[0], x_sc.shape[1]))

            # Expand to (B, L, 1) for consistent shape with other streams
            all_rollout.append(rollout[:, :, np.newaxis])
            all_masks.append(mask.cpu().numpy().astype(bool))
            all_ids.extend(prot_ids)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "attention": np.concatenate(all_rollout, axis=0),
            },
            feature_names={"attention": ["rollout_score"]},
            masks=np.concatenate(all_masks, axis=0),
        )


# ---------------------------------------------------------------------------
# Interpreter 6 — Feature-Value × LIP Correlation (no Captum)
# ---------------------------------------------------------------------------


@register("feature_value_correlation")
class FeatureValueCorrelationAnalyzer(BaseInterpreter):
    """
    Collects raw feature values alongside the model's predicted LIP
    probability and computes per-feature statistics:

      - Pearson r between feature value and LIP probability
      - Mean feature value in LIP residues vs non-LIP residues
      - Effect size (Cohen's d)

    This is *not* an attribution method but a statistical analysis that
    directly answers: "What feature values are characteristic of LIP binding?"

    Returns an AttributionResult where attributions["scalar"] etc. are the
    raw feature values (not gradients), and metadata contains the statistics.
    """

    def run(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
        **kwargs,
    ) -> AttributionResult:
        all_sc, all_lo, all_pw = [], [], []
        all_probs, all_masks, all_ids = [], [], []

        with torch.no_grad():
            for batch in loader:
                (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = (
                    self._unpack_batch(batch, self.device)
                )

                logits = self.model(tokens, x_sc, x_lo, x_pw, mask, plm_pad)
                if logits.dim() == 3 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_sc.append(x_sc.cpu().numpy())
                all_lo.append(x_lo.cpu().numpy())
                all_pw.append(x_pw.cpu().numpy())
                all_probs.append(probs)
                all_masks.append(mask.cpu().numpy().astype(bool))
                all_ids.extend(prot_ids)

        sc = np.concatenate(all_sc, axis=0)  # (N, L, F_s)
        lo = np.concatenate(all_lo, axis=0)
        pw = np.concatenate(all_pw, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        stats = {}
        for stream_name, arr, names in [
            ("scalar", sc, self.scalar_names),
            ("local", lo, self.local_names),
            ("pairwise", pw, self.pairwise_names),
        ]:
            # For pairwise: collapse spatial dims → (N, L, F_p)
            if arr.ndim == 4:
                arr = arr.mean(axis=2)

            rows = []
            flat_probs = probs[masks]  # (valid_residues,)
            lip_mask = flat_probs >= threshold

            for f_idx, fname in enumerate(names):
                flat_feat = arr[:, :, f_idx][masks]
                lip_vals = flat_feat[lip_mask]
                non_vals = flat_feat[~lip_mask]

                # Pearson r
                corr = float(np.corrcoef(flat_feat, flat_probs)[0, 1])

                # Cohen's d
                mu1, mu2 = lip_vals.mean(), non_vals.mean()
                pooled_std = np.sqrt(
                    (lip_vals.var() * len(lip_vals) + non_vals.var() * len(non_vals))
                    / (len(lip_vals) + len(non_vals) - 2 + 1e-12)
                )
                cohens_d = float((mu1 - mu2) / (pooled_std + 1e-12))

                rows.append(
                    {
                        "stream": stream_name,
                        "feature": fname,
                        "pearson_r": corr,
                        "cohens_d": cohens_d,
                        "mean_LIP": float(mu1),
                        "mean_nonLIP": float(mu2),
                        "std_LIP": float(lip_vals.std()),
                        "std_nonLIP": float(non_vals.std()),
                        "n_LIP": int(lip_mask.sum()),
                        "n_nonLIP": int((~lip_mask).sum()),
                    }
                )
            stats[stream_name] = pd.DataFrame(rows)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "scalar": np.concatenate(all_sc, axis=0),
                "local": np.concatenate(all_lo, axis=0),
                "pairwise": np.concatenate(all_pw, axis=0),
            },
            feature_names=self.feature_names,
            masks=masks,
            metadata={"statistics": stats},
        )

    def get_statistics(self, result: AttributionResult) -> pd.DataFrame:
        """Concatenate all stream stats into one ranked DataFrame."""
        stats = result.metadata["statistics"]
        df = pd.concat(stats.values(), ignore_index=True)
        return df.reindex(
            df["pearson_r"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interpreter 7 — Feature Range Profiler (value binning × LIP rate)
# ---------------------------------------------------------------------------


@register("feature_range_profiler")
class FeatureRangeProfiler(BaseInterpreter):
    """
    Bins each feature into quantile ranges and computes the mean LIP
    prediction probability within each bin.

    Biological reading: if bin [low, mid] has LIP_rate=0.05 and
    bin [high] has LIP_rate=0.45, the high end of this feature is
    *characteristic of LIP-binding regions*.

    Parameters
    ----------
    n_bins : number of quantile bins (default 10)
    """

    def run(
        self,
        loader: DataLoader,
        n_bins: int = 10,
        **kwargs,
    ) -> AttributionResult:
        all_sc, all_lo, all_pw = [], [], []
        all_probs, all_masks, all_ids = [], [], []

        with torch.no_grad():
            for batch in loader:
                (x_sc, x_lo, x_pw, tokens, mask, prot_ids, plm_pad) = (
                    self._unpack_batch(batch, self.device)
                )

                logits = self.model(tokens, x_sc, x_lo, x_pw, mask, plm_pad)
                if logits.dim() == 3 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_sc.append(x_sc.cpu().numpy())
                all_lo.append(x_lo.cpu().numpy())
                all_pw.append(x_pw.cpu().numpy())
                all_probs.append(probs)
                all_masks.append(mask.cpu().numpy().astype(bool))
                all_ids.extend(prot_ids)

        sc = np.concatenate(all_sc, axis=0)
        lo = np.concatenate(all_lo, axis=0)
        pw = np.concatenate(all_pw, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        flat_probs = probs[masks]
        profiles: Dict[str, pd.DataFrame] = {}

        for stream_name, arr, names in [
            ("scalar", sc, self.scalar_names),
            ("local", lo, self.local_names),
            ("pairwise", pw, self.pairwise_names),
        ]:
            if arr.ndim == 4:
                arr = arr.mean(axis=2)

            rows = []
            for f_idx, fname in enumerate(names):
                flat_feat = arr[:, :, f_idx][masks]
                quantiles = np.quantile(flat_feat, np.linspace(0, 1, n_bins + 1))
                quantiles = np.unique(quantiles)  # handle degenerate distributions

                for q_lo, q_hi in zip(quantiles[:-1], quantiles[1:]):
                    in_bin = (flat_feat >= q_lo) & (flat_feat <= q_hi)
                    if in_bin.sum() == 0:
                        continue
                    rows.append(
                        {
                            "stream": stream_name,
                            "feature": fname,
                            "bin_low": float(q_lo),
                            "bin_high": float(q_hi),
                            "bin_center": float((q_lo + q_hi) / 2),
                            "mean_LIP_prob": float(flat_probs[in_bin].mean()),
                            "n_residues": int(in_bin.sum()),
                        }
                    )

            profiles[stream_name] = pd.DataFrame(rows)

        return AttributionResult(
            method=self.method_name,
            protein_ids=all_ids,
            attributions={
                "scalar": np.concatenate(all_sc, axis=0),
                "local": np.concatenate(all_lo, axis=0),
                "pairwise": np.concatenate(all_pw, axis=0),
            },
            feature_names=self.feature_names,
            masks=masks,
            metadata={"profiles": profiles},
        )

    def get_profiles(self, result: AttributionResult) -> pd.DataFrame:
        """Concatenate all stream profiles into one DataFrame."""
        return pd.concat(result.metadata["profiles"].values(), ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience: run all (or a subset) of registered interpreters
# ---------------------------------------------------------------------------


def run_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scalar_names: List[str],
    local_names: List[str],
    pairwise_names: List[str],
    methods: Optional[List[str]] = None,
    method_kwargs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, AttributionResult]:
    """
    Run a set of interpreters over *loader* and return all results.

    Parameters
    ----------
    model           : trained model in eval mode
    loader          : DataLoader (same collate_fn as training/inference)
    device          : torch device
    scalar_names    : list of scalar feature names
    local_names     : list of local  feature names
    pairwise_names  : list of pairwise feature names
    methods         : list of method names to run; defaults to all registered
    method_kwargs   : dict of {method_name: {kwarg: value}} for per-method params

    Returns
    -------
    dict mapping method_name → AttributionResult
    """
    methods = methods or list(REGISTRY.keys())
    method_kwargs = method_kwargs or {}
    results: Dict[str, AttributionResult] = {}

    for name in methods:
        if name not in REGISTRY:
            warnings.warn(f"Unknown method '{name}' — skipping.")
            continue
        print(f"[lip_interpretability] Running: {name} ...", flush=True)
        cls = REGISTRY[name]
        interpreter = cls(
            model=model,
            scalar_names=scalar_names,
            local_names=local_names,
            pairwise_names=pairwise_names,
            device=device,
        )
        try:
            results[name] = interpreter.run(loader, **method_kwargs.get(name, {}))
            print(f"[lip_interpretability]   ✓ {name} done")
        except Exception as e:
            warnings.warn(f"[lip_interpretability] {name} failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Summary report helper
# ---------------------------------------------------------------------------


def summary_report(results: Dict[str, AttributionResult]) -> pd.DataFrame:
    """
    Build a single ranked feature-importance DataFrame from all available
    attribution results.  Scores are min-max normalised per method then
    averaged for a consensus ranking.

    Returns a DataFrame with columns:
        stream, feature, <method1>_score, <method2>_score, ..., consensus_score
    """
    dfs = {}
    for name, res in results.items():
        if name in (
            "feature_value_correlation",
            "feature_range_profiler",
            "attention_rollout",
        ):
            continue  # non-gradient methods handled separately
        df = res.mean_per_feature()[["stream", "feature", "mean_abs_attr"]]
        df = df.rename(columns={"mean_abs_attr": name})
        dfs[name] = df.set_index(["stream", "feature"])

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs.values(), axis=1)

    # Normalise each method column 0→1
    for col in combined.columns:
        col_range = combined[col].max() - combined[col].min()
        if col_range > 0:
            combined[col] = (combined[col] - combined[col].min()) / col_range

    combined["consensus_score"] = combined.mean(axis=1)
    return combined.sort_values("consensus_score", ascending=False).reset_index()
