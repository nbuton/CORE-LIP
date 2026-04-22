"""
CORE-LIP: COformational Representation Ensemble for LIP prediction
===================================================================
ProteinMultiScaleTransformer — the main model.

Integrates:
  - Sequence embeddings + sinusoidal positional encodings
  - Per-residue local features  (x_local)
  - Per-protein scalar features (x_scalar)
  - Pairwise residue features   (x_pairwise)

through a series of CNN-biased Transformer blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProteinModelConfig:
    """Central configuration for all model hyperparameters."""

    # ── Vocabulary / sequence ──────────────────────────────────────────────
    vocab_size: int = 25  # number of amino-acid tokens (incl. special)
    pad_token_id: int = 0

    # ── Feature dimensions ────────────────────────────────────────────────
    nb_scalar: int = 16  # number of scalar (per-protein) features
    nb_local: int = 32  # number of local  (per-residue) features
    nb_pairwise: int = 8  # number of pairwise feature channels

    # ── Embedding / model width ───────────────────────────────────────────
    embed_dim: int = 128  # E  — main hidden dimension
    max_seq_len: int = 1024  # maximum protein length for positional emb

    # ── Transformer blocks ────────────────────────────────────────────────
    num_blocks: int = 4  # k  — number of Transformer blocks
    num_heads: int = 8  # attention heads (embed_dim % num_heads == 0)
    ffn_expansion: int = 2  # FFN hidden = embed_dim * ffn_expansion
    dropout: float = 0.1

    # ── Pairwise CNN (inside each block) ──────────────────────────────────
    pairwise_cnn_channels: int = 32  # intermediate CNN channels
    pairwise_cnn_kernel: int = 3  # spatial kernel for pairwise CNN
    dilatations_cnn: tuple[int, ...] = (1, 2, 3)

    # ── Classification head ───────────────────────────────────────────────
    num_classes: int = 2

    # ── MLP hidden sizes (defaults derived from embed_dim) ────────────────
    local_mlp_hidden: int = field(default=-1)  # -1 → embed_dim
    scalar_mlp_hidden: int = field(default=-1)  # -1 → embed_dim

    def __post_init__(self):
        if self.local_mlp_hidden < 0:
            self.local_mlp_hidden = self.embed_dim
        if self.scalar_mlp_hidden < 0:
            self.scalar_mlp_hidden = self.embed_dim
        assert self.embed_dim % 2 == 0, "embed_dim must be even (pairwise windowing)"
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"


# ---------------------------------------------------------------------------
# 1.  Small reusable primitives
# ---------------------------------------------------------------------------


class MLP2(nn.Module):
    """Two-layer MLP: in_dim → hidden_dim → out_dim with ReLU + optional dropout."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN: E → 2E → E (as used inside each Transformer block).
    Wraps with LayerNorm + residual.
    """

    def __init__(self, embed_dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = embed_dim * expansion
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, E]  →  [B, L, E]"""
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# 2.  Input embedding components
# ---------------------------------------------------------------------------


class SequenceEmbedding(nn.Module):
    """
    Learned token embedding + sinusoidal positional encoding.
    Returns [B, L, E].
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, max_len: int, dropout: float = 0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute sinusoidal positional encoding (not learned)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, E]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] (int)  →  [B, L, E]"""
        L = tokens.size(1)
        x = self.token_emb(tokens) + self.pe[:, :L]
        return self.dropout(x)


class LocalFeatureProjector(nn.Module):
    """
    Projects per-residue local features to embedding space.
    x_local [B, nb_local, L]  →  [B, L, E]
    """

    def __init__(
        self, nb_local: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.mlp = MLP2(nb_local, hidden_dim, embed_dim, dropout)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        """x_local: [B, nb_local, L]  →  [B, L, E]"""
        x = x_local.permute(0, 2, 1)  # [B, L, nb_local]
        return self.mlp(x)  # [B, L, E]


class ScalarFeatureProjector(nn.Module):
    """
    Projects per-protein scalar features, then broadcasts to all residues.
    x_scalar [B, nb_scalar]  →  [B, L, E]
    """

    def __init__(
        self, nb_scalar: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.mlp = MLP2(nb_scalar, hidden_dim, embed_dim, dropout)

    def forward(self, x_scalar: torch.Tensor, L: int) -> torch.Tensor:
        """x_scalar: [B, nb_scalar]  →  [B, L, E]"""
        protein_repr = self.mlp(x_scalar)  # [B, E]
        return protein_repr.unsqueeze(1).expand(-1, L, -1)  # [B, L, E]


class PairwiseContextProjector(nn.Module):
    """
    Converts pairwise features into per-residue context and adds to embedding.

    For each residue i, gathers a window of E//2 neighbours on each side from
    the pairwise matrix (padding at boundaries), then projects the flattened
    window to the embedding dimension with a 2-layer MLP.
    """

    def __init__(self, nb_pairwise: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.E = embed_dim
        self.half = embed_dim // 2
        self.window = embed_dim  # total window width = E
        in_dim = nb_pairwise * (self.window + 1)  # +1 for the centre residue
        self.mlp = MLP2(in_dim, embed_dim, embed_dim, dropout)

    def _extract_windows(self, x_pairwise: torch.Tensor) -> torch.Tensor:
        """
        x_pairwise: [B, C, L, L]
        Returns   : [B, L, C, window+1]
        """
        B, C, L, _ = x_pairwise.shape
        half = self.half
        window = self.window
        padded = F.pad(x_pairwise, (half, half), mode="constant", value=0.0)
        flat2 = padded.reshape(B * C, L, L + window)
        slices2 = flat2.unfold(2, window + 1, 1)  # [BC, L, L, window+1]
        row_idx = torch.arange(L, device=x_pairwise.device)
        row_idx_exp = row_idx.view(1, L, 1, 1).expand(B * C, -1, 1, window + 1)
        windows_diag = slices2.gather(2, row_idx_exp).squeeze(2)
        windows_diag = windows_diag.reshape(B, C, L, window + 1)
        return windows_diag.permute(0, 2, 1, 3)  # [B, L, C, window+1]

    def forward(self, x_pairwise: torch.Tensor) -> torch.Tensor:
        """x_pairwise: [B, C, L, L]  →  [B, L, E]"""
        B, C, L, _ = x_pairwise.shape
        windows = self._extract_windows(x_pairwise)  # [B, L, C, window+1]
        flat = windows.reshape(B, L, C * (self.window + 1))  # [B, L, C*(E+1)]
        return self.mlp(flat)  # [B, L, E]


# ---------------------------------------------------------------------------
# 3.  Transformer block sub-components
# ---------------------------------------------------------------------------


def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the largest valid number of groups ≤ max_groups."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class PairwiseCNN(nn.Module):
    """
    Pairwise feature extractor for protein contact/distance maps.

    Input : [B, C_in, L, L]
    Output: [B, num_heads, L, L]

    Pipeline:
      1) Depthwise conv — independent spatial processing per input channel
      2) Three parallel dilated conv branches (dilations 1, 2, 3)
      3) Concatenate + GroupNorm + GELU
      4) 1×1 conv to num_heads
    """

    def __init__(
        self,
        nb_pairwise: int,
        cnn_channels: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve L×L shape.")

        pad = kernel_size // 2
        self.dilations = dilations

        # Stage 1: independent spatial processing per channel
        self.depthwise = nn.Conv2d(
            nb_pairwise,
            nb_pairwise,
            kernel_size=kernel_size,
            padding=pad,
            groups=nb_pairwise,
            bias=False,
        )
        self.depthwise_norm = _make_group_norm(nb_pairwise)
        self.depthwise_act = nn.GELU()

        # Stage 2: three dilated branches for multi-scale mixing
        branches = []
        for d in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        nb_pairwise,
                        cnn_channels,
                        kernel_size=kernel_size,
                        padding=d * pad,
                        dilation=d,
                        bias=False,
                    ),
                    _make_group_norm(cnn_channels),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)

        merged_channels = cnn_channels * len(dilations)
        self.post_norm = _make_group_norm(merged_channels)
        self.post_act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Final projection to attention heads
        self.to_heads = nn.Conv2d(merged_channels, num_heads, kernel_size=1, bias=True)

    def forward(self, x_pairwise: torch.Tensor) -> torch.Tensor:
        """x_pairwise: [B, nb_pairwise, L, L]  →  [B, num_heads, L, L]"""
        x = self.depthwise_act(self.depthwise_norm(self.depthwise(x_pairwise)))
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.post_act(self.post_norm(x))
        x = self.dropout(x)
        return self.to_heads(x)


class BiasedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with an additive per-head pairwise bias.

    Each head has a learnable scalar gate (α_h) controlling how much the
    pairwise bias contributes:
        logits_h = QKᵀ / √d + α_h · bias_h

    x:    [B, L, E]
    bias: [B, num_heads, L, L]
    mask: [B, L]  (1 = real residue, 0 = padding)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # One learnable gate per head
        self.bias_gate = nn.Parameter(torch.zeros(num_heads))

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, E = x.shape
        H, D = self.num_heads, self.head_dim

        residual = x
        x = self.norm(x)

        def _proj_reshape(proj, t):
            return proj(t).reshape(B, L, H, D).transpose(1, 2)

        Q = _proj_reshape(self.q_proj, x)
        K = _proj_reshape(self.k_proj, x)
        V = _proj_reshape(self.v_proj, x)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.bias_gate.view(1, H, 1, 1) * bias

        if mask is not None:
            pad_mask = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            attn_logits = attn_logits + pad_mask

        attn_weights = self.attn_dropout(torch.softmax(attn_logits, dim=-1))

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, L, E)
        return residual + self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    One full Transformer block:
        1. PairwiseCNN       — refines x_pairwise, produces per-head attention bias
        2. BiasedMHA         — self-attention with the pairwise bias
        3. FeedForwardNetwork — position-wise FFN (E → 2E → E)
    """

    def __init__(self, cfg: ProteinModelConfig):
        super().__init__()
        self.pairwise_cnn = PairwiseCNN(
            nb_pairwise=cfg.nb_pairwise,
            cnn_channels=cfg.pairwise_cnn_channels,
            num_heads=cfg.num_heads,
            kernel_size=cfg.pairwise_cnn_kernel,
            dilations=cfg.dilatations_cnn,
            dropout=cfg.dropout,
        )
        self.attention = BiasedMultiHeadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.ffn = FeedForwardNetwork(
            embed_dim=cfg.embed_dim,
            expansion=cfg.ffn_expansion,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_pairwise: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        x:          [B, L, E]
        x_pairwise: [B, nb_pairwise, L, L]
        mask:       [B, L]
        Returns:    (x, x_pairwise) same shapes as input
        """
        attn_bias = self.pairwise_cnn(x_pairwise)  # [B, H, L, L]
        x = self.attention(x, attn_bias, mask)
        x = self.ffn(x)
        return x, x_pairwise


# ---------------------------------------------------------------------------
# 4.  Pooling + classification head
# ---------------------------------------------------------------------------


class MaskedMeanPool(nn.Module):
    """Average-pools [B, L, E] → [B, E] ignoring padded positions."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        return x.sum(dim=1) / denom  # [B, E]


class ClassificationHead(nn.Module):
    """Maps pooled [B, E] representation to class logits."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 5.  Main model
# ---------------------------------------------------------------------------


class ProteinMultiScaleTransformer(nn.Module):
    """
    CORE-LIP: multi-scale protein representation model for LIP prediction.

    Inputs
    ------
    tokens    : [B, L]                  amino-acid token IDs
    x_scalar  : [B, nb_scalar]          per-protein global conformational features
    x_local   : [B, nb_local, L]        per-residue local features
    x_pairwise: [B, nb_pairwise, L, L]  pairwise residue features
    mask      : [B, L]                  1 for real residue, 0 for padding

    Output
    ------
    logits    : [B, num_classes]
    """

    def __init__(self, cfg: ProteinModelConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.embed_dim

        # ── Input embeddings ──────────────────────────────────────────────
        self.seq_emb = SequenceEmbedding(
            cfg.vocab_size, E, cfg.max_seq_len, cfg.dropout
        )
        self.local_proj = LocalFeatureProjector(
            cfg.nb_local, E, cfg.local_mlp_hidden, cfg.dropout
        )
        self.scalar_proj = ScalarFeatureProjector(
            cfg.nb_scalar, E, cfg.scalar_mlp_hidden, cfg.dropout
        )
        self.pairwise_init_proj = PairwiseContextProjector(
            cfg.nb_pairwise, E, cfg.dropout
        )
        self.embed_norm = nn.LayerNorm(E)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_blocks)]
        )

        # ── Pooling + head ────────────────────────────────────────────────
        self.pool = MaskedMeanPool()
        self.head = ClassificationHead(E, cfg.num_classes, cfg.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        x_scalar: torch.Tensor,
        x_local: torch.Tensor,
        x_pairwise: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = tokens.shape

        # 1. Build initial [B, L, E] embedding
        x = self.seq_emb(tokens)  # sequence + positional
        x = x + self.local_proj(x_local)  # add per-residue local
        x = x + self.scalar_proj(x_scalar, L)  # add broadcast global scalar
        x = x + self.pairwise_init_proj(x_pairwise)  # add pairwise context
        x = self.embed_norm(x)

        # 2. Transformer blocks
        for block in self.blocks:
            x, x_pairwise = block(x, x_pairwise, mask)

        # 3. Masked mean pooling → [B, E]
        x = self.pool(x, mask)

        # 4. Classification head
        return self.head(x)  # [B, num_classes]


# ---------------------------------------------------------------------------
# 6.  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ProteinModelConfig(
        vocab_size=25,
        nb_scalar=16,
        nb_local=32,
        nb_pairwise=8,
        embed_dim=64,
        num_blocks=2,
        num_heads=4,
        max_seq_len=512,
    )
    model = ProteinMultiScaleTransformer(cfg)

    B, L = 2, 128
    tokens = torch.randint(1, 25, (B, L))
    x_scalar = torch.randn(B, cfg.nb_scalar)
    x_local = torch.randn(B, cfg.nb_local, L)
    x_pairwise = torch.randn(B, cfg.nb_pairwise, L, L)
    mask = torch.ones(B, L)
    mask[0, 40:] = 0

    logits = model(tokens, x_scalar, x_local, x_pairwise, mask)
    print("Output shape:", logits.shape)  # expect [2, 2]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
