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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_lip.config import ProteinModelConfig

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
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int,
        dropout: float = 0.1,
        only_pos_embedding: bool = False,
    ):
        super().__init__()
        self.only_pos_embedding = only_pos_embedding
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
        if self.only_pos_embedding:
            x = self.pe[:, :L]
        else:
            x = self.token_emb(tokens) + self.pe[:, :L]
        return self.dropout(x)


class LocalFeatureProjector(nn.Module):
    """
    Projects per-residue local features to embedding space with learned scaling.
    x_local [B, nb_local, L]  →  [B, L, E]
    """

    def __init__(
        self,
        nb_local: int,
        embed_dim: int,
        hidden_dim: int,
        means: torch.Tensor,
        stds: torch.Tensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1. Initialize the normalization layer with local feature stats
        self.scaler = LearnedScalarNorm(
            nb_local, initial_means=means, initial_stds=stds
        )

        # 2. Project the normalized features
        self.mlp = MLP2(nb_local, hidden_dim, embed_dim, dropout)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        """x_local: [B, nb_local, L]  →  [B, L, E]"""
        # Step 1: Move features to the last dimension [B, L, nb_local]
        x = x_local.permute(0, 2, 1)

        # Step 2: Apply learned scaling per feature
        x_scaled = self.scaler(x)

        # Step 3: Project to embedding space
        return self.mlp(x_scaled)  # [B, L, E]


class LearnedScalarNorm(nn.Module):
    def __init__(
        self,
        nb_scalar: int,
        initial_means: torch.Tensor = None,
        initial_stds: torch.Tensor = None,
    ):
        super().__init__()
        # Initialize shift (mu) and scale (sigma)
        # If no stats provided, start at 0 and 1
        if initial_means is None:
            initial_means = torch.zeros(nb_scalar)
        if initial_stds is None:
            initial_stds = torch.ones(nb_scalar)

        # We use nn.Parameter so the optimizer can update them
        self.mean = nn.Parameter(initial_means)
        self.log_std = nn.Parameter(torch.log(initial_stds + 1e-6))

    def forward(self, x):
        # We use exp(log_std) to ensure the standard deviation stays positive
        return (x - self.mean) / (torch.exp(self.log_std) + 1e-6)


class ScalarFeatureProjector(nn.Module):
    def __init__(
        self,
        nb_scalar: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        means: torch.Tensor,
        stds: torch.Tensor,
    ):
        super().__init__()
        # 1. The Scaling Layer (Independent scaling per feature)
        self.scaler = LearnedScalarNorm(
            nb_scalar, initial_means=means, initial_stds=stds
        )

        # 2. The Projection Layer
        self.mlp = MLP2(nb_scalar, hidden_dim, embed_dim, dropout=dropout)

    def forward(self, x_scalar: torch.Tensor, L: int) -> torch.Tensor:
        # Step 1: Scale raw values
        x_scaled = self.scaler(x_scalar)
        # Step 2: Project
        protein_repr = self.mlp(x_scaled)
        return protein_repr.unsqueeze(1).expand(-1, L, -1)


class PairwiseContextProjector(nn.Module):
    """
    Converts pairwise features into per-residue context and adds to embedding.
    For each residue i, gathers a window of window_size//2 neighbours on each side
    from the pairwise matrix (padding at boundaries), then projects the flattened
    window to the embedding dimension with a 2-layer MLP.
    """

    def __init__(
        self,
        nb_pairwise: int,
        embed_dim: int,
        dropout: float = 0.1,
        window_size: int = 1024,
    ):
        super().__init__()
        self.E = embed_dim
        self.half = window_size // 2
        self.window = window_size  # fixed window, independent of embed_dim

        in_dim = nb_pairwise * (self.window + 1)
        self.mlp = MLP2(in_dim, embed_dim, embed_dim, dropout)

    def _extract_windows(self, x_pairwise: torch.Tensor) -> torch.Tensor:
        """
        x_pairwise: [B, C, L, L]
        Returns   : [B, L, C, window+1]
        """
        B, C, L, _ = x_pairwise.shape
        half = self.half
        window = self.window

        # 1. Pad the last dimension
        padded = F.pad(x_pairwise, (half, half), mode="constant", value=0.0)

        # 2. Flatten the last two dimensions
        padded_flat = padded.view(B, C, -1)

        # 3. Extract the diagonal windows using a single strided unfold
        # FIX: PyTorch Tensor.unfold only accepts positional arguments
        windows_diag = padded_flat.unfold(-1, window + 1, L + window + 1)

        # 4. Permute to the target shape and ensure it is contiguous in memory
        return windows_diag.permute(0, 2, 1, 3).contiguous()

    def forward(self, x_pairwise: torch.Tensor) -> torch.Tensor:
        """x_pairwise: [B, C, L, L]  →  [B, L, E]"""
        B, C, L, _ = x_pairwise.shape

        windows = self._extract_windows(x_pairwise)  # [B, L, C, window+1]
        flat = windows.reshape(B, L, C * (self.window + 1))
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        activate_bias: bool = True,
    ):
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
        self.bias_gate = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.activate_bias = activate_bias

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
        if self.activate_bias:
            attn_logits = attn_logits + self.bias_gate.view(1, H, 1, 1) * bias

        if mask is not None:
            # Mask padding key positions: [B, 1, 1, L]
            key_mask = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
            attn_logits = attn_logits + key_mask

        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Zero out nan from fully-padded rows and padding query positions
        if mask is not None:
            query_mask = mask.float().unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            attn_weights = attn_weights.nan_to_num(0.0) * query_mask

        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, L, E)

        # Zero padding positions in output before residual
        if mask is not None:
            out = out * mask.float().unsqueeze(-1)  # [B, L, 1]

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
        self.activate_pairwise_bias = cfg.activate_pairwise_bias
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
            activate_bias=cfg.activate_pairwise_bias,
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
        if self.activate_pairwise_bias:
            attn_bias = self.pairwise_cnn(x_pairwise)  # [B, H, L, L]
        else:
            attn_bias = None
        x = self.attention(x, attn_bias, mask)
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# 4.  Pooling + classification head
# ---------------------------------------------------------------------------


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
    """

    def __init__(self, cfg: ProteinModelConfig, stats):
        super().__init__()
        self.cfg = cfg
        self.E = cfg.embed_dim
        self.inputs_features = cfg.inputs_features

        # ── Input embeddings ──────────────────────────────────────────────
        only_pos_embedding = "seq_emb" not in self.inputs_features
        self.seq_emb = SequenceEmbedding(
            cfg.vocab_size, self.E, cfg.max_seq_len, cfg.dropout, only_pos_embedding
        )
        # Assuming cfg contains 'plm_dim' (e.g., 1280 for ESM-2 or 1536 for ESM-3)
        if "plm_embedding" in self.inputs_features:
            self.plm_proj = nn.Sequential(
                nn.Linear(cfg.plm_dim, self.E),  # Try adding L1 on these weights?
                nn.Dropout(0.6),
                nn.GELU(),
                nn.Linear(self.E, self.E),
                nn.Dropout(cfg.dropout),
            )

        self.scalar_proj = ScalarFeatureProjector(
            cfg.nb_scalar,
            self.E,
            cfg.scalar_mlp_hidden,
            cfg.dropout,
            stats["scalar"]["means"],
            stats["scalar"]["stds"],
        )
        self.local_proj = LocalFeatureProjector(
            cfg.nb_local,
            self.E,
            cfg.local_mlp_hidden,
            stats["local"]["means"],
            stats["local"]["stds"],
            cfg.dropout,
        )
        self.pairwise_init_proj = PairwiseContextProjector(
            cfg.nb_pairwise,
            self.E,
            cfg.dropout,
            cfg.window_size_pairwise_input,
        )
        self.pair_wise_scaler = LearnedScalarNorm(
            cfg.nb_pairwise,
            initial_means=stats["pairwise"]["means"],
            initial_stds=stats["pairwise"]["stds"],
        )
        self.embed_norm = nn.LayerNorm(self.E)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_blocks)]
        )

        # ── Pooling + head ────────────────────────────────────────────────
        self.head = ClassificationHead(self.E, cfg.num_classes, cfg.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        x_scalar: torch.Tensor,
        x_local: torch.Tensor,
        x_pairwise: torch.Tensor,
        mask: torch.Tensor,
        plm_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = tokens.shape
        if mask is not None:
            m = mask.float()
            pairwise_mask = m.unsqueeze(1).unsqueeze(-1) * m.unsqueeze(1).unsqueeze(2)
            x_pairwise = x_pairwise * pairwise_mask

        x_pairwise_permute = x_pairwise.permute(0, 2, 3, 1)  # [B, L, L, C]
        x_pairwise_permute_scaled = self.pair_wise_scaler(x_pairwise_permute)
        x_pairwise_scaled = x_pairwise_permute_scaled.permute(
            0, 3, 1, 2
        )  # [B, C, L, L] back for extraction

        x = torch.zeros((B, L, self.E), device=tokens.device)
        # 1. Build initial [B, L, E] embedding
        if "token_embedding" in self.inputs_features:
            x = self.seq_emb(tokens)  # sequence + positional

        if "scalar_features" in self.inputs_features:
            x = x + self.scalar_proj(x_scalar, L)

        if "local_features" in self.inputs_features:
            x = x + self.local_proj(x_local)

        if "pairwise_features" in self.inputs_features:
            x = x + self.pairwise_init_proj(x_pairwise_scaled)

        if "plm_embedding" in self.inputs_features:
            x = x + self.plm_proj(plm_pad)

        x = self.embed_norm(x)

        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x, x_pairwise_scaled, mask)

        # 3. Classification head
        return self.head(x)


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
    plm_pad = None

    logits = model(tokens, x_scalar, x_local, x_pairwise, mask, plm_pad)
    print("Output shape:", logits.shape)  # expect [2, 2]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
