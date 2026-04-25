from pydantic import BaseModel, model_validator
from pathlib import Path
from typing import List, Tuple
from typing import Dict, Any


class ProteinModelConfig(BaseModel):
    """Central configuration for all model hyperparameters."""

    # ── Vocabulary / sequence ──────────────────────────────────────────────
    vocab_size: int = 25
    pad_token_id: int = 0

    # ── Feature dimensions ────────────────────────────────────────────────
    nb_scalar: int = 16
    nb_local: int = 32
    nb_pairwise: int = 8
    inputs_features: List = [
        "seq_emb",
        "scalar_features",
        "local_features",
        "pairwise_features",
    ]

    # ── Embedding / model width ───────────────────────────────────────────
    embed_dim: int = 128
    max_seq_len: int = 1024
    window_size_pairwise_input: int = 1024
    activate_pairwise_bias: bool = True

    # ── Transformer blocks ────────────────────────────────────────────────
    num_blocks: int = 4
    num_heads: int = 8
    ffn_expansion: int = 2
    dropout: float = 0.1

    # ── Pairwise CNN (inside each block) ──────────────────────────────────
    pairwise_cnn_channels: int = 32
    pairwise_cnn_kernel: int = 3
    dilatations_cnn: Tuple[int, ...] = (1, 2, 3)

    # ── Classification head ───────────────────────────────────────────────
    num_classes: int = 1

    # ── MLP hidden sizes (defaults derived from embed_dim) ────────────────
    local_mlp_hidden: int = -1
    scalar_mlp_hidden: int = -1

    # Add pre trainned embeddings
    plm_dim: int = 6144

    # ── Post-Initialization & Validation ──────────────────────────────────
    @model_validator(mode="after")
    def validate_and_set_defaults(self) -> "ProteinModelConfig":
        # 1. Set dynamic defaults
        if self.local_mlp_hidden < 0:
            self.local_mlp_hidden = self.embed_dim
        if self.scalar_mlp_hidden < 0:
            self.scalar_mlp_hidden = self.embed_dim

        # 2. Replicate your assertions as proper Pydantic validations
        if self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (pairwise windowing)")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        return self


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    accumulation: int
    loss_type: str
    loss_params: Dict[str, Any] = {}
    val_prop: float
    lr: float
    weight_decay: float
    seed: int
    h5_properties: Path
    training_dataset: Path


# You can now hook this up to your main config just like before:
class FullConfig(BaseModel):
    training: TrainingConfig  # (From the previous example)
    model: ProteinModelConfig
