# CORE-LIP

**CO**nformational **R**epresentation **E**nsemble for **L**inear **I**nteraction **P**eptide prediction

CORE-LIP predicts LIP propensity from an ensemble of protein conformations.
It takes MD trajectories (or IDPFold2 backmapped predictions) as input, computes
structural and dynamical features, and feeds them into a multi-scale Transformer
(`ProteinMultiScaleTransformer`) that jointly exploits sequence, per-residue local,
global scalar, and pairwise conformational signals.

---

## Project layout

```
CORE-LIP/
├── core_lip/                  # Installable Python package
│   ├── __init__.py
│   ├── model.py               # ProteinMultiScaleTransformer + ProteinModelConfig
│   ├── datasets.py            # Dataset classes and collate functions
│   └── utils.py               # I/O helpers and feature extraction
│
├── scripts/                   # End-to-end pipeline (run in order)
│   ├── compute_properties.py  # 1 – Extract MD features → HDF5
│   ├── visualize_features.py  # 2 – Compare LIP / non-LIP feature distributions
│   ├── train.py               # 3 – Train the model
│   ├── predict.py             # 4 – Generate per-residue predictions
│   ├── evaluate.py            # 5 – Benchmark against baselines
│   └── filter_dataset.py      #     Utility – filter dataset to available proteins
│
├── data/
│   ├── CLIP_dataset/          # TR1000.txt, TE440.txt (and *_reduced.txt after filtering)
│   ├── conformational_ensemble/  # Per-protein folders with DCD/PDB trajectories
│   ├── protein_MD_properties.h5  # Output of compute_properties.py
│   ├── models/                # Saved model checkpoints
│   └── predictions/           # CSV files from predict.py
│
├── results/                   # Figures and evaluation outputs
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # installs core_lip as an editable package
```

---

## Quick start

### 1 — Compute conformational properties

```bash
python scripts/compute_properties.py \
    --input_dir  data/conformational_ensemble/ \
    --output_h5  data/protein_MD_properties.h5 \
    --workers    15
```

### 2 — Filter the dataset

Keep only proteins for which features were successfully computed:

```bash
python scripts/filter_dataset.py \
    --h5         data/protein_MD_properties.h5 \
    --train_in   data/CLIP_dataset/TR1000.txt \
    --train_out  data/CLIP_dataset/TR1000_reduced.txt \
    --test_in    data/CLIP_dataset/TE440.txt \
    --test_out   data/CLIP_dataset/TE440_reduced.txt
```

### 3 — Visualise feature distributions

```bash
python scripts/visualize_features.py \
    --dataset  data/CLIP_dataset/TR1000_reduced.txt \
    --h5       data/protein_MD_properties.h5 \
    --output   results/feature_comparison_violin.pdf
```

### 4 — Train the model

```bash
python scripts/train.py \
    --dataset  data/CLIP_dataset/TR1000_reduced.txt \
    --h5       data/protein_MD_properties.h5 \
    --model    data/models/core_lip.pt \
    --epochs   250
```

### 5 — Make predictions

```bash
python scripts/predict.py \
    --model     data/models/core_lip.pt \
    --h5        data/protein_MD_properties.h5 \
    --datasets  data/CLIP_dataset/TR1000_reduced.txt \
               data/CLIP_dataset/TE440_reduced.txt \
    --output_dir data/predictions/
```

### 6 — Evaluate

```bash
python scripts/evaluate.py \
    --train_truth  data/CLIP_dataset/TR1000_reduced.txt \
    --train_preds  data/predictions/core_lip_TR1000_reduced.csv \
    --test_truth   data/CLIP_dataset/TE440_reduced.txt \
    --test_preds   data/predictions/core_lip_TE440_reduced.csv \
    --output_dir   results/
```

To add a comparison model (e.g. CLIP), pass `--clip_preds data/predictions/CLIP_TE440.txt`.
Additional baselines can be declared inside `evaluate.py` using the `ComparisonModel` dataclass.

---

## Model architecture

`ProteinMultiScaleTransformer` fuses four input streams:

| Stream | Shape | Encoding |
|---|---|---|
| Amino-acid sequence | `[B, L]` | Learned embedding + sinusoidal positional encoding |
| Local (per-residue) | `[B, nb_local, L]` | 2-layer MLP projection |
| Scalar (per-protein) | `[B, nb_scalar]` | 2-layer MLP → broadcast to `[B, L, E]` |
| Pairwise | `[B, nb_pairwise, L, L]` | Windowed row extraction + MLP |

These are summed into a single `[B, L, E]` embedding and passed through
`num_blocks` Transformer blocks.  Each block contains:
1. **PairwiseCNN** — multi-scale dilated 2D CNN producing per-head attention bias
2. **BiasedMultiHeadAttention** — standard MHA with learnable pairwise gating
3. **FeedForwardNetwork** — position-wise FFN with GELU

After the last block, masked mean pooling collapses the sequence dimension and
a linear classification head outputs class logits.

---

## Features

### Scalar (per-protein, from MD ensemble)
Asphericity, radius of gyration, end-to-end distance, shape anisotropy metrics,
gyration eigenvalues, scaling exponent, etc.

### Local (per-residue)
φ/ψ dihedral entropies, absolute/relative SASA (mean & std), secondary
structure propensities (H, G, I, E, B, T, S, C).

### Pairwise
Dynamic cross-correlation matrix (DCCM), ensemble-averaged contact map,
distance fluctuation matrix.

---

## Dependencies

See `requirements.txt`. Key libraries: `torch`, `h5py`, `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `scipy`, `mdtraj`, `tqdm`.
The `idpmdp` package (for `ProteinAnalyzer`) is required for Step 1 only.