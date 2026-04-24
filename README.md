# CORE-LIP

**CO**nformational **R**epresentation **E**nsemble for **L**inear **I**nteraction **P**eptide prediction

CORE-LIP predicts LIP propensity from an ensemble of protein conformations.
It takes full atom ensemble of conformation as input, computes
structural and dynamical features, and feeds them into a multi-scale Transformer
(`ProteinMultiScaleTransformer`) that jointly exploits sequence, per-residue local,
global scalar, and pairwise conformational signals.

---

## Project layout

```
CORE-LIP/
├── core_lip/                         # Core Library
│   ├── config.py                     # Pydantic configuration schemas
│   ├── data/                         # Data handling & Pre-processing
│   │   ├── datasets.py               # PyTorch Dataset & Collation
│   │   ├── io.py                     # I/O, Parsing (Truth/CSV)
│   │   ├── properties_extraction.py  # Feature extraction
│   │   └── features.py               # Canonical feature name definitions
│   ├── modeling/                     # Neural Network Architecture
│   │   ├── transformer.py            # ProteinMultiScaleTransformer
│   │   └── loss.py                   # FocalLoss implementation
│   ├── engine/                       # Execution Logic
│   │   ├── trainer.py                # Training loops & gradient utilities
│   │   └── predictor.py              # Inference wrappers & checkpoint loading
│   └── eval/                         # Statistics & Visualization
│       ├── metrics.py                # AUC, MCC, and thresholding logic
│       ├── plotting.py               # Publication-ready figure generation
│       └── structures.py             # ResidueExample tracking objects
│
├── scripts/                          # End-to-end pipeline (run in order)
│   ├── compute_properties.py         # 1 – Extract conformational structure set features → HDF5
│   ├── visualize_features.py         # 2 – Compare LIP / non-LIP feature distributions
│   ├── train.py                      # 3 – Train the model
│   ├── predict.py                    # 4 – Generate per-residue predictions
│   └── evaluate.py                   # 5 – Benchmark against previous other models
│
├── data/
│   ├── CLIP_dataset/          # TR1000.txt, TE440.txt (and *_max_1024.txt after filtering)
│   ├── conformational_ensemble/  # Per-protein folders with DCD/PDB trajectories
│   ├── properties/  # Output of compute_properties.py
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
git clone https://github.com/nbuton/CORE-LIP
pip install -r requirements.txt
pip install -e .          # installs core_lip as an editable package
```

---

##  Using CORE-LIP as a Prediction Tool

We provide two ways to run CORE-LIP: a "zero-setup" cloud version and a high-throughput local version.

### Option A: Google Colab (Recommended for single sequences)
For a quick prediction without local installation, use our managed notebook. It handles environment setup and dependencies automatically.
> [!TIP]
> **[Open in Colab](https://colab.research.google.com/github/xxxxx)** — Input your sequence and click **Runtime > Run All**.

---

### Option B: Local Production Pipeline
Use this for large-scale datasets or custom workflows. This pipeline requires pre-generated conformational ensembles.

#### 1. Generate Conformational Ensembles
CORE-LIP predicts based on structural dynamics. You must first generate full-atom ensembles:
* **Fold:** Generate coarse-grained ensembles using [IDPFold2](https://github.com/Junjie-Zhu/IDPFold2).
* **Backmap:** Convert to full-atom resolution using `cg2all` (refer to the script in the IDPFold2 repository).
* **Organize:** Place the output in `data/conformational_ensemble/IDPFold2/`. 

**Required Directory Structure:**
```text
data/conformational_ensemble/IDPFold2/
└── [Protein_ID]/
    ├── top_AA.pdb   # All-atom topology
    └── traj_AA.xtc  # All-atom trajectory
```

#### 2. Run Production Inference
Once your ensembles are ready, run the unified production script. This script handles property computation and LIP prediction in one go:

```bash
python scripts/train.py \
    --config  data/models/CORE_LIP_IDPFold2/config.yaml \
    --device cuda
```

---

## Paper Reproduction Workflow
### 0 - Preliminary Data Acquisition (Optional)

If you wish to replicate the specific examples discussed in the paper, please download the full-atom conformational ensembles and other data from Zenodo:
https://zenodo.org/records/xxxxxxx and place them into the data/ folder

### 1 — Compute conformational properties with EnsembleMDP library

```bash
python scripts/compute_properties.py \
    --input_dir  data/conformational_ensemble/[CONFORMATION_ORIGIN]/ \
    --workers    15
```

- CONFORMATION_ORIGIN can be IDPFold2 / CALVADOS3 or even STARLING   
- The output will be inside data/properties/[CONFORMATION_ORIGIN]_derived_properties.h5


### 2 — Visualise feature distributions

```bash
python scripts/visualize_features.py \
    --dataset  data/CLIP_dataset/TR1000_less_than_1024.txt \
    --h5       data/properties/IDPFold2_derived_properties.h5 \
    --output   results/feature_comparison_violin.pdf
```
This create a violin plot with all the features with the name results/{dataset_stem}_{h5_stem}_feature_comparison_violin.pdf

### 3 — Train the model

```bash
python scripts/train.py \
    --config  data/models/CORE_LIP_IDPFold2/config.yaml \
    --device mps
```

### 4 — Make predictions

```bash
─ python scripts/predict.py \
    --model     data/models/CORE_LIP_IDPFold2/core_lip.pt \
    --h5        data/properties/IDPFold2_derived_properties.h5 \
    --datasets  data/CLIP_dataset/TE440_less_than_1024.txt \
    --output_dir data/predictions/
```

### 5 — Evaluate

```bash
python scripts/evaluate.py \
    --test_truth  data/CLIP_dataset/TE440_less_than_1024.txt \
    --pred_files  data/predictions/core_lip_TE440_less_than_1024.csv \
                  data/predictions/CLIP_TE440.csv \
                  data/predictions/MoRFchibi_TE440.csv \
    --names       "CORE-LIP IDPFold2 v1" "CLIP" "MoRFchibi V2.0" \
    --output_dir  results/
```

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

### Scalar (per-protein, from conformational ensemble)
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