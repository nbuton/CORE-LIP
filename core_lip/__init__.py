"""
CORE-LIP
========
COformational Representation Ensemble for Linear Interaction Peptide (LIP) prediction.

Predicts LIP propensity from an ensemble of protein conformations using a
multi-scale Transformer that integrates sequence, per-residue, global scalar,
and pairwise conformational features derived from MD trajectories.
"""

__all__ = [
    "ProteinModelConfig",
    "ProteinMultiScaleTransformer",
    "AA_TO_INT",
    "ProteinDataset",
    "ProteinInferenceDataset",
    "collate_proteins",
    "collate_inference",
    "protein_label_from_residue_labels",
    "prepare_data",
    "read_protein_data",
    "filter_protein_file",
]
