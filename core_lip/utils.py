"""
CORE-LIP — Data preparation and I/O utilities
=============================================
"""

import numpy as np
import pandas as pd

from core_lip.datasets import AA_TO_INT


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_protein_data(file_path: str) -> pd.DataFrame:
    """
    Read the CLIP-format dataset file.

    Expected format (3 lines per protein, no blank separator):
        >PROTEIN_ID
        AMINO_ACID_SEQUENCE
        LIP_ANNOTATION_BINARY_STRING   (e.g. "00011100")

    Returns
    -------
    pd.DataFrame with columns: protein_id, sequence, LIP_annotations
    """
    data = []
    with open(file_path, "r") as f:
        while True:
            line1 = f.readline().strip()
            if not line1:
                break
            protein_id = line1.lstrip(">")
            sequence = f.readline().strip()
            annotations = f.readline().strip()
            data.append(
                {
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "LIP_annotations": annotations,
                }
            )
    return pd.DataFrame(data)


def filter_protein_file(
    input_path: str,
    protein_ids: list[str],
    output_path: str,
) -> None:
    """
    Write a subset of a CLIP-format file, keeping only the given protein IDs.

    Parameters
    ----------
    input_path  : source CLIP-format file
    protein_ids : list of IDs to retain
    output_path : destination file
    """
    target_ids = {f">{pid.strip()}" for pid in protein_ids}

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        while True:
            header = infile.readline()
            sequence = infile.readline()
            mask = infile.readline()
            if not header:
                break
            if header.strip() in target_ids:
                outfile.write(header)
                outfile.write(sequence)
                outfile.write(mask)

    print(f"Filtered file saved to: {output_path}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def prepare_data(
    df: pd.DataFrame,
    h5_data,
    scalar_features: list[str],
    local_features: list[str],
    pairwise_features: list[str],
    aa_to_int_dict: dict[str, int] = AA_TO_INT,
):
    """
    Extract and organise features from an HDF5 file for all proteins in *df*.

    Parameters
    ----------
    df               : DataFrame from read_protein_data()
    h5_data          : open h5py.File object with pre-computed MD properties
    scalar_features  : list of scalar (per-protein) feature names
    local_features   : list of per-residue feature names
    pairwise_features: list of pairwise (L×L) feature names
    aa_to_int_dict   : amino-acid → integer vocabulary

    Returns
    -------
    X_scalar_list   : list of (nb_scalar,)         float32 arrays
    X_local_list    : list of (nb_local, L)         float32 arrays
    X_pairwise_list : list of (nb_pairwise, L, L)   float32 arrays
    seq_enc_list    : list of (L,)                  int64 arrays
    y_list          : list of (L,)                  int arrays
    list_ids        : list of protein ID strings
    """
    list_ids, X_scalar_list, X_local_list = [], [], []
    X_pairwise_list, seq_enc_list, y_list = [], [], []

    for _, row in df.iterrows():
        pid = row["protein_id"]

        # Sequence encoding
        seq_enc = np.array(
            [aa_to_int_dict.get(aa, 0) for aa in row["sequence"]], dtype=np.int64
        )

        # Scalar features: shape (nb_scalar,)
        scalar_feats = np.array(
            [h5_data[pid][f][()] for f in scalar_features], dtype=np.float32
        )

        # Local features: shape (nb_local, L)
        local_feats = np.stack([h5_data[pid][f][()] for f in local_features], axis=0)

        # Pairwise features: shape (nb_pairwise, L, L)
        if pairwise_features:
            pairwise_feats = np.stack(
                [h5_data[pid][f][()] for f in pairwise_features], axis=0
            )
        else:
            pairwise_feats = np.empty((0,), dtype=np.float32)

        # Labels: "00011" → [0, 0, 0, 1, 1]
        labels = np.array([int(c) for c in row["LIP_annotations"]])

        X_scalar_list.append(scalar_feats)
        X_local_list.append(local_feats)
        X_pairwise_list.append(pairwise_feats)
        seq_enc_list.append(seq_enc)
        y_list.append(labels)
        list_ids.append(pid)

    return X_scalar_list, X_local_list, X_pairwise_list, seq_enc_list, y_list, list_ids
