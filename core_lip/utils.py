"""
CORE-LIP — Data preparation and I/O utilities
=============================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
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


def get_all_feature_stats(X_scalar_list, X_local_list, X_pairwise_list):
    """
    Computes means and stds for Scalar, Local, and Pairwise features.

    Returns:
        Dict of { 'means': Tensor, 'stds': Tensor } for each type.
    """
    stats = {}

    # --- 1. Scalar Stats: Shape (N, nb_scalar) ---
    X_scalar_matrix = np.stack(X_scalar_list)
    s_scaler = StandardScaler().fit(X_scalar_matrix)
    stats["scalar"] = {
        "means": torch.from_numpy(s_scaler.mean_).float(),
        "stds": torch.from_numpy(s_scaler.scale_).float(),
    }

    # --- 2. Local Stats: List of (nb_local, L) ---
    # We must flatten all L dimensions across all proteins
    # Resulting shape for fit: (Total_Residues_in_Dataset, nb_local)
    X_local_flat = np.concatenate([arr.T for arr in X_local_list], axis=0)
    l_scaler = StandardScaler().fit(X_local_flat)
    stats["local"] = {
        "means": torch.from_numpy(l_scaler.mean_).float(),
        "stds": torch.from_numpy(l_scaler.scale_).float(),
    }

    # --- 3. Pairwise Stats: List of (nb_pairwise, L, L) ---
    # We flatten both L dimensions: (nb_pairwise, L*L)
    # Resulting shape for fit: (Total_Pairs_in_Dataset, nb_pairwise)
    if X_pairwise_list and X_pairwise_list[0].size > 0:
        # Move nb_features to the last dim and flatten spatial dims
        X_pair_flat = np.concatenate(
            [
                arr.transpose(1, 2, 0).reshape(-1, arr.shape[0])
                for arr in X_pairwise_list
            ],
            axis=0,
        )
        p_scaler = StandardScaler().fit(X_pair_flat)
        stats["pairwise"] = {
            "means": torch.from_numpy(p_scaler.mean_).float(),
            "stds": torch.from_numpy(p_scaler.scale_).float(),
        }
    else:
        stats["pairwise"] = {"means": torch.tensor([]), "stds": torch.tensor([])}

    return stats


import os
import subprocess
import tempfile
import pandas as pd


def cluster_sequences_mmseqs2(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    id_col: str = "id",
    output_file: str = "data/TR1000_cluster.csv",
    seq_identity: float = 0.25,
) -> pd.DataFrame:
    """
    Cluster sequences using MMseqs2 at a given sequence identity threshold.

    Args:
        df: DataFrame with at least a sequence and an id column.
        sequence_col: Name of the column containing sequences.
        id_col: Name of the column containing sequence identifiers.
        output_file: Path to the output CSV file.
        seq_identity: Sequence identity threshold for clustering (default 0.25).

    Returns:
        DataFrame with columns [id_col, sequence_col, 'cluster'].
    """
    if os.path.exists(output_file):
        print(f"[clustering] {output_file} already exists, loading it.")
        return pd.read_csv(output_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        db_path = os.path.join(tmpdir, "seqdb")
        cluster_db = os.path.join(tmpdir, "clusterdb")
        tmp_path = os.path.join(tmpdir, "tmp")
        tsv_path = os.path.join(tmpdir, "clusters.tsv")

        # Write FASTA
        with open(fasta_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f">{row[id_col]}\n{row[sequence_col]}\n")

        # Create MMseqs2 DB
        subprocess.run(
            ["mmseqs", "createdb", fasta_path, db_path],
            check=True,
            capture_output=True,
        )

        # Cluster at given seq identity
        # --min-seq-id   : sequence identity threshold
        # -c 0.8         : coverage threshold
        # --cov-mode 0   : coverage of both query and target
        # --cluster-mode 1: connected component clustering (better for low identity)
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                db_path,
                cluster_db,
                tmp_path,
                "--min-seq-id",
                str(seq_identity),
                "-c",
                "0.8",
                "--cov-mode",
                "0",
                "--cluster-mode",
                "1",
                "--threads",
                "4",
            ],
            check=True,
            capture_output=True,
        )

        # Convert cluster DB to TSV (representative \t member)
        subprocess.run(
            ["mmseqs", "createtsv", db_path, db_path, cluster_db, tsv_path],
            check=True,
            capture_output=True,
        )

        cluster_df = pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=["cluster_representative", id_col],
        )

    # Map representative id → integer cluster index
    reps = cluster_df["cluster_representative"].unique()
    rep_to_idx = {rep: idx for idx, rep in enumerate(reps)}
    cluster_df["cluster"] = cluster_df["cluster_representative"].map(rep_to_idx)

    # Merge with original df to get sequences
    result = df[[id_col, sequence_col]].merge(
        cluster_df[[id_col, "cluster"]], on=id_col, how="left"
    )

    result.to_csv(output_file, index=False)
    print(f"[clustering] Done. {result['cluster'].nunique()} clusters found.")
    print(f"[clustering] Saved to {output_file}")

    return result
