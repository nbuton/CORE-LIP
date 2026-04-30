"""
core_lip/data/io.py
-------------------
Data preparation, parsing, feature extraction, and I/O utilities.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Adjust this import based on your exact structure
from core_lip.data.datasets import AA_TO_INT
from core_lip.eval.structures import ResidueExample

# Allow reading large CSV fields for massive protein sequences
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# 1. Dataset I/O (FASTA-like & CLIP format)
# ===========================================================================


def read_protein_data(file_path: str | Path) -> pd.DataFrame:
    """Read a CLIP-format dataset file into a DataFrame."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
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
    input_path: str | Path,
    protein_ids: list[str],
    output_path: str | Path,
) -> None:
    """Write a subset of a CLIP-format file, keeping only the given protein IDs."""
    target_ids = {f">{pid.strip()}" for pid in protein_ids}

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
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


# ===========================================================================
# 2. Evaluation Parsers (Truth & Predictions)
# ===========================================================================


def _read_blocks(path: str | Path) -> List[List[str]]:
    """Split a FASTA-like file into header+body blocks."""
    blocks: List[List[str]] = []
    current: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_binary_string(s: str) -> np.ndarray:
    return np.fromiter((1 if c == "1" else 0 for c in s.strip()), dtype=np.int8)


def _parse_prob_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in s.split(",") if x], dtype=np.float64)


def _parse_binary_csv_string(s: str) -> np.ndarray:
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.int8)
    return np.array([int(x) for x in s.split(",") if x], dtype=np.int8)


def parse_truth_file(path: str | Path) -> Dict[str, ResidueExample]:
    """Parse a FASTA-like ground-truth file into ResidueExample objects."""
    records: Dict[str, ResidueExample] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed truth block (expected ≥3 lines): {block}")
        protein_id = block[0][1:].strip()
        sequence = block[1].strip()
        y_true = _parse_binary_string("".join(block[2:]).strip())
        if len(sequence) != len(y_true):
            raise ValueError(
                f"Length mismatch for {protein_id}: seq={len(sequence)}, labels={len(y_true)}"
            )
        records[protein_id] = ResidueExample(protein_id, sequence, y_true)
    return records


def parse_prediction_csv(
    path: str | Path,
    records: Dict[str, ResidueExample],
    model_name: str,
) -> None:
    """Load per-residue predictions from a CSV into existing ResidueExample objects."""
    required = {"protein_id", "length", "predictions", "binary_predictions"}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            pid = row["protein_id"].strip()
            if pid not in records:
                continue
            expected_len = int(row["length"])
            scores = _parse_prob_string(row["predictions"])
            binary = _parse_binary_csv_string(row["binary_predictions"])

            if len(scores) != expected_len or len(binary) != expected_len:
                raise ValueError(
                    f"Length mismatch for {pid} in {path}: expected {expected_len}, "
                    f"got scores={len(scores)}, binary={len(binary)}"
                )
            records[pid].add_prediction(model_name, scores, binary)

    missing = [pid for pid in records if model_name not in records[pid].scores]
    if missing:
        raise ValueError(
            f"Model '{model_name}' missing predictions for {len(missing)} proteins. Missing: {missing}"
        )


# ===========================================================================
# 3. Feature Extraction & Statistics
# ===========================================================================


def prepare_data(
    df: pd.DataFrame,
    h5_data,
    scalar_features: list[str],
    local_features: list[str],
    pairwise_features: list[str],
    aa_to_int_dict: dict[str, int] = AA_TO_INT,
):
    """Extract and organise features from an HDF5 file for all proteins in df."""
    list_ids, X_scalar_list, X_local_list = [], [], []
    X_pairwise_list, seq_enc_list, y_list = [], [], []

    for _, row in df.iterrows():
        pid = row["protein_id"]
        seq_enc = np.array(
            [aa_to_int_dict.get(aa, 0) for aa in row["sequence"]], dtype=np.int64
        )
        scalar_feats = np.array(
            [h5_data[pid][f][()] for f in scalar_features], dtype=np.float32
        )
        local_feats = np.stack([h5_data[pid][f][()] for f in local_features], axis=0)

        if pairwise_features:
            pairwise_feats = np.stack(
                [h5_data[pid][f][()] for f in pairwise_features], axis=0
            )
        else:
            pairwise_feats = np.empty((0,), dtype=np.float32)

        # Convert your list, mapping '-' to -1
        mapping = {"0": 0, "1": 1, "-": -1}
        labels = np.array(
            [mapping[c] for c in row["LIP_annotations"]], dtype=np.float32
        )

        X_scalar_list.append(scalar_feats)
        X_local_list.append(local_feats)
        X_pairwise_list.append(pairwise_feats)
        seq_enc_list.append(seq_enc)
        y_list.append(labels)
        list_ids.append(pid)

    return X_scalar_list, X_local_list, X_pairwise_list, seq_enc_list, y_list, list_ids


def get_all_feature_stats(X_scalar_list, X_local_list, X_pairwise_list):
    """Computes means and stds for Scalar, Local, and Pairwise features."""
    stats = {}

    # Scalar Stats
    X_scalar_matrix = np.stack(X_scalar_list)
    s_scaler = StandardScaler().fit(X_scalar_matrix)
    stats["scalar"] = {
        "means": torch.from_numpy(s_scaler.mean_).float(),
        "stds": torch.from_numpy(s_scaler.scale_).float(),
    }

    # Local Stats
    X_local_flat = np.concatenate([arr.T for arr in X_local_list], axis=0)
    l_scaler = StandardScaler().fit(X_local_flat)
    stats["local"] = {
        "means": torch.from_numpy(l_scaler.mean_).float(),
        "stds": torch.from_numpy(l_scaler.scale_).float(),
    }

    # Pairwise Stats
    if X_pairwise_list and X_pairwise_list[0].size > 0:
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


# ===========================================================================
# 4. Clustering (External Tool Dispatch)
# ===========================================================================


def cluster_sequences_mmseqs2(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    id_col: str = "id",
    output_file: str = "data/TR1000_cluster.csv",
    seq_identity: float = 0.25,
) -> pd.DataFrame:
    """Cluster sequences using MMseqs2 at a given sequence identity threshold."""
    # --- Cache Verification Logic ---
    if os.path.exists(output_file):
        cached_df = pd.read_csv(output_file)

        # Check if all IDs in the current input df exist in the cached file
        missing_ids = set(df[id_col]) - set(cached_df[id_col])

        if not missing_ids:
            print(f"[clustering] {output_file} exists and contains all IDs. Loading.")
            return cached_df
        else:
            print(
                f"[clustering] {len(missing_ids)} IDs missing from cache. Re-clustering..."
            )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        db_path = os.path.join(tmpdir, "seqdb")
        cluster_db = os.path.join(tmpdir, "clusterdb")
        tmp_path = os.path.join(tmpdir, "tmp")
        tsv_path = os.path.join(tmpdir, "clusters.tsv")

        with open(fasta_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f">{row[id_col]}\n{row[sequence_col]}\n")

        subprocess.run(
            ["mmseqs", "createdb", fasta_path, db_path], check=True, capture_output=True
        )

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

    reps = cluster_df["cluster_representative"].unique()
    rep_to_idx = {rep: idx for idx, rep in enumerate(reps)}
    cluster_df["cluster"] = cluster_df["cluster_representative"].map(rep_to_idx)

    result = df[[id_col, sequence_col]].merge(
        cluster_df[[id_col, "cluster"]], on=id_col, how="left"
    )

    result.to_csv(output_file, index=False)
    print(f"[clustering] Done. {result['cluster'].nunique()} clusters found.")
    print(f"[clustering] Saved to {output_file}")

    return result
