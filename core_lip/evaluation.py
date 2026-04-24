"""
core_lip/evaluation.py
----------------------
Data structures and parsers for residue-level evaluation.

    ResidueExample      : holds ground truth + per-model predictions for one protein
    parse_truth_file    : read a FASTA-like ground-truth file
    parse_prediction_csv: load a predict.py CSV into existing ResidueExample records
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


class ResidueExample:
    """
    Holds ground truth and per-model predictions for a single protein.

    Attributes
    ----------
    protein_id : str
    sequence   : str
    y_true     : np.ndarray[int8]  — per-residue ground-truth labels
    scores     : dict[model_name → np.ndarray[float64]]  — continuous scores
    binary     : dict[model_name → np.ndarray[int8]]     — binary predictions
    """

    def __init__(self, protein_id: str, sequence: str, y_true: np.ndarray) -> None:
        self.protein_id = protein_id
        self.sequence = sequence
        self.y_true: np.ndarray = y_true
        self.scores: Dict[str, np.ndarray] = {}
        self.binary: Dict[str, np.ndarray] = {}

    def add_prediction(
        self, model_name: str, scores: np.ndarray, binary: np.ndarray
    ) -> None:
        n = len(self.sequence)
        for arr, label in [(scores, "scores"), (binary, "binary")]:
            if len(arr) != n:
                raise ValueError(
                    f"Length mismatch for {self.protein_id} / {model_name} "
                    f"({label}): seq={n}, pred={len(arr)}"
                )
        self.scores[model_name] = scores
        self.binary[model_name] = binary


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


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
    """Convert a compact binary string (e.g. '010011') to an int8 array."""
    return np.fromiter((1 if c == "1" else 0 for c in s.strip()), dtype=np.int8)


def _parse_prob_string(s: str) -> np.ndarray:
    """Parse a comma-separated probability string to a float64 array."""
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in s.split(",") if x], dtype=np.float64)


def _parse_binary_csv_string(s: str) -> np.ndarray:
    """Parse a comma-separated binary prediction string to an int8 array."""
    s = s.strip().strip('"')
    if not s:
        return np.array([], dtype=np.int8)
    return np.array([int(x) for x in s.split(",") if x], dtype=np.int8)


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------


def parse_truth_file(path: str | Path) -> Dict[str, ResidueExample]:
    """
    Parse a FASTA-like ground-truth file with the format::

        >protein_id
        SEQUENCE
        0101010...   ← per-residue binary labels (compact string, no separator)

    Returns
    -------
    dict mapping protein_id → ResidueExample
    """
    records: Dict[str, ResidueExample] = {}
    for block in _read_blocks(path):
        if len(block) < 3:
            raise ValueError(f"Malformed truth block (expected ≥3 lines): {block}")
        protein_id = block[0][1:].strip()
        sequence = block[1].strip()
        y_true = _parse_binary_string("".join(block[2:]).strip())
        if len(sequence) != len(y_true):
            raise ValueError(
                f"Length mismatch for {protein_id}: "
                f"seq={len(sequence)}, labels={len(y_true)}"
            )
        records[protein_id] = ResidueExample(protein_id, sequence, y_true)
    return records


def parse_prediction_csv(
    path: str | Path,
    records: Dict[str, ResidueExample],
    model_name: str,
) -> None:
    """
    Load per-residue predictions from a CSV produced by ``predict.py`` into
    existing ``ResidueExample`` objects.

    The CSV must contain at minimum these columns (extra columns are ignored)::

        protein_id, length, predictions, binary_predictions

    Proteins absent from *records* are silently skipped.  If any protein in
    *records* has no entry in the CSV a ``ValueError`` is raised after loading.

    Parameters
    ----------
    path       : path to the prediction CSV
    records    : dict of ResidueExample objects to populate (modified in place)
    model_name : key under which predictions are stored in each ResidueExample
    """
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
                    f"Length mismatch for {pid} in {path}: "
                    f"expected {expected_len}, "
                    f"got scores={len(scores)}, binary={len(binary)}"
                )
            records[pid].add_prediction(model_name, scores, binary)

    missing = [pid for pid in records if model_name not in records[pid].scores]
    if missing:
        raise ValueError(
            f"Model '{model_name}' is missing predictions for "
            f"{len(missing)} protein(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
