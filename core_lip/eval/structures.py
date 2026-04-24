"""
core_lip/eval/structures.py
---------------------------
Data structures for residue-level evaluation tracking.
"""

from typing import Dict
import numpy as np


class ResidueExample:
    """
    Holds ground truth and per-model predictions for a single protein.
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
