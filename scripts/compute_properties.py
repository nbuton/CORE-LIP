"""
CORE-LIP — Step 1: Compute conformational properties
=====================================================
Processes a folder of full atoms ensemble conformation, converts trajectory
formats, and computes MD-derived features for each protein. Results are
saved to a compressed HDF5 file.

Usage
-----
    python scripts/compute_properties.py \
        --input_dir  data/conformational_ensemble/ \
        --output_h5  data/protein_MD_properties.h5 \
        --workers    15
"""

import argparse
import ctypes
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
import concurrent.futures

import h5py
import mdtraj as md
import numpy as np
from tqdm import tqdm

from EnsembleMDP.analysis.orchestrator import ProteinAnalyzer

# Suppress noisy library warnings
warnings.filterwarnings("ignore", module="MDAnalysis")
warnings.filterwarnings("ignore", module="mdtraj")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SuppressCStdout:
    """Context manager that redirects C-level stdout/stderr to /dev/null."""

    def __init__(self):
        try:
            self.libc = ctypes.CDLL(None)
        except Exception:
            self.libc = ctypes.cdll.msvcrt

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.old_stdout_fd = os.dup(sys.stdout.fileno())
        self.old_stderr_fd = os.dup(sys.stderr.fileno())
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull_fd, sys.stdout.fileno())
        os.dup2(self.devnull_fd, sys.stderr.fileno())

    def __exit__(self, *_):
        try:
            self.libc.fflush(None)
        except Exception:
            pass
        os.dup2(self.old_stdout_fd, sys.stdout.fileno())
        os.dup2(self.old_stderr_fd, sys.stderr.fileno())
        os.close(self.old_stdout_fd)
        os.close(self.old_stderr_fd)
        os.close(self.devnull_fd)


def convert_trajectory_format(folder_path: Path) -> None:
    """
    Rename topology and convert trajectory within a protein folder:
        aa_topology.pdb  →  top_AA.pdb
        aa_traj.dcd      →  traj_AA.xtc
    """
    old_pdb = folder_path / "aa_topology.pdb"
    new_pdb = folder_path / "top_AA.pdb"

    if old_pdb.exists():
        shutil.move(str(old_pdb), str(new_pdb))
        logging.debug(f"Renamed: {old_pdb} → {new_pdb}")

    old_dcd = folder_path / "aa_traj.dcd"
    new_xtc = folder_path / "traj_AA.xtc"

    if old_dcd.exists() and new_pdb.exists():
        logging.debug(f"Converting {old_dcd} → {new_xtc} …")
        with SuppressCStdout():
            traj = md.load(str(old_dcd), top=str(new_pdb))
            traj.save_xtc(str(new_xtc))
        logging.debug(f"Saved: {new_xtc}")


def compute_properties(protein_dir: Path) -> dict:
    """Run ProteinAnalyzer on a converted protein directory."""
    pdb_path = protein_dir / "top_AA.pdb"
    xtc_path = protein_dir / "traj_AA.xtc"
    analyzer = ProteinAnalyzer(pdb_path, xtc_path)
    return analyzer.compute_all(
        sasa_n_sphere=1600,
        contact_cutoff=8.0,
        scaling_min_sep=5,
    )


def process_single_protein(protein_dir: Path):
    """Worker: convert format + compute properties for one protein."""
    protein_id = protein_dir.stem
    traj_path_dcd = protein_dir / "aa_traj.dcd"
    traj_path_xtc = protein_dir / "traj_AA.xtc"

    if not traj_path_dcd.exists() and not traj_path_xtc.exists():
        raise RuntimeError(
            f"No trajectory file found inside this directory: {protein_dir}"
        )

    if not traj_path_xtc.exists():
        convert_trajectory_format(protein_dir)
    properties = compute_properties(protein_dir)
    return protein_id, properties


def save_properties_to_h5(dico_properties: dict, output_filepath: str) -> None:
    """
    Save a nested {protein_id: {feature_name: array}} dict to HDF5.
    Multi-dimensional arrays are gzip-compressed at level 4.
    """
    with h5py.File(output_filepath, "w") as h5f:
        for protein_id, props in dico_properties.items():
            grp = h5f.create_group(protein_id)
            for name, value in props.items():
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                if value.ndim > 0:
                    grp.create_dataset(
                        name, data=value, compression="gzip", compression_opts=4
                    )
                else:
                    grp.create_dataset(name, data=value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute conformational MD properties."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/conformational_ensemble/"),
        help="Directory containing per-protein subdirectories with DCD/PDB trajectories.",
    )
    parser.add_argument(
        "--output_h5",
        type=Path,
        default=Path("data/protein_MD_properties.h5"),
        help="Path for the output HDF5 feature file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    directories = [d for d in args.input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(directories)} protein directories.")
    print(f"Starting parallel processing with {args.workers} workers …")

    dico_properties: dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_protein, directories),
                total=len(directories),
            )
        )

    for res in results:
        if res is not None:
            protein_id, properties = res
            dico_properties[protein_id] = properties
        else:
            logging.debug(f"Error for one res")

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    save_properties_to_h5(dico_properties, str(args.output_h5))
    print(f"Saved {len(dico_properties)} proteins → {args.output_h5}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()
