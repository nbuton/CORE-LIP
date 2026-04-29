from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import h5py
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from core_lip.data.datasets import ProteinDataset, collate_proteins
from core_lip.data.io import prepare_data, read_protein_data
from core_lip.engine.interpretability import AttributionResult, REGISTRY
from core_lip.engine.predictor import load_checkpoint


def run_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scalar_names: List[str],
    local_names: List[str],
    pairwise_names: List[str],
    methods: Optional[List[str]] = None,
    method_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, AttributionResult]:
    """
    Run a set of interpreters over *loader* and return all results.
    """
    methods = methods or list(REGISTRY.keys())
    method_kwargs = method_kwargs or {}
    results: Dict[str, AttributionResult] = {}

    for name in methods:
        if name not in REGISTRY:
            warnings.warn(f"Unknown method '{name}' - skipping.")
            continue
        print(f"[lip_interpretability] Running: {name} ...", flush=True)
        cls = REGISTRY[name]
        interpreter = cls(
            model=model,
            scalar_names=scalar_names,
            local_names=local_names,
            pairwise_names=pairwise_names,
            device=device,
        )
        try:
            results[name] = interpreter.run(loader, **method_kwargs.get(name, {}))
            print(f"[lip_interpretability]   done: {name}")
        except Exception as exc:
            warnings.warn(f"[lip_interpretability] {name} failed: {exc}")

    return results


def summary_report(results: Dict[str, AttributionResult]) -> pd.DataFrame:
    """
    Build a ranked consensus feature-importance table.
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for name, res in results.items():
        if name in (
            "feature_value_correlation",
            "feature_range_profiler",
            "attention_rollout",
        ):
            continue
        df = res.mean_per_feature()[["stream", "feature", "mean_abs_attr"]]
        df = df.rename(columns={"mean_abs_attr": name})
        dfs[name] = df.set_index(["stream", "feature"])

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs.values(), axis=1)

    for col in combined.columns:
        col_range = combined[col].max() - combined[col].min()
        if col_range > 0:
            combined[col] = (combined[col] - combined[col].min()) / col_range

    combined["consensus_score"] = combined.mean(axis=1)
    return combined.sort_values("consensus_score", ascending=False).reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CORE-LIP interpretability methods over one or more datasets."
    )
    parser.add_argument(
        "--model",
        default="data/models/CORE_LIP_IDPFold2/core_lip.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="data/models/CORE_LIP_IDPFold2/config.yaml",
        help="Path to the training/config YAML used to recover feature names.",
    )
    parser.add_argument(
        "--h5",
        default="data/protein_MD_properties.h5",
        help="Feature HDF5 file with MD-derived inputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/CLIP_dataset/TE440_reduced.txt"],
        help="One or more dataset text files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interpretability/",
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of interpretability methods to run. Default: all registered methods.",
    )
    parser.add_argument(
        "--method-kwargs",
        default=None,
        help="JSON string or JSON file mapping method name to kwargs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the interpretation DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--plm-h5",
        default="data/embeddings/esm3-large-2024-03_merged.h5",
        help="Path to the PLM embedding HDF5 used by ProteinDataset.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also pickle the raw results dictionary for each dataset.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.csv",
        help="Filename used for the consensus summary inside each dataset output directory.",
    )
    return parser.parse_args()


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def parse_method_kwargs(value: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}

    candidate = Path(value)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)

    if not isinstance(payload, dict):
        raise ValueError("--method-kwargs must decode to a JSON object.")

    parsed: Dict[str, Dict[str, Any]] = {}
    for method_name, kwargs in payload.items():
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"Expected kwargs for method '{method_name}' to be a JSON object."
            )
        parsed[method_name] = kwargs
    return parsed


def _as_name_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def extract_feature_names(
    config: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if checkpoint is not None:
        scalar_names = _as_name_list(checkpoint.get("scalar_features"))
        local_names = _as_name_list(checkpoint.get("local_features"))
        pairwise_names = _as_name_list(checkpoint.get("pairwise_features"))
        if scalar_names or local_names or pairwise_names:
            return scalar_names, local_names, pairwise_names

    candidates = [
        (
            config.get("features", {}),
            ("scalar_names", "local_names", "pairwise_names"),
        ),
        (
            config,
            ("scalar_names", "local_names", "pairwise_names"),
        ),
        (
            config.get("feature_names", {}),
            ("scalar", "local", "pairwise"),
        ),
        (
            config.get("data", {}),
            ("scalar_names", "local_names", "pairwise_names"),
        ),
    ]

    for source, keys in candidates:
        if not isinstance(source, dict):
            continue
        scalar_names = _as_name_list(source.get(keys[0]))
        local_names = _as_name_list(source.get(keys[1]))
        pairwise_names = _as_name_list(source.get(keys[2]))
        if scalar_names or local_names or pairwise_names:
            return scalar_names, local_names, pairwise_names

    raise KeyError(
        "Could not recover feature names from the config. "
        "Expected keys like features.scalar_names / features.local_names / "
        "features.pairwise_names."
    )


def build_loader(
    dataset_path: str,
    h5_path: str,
    checkpoint: Dict[str, Any],
    plm_h5_path: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    with h5py.File(h5_path, "r") as h5_features:
        df = read_protein_data(dataset_path)
        X_scalar, X_local, X_pairwise, seqs, _labels, ids = prepare_data(
            df,
            h5_features,
            checkpoint["scalar_features"],
            checkpoint["local_features"],
            checkpoint["pairwise_features"],
        )

    dataset = ProteinDataset(
        X_scalar,
        X_local,
        X_pairwise,
        seqs,
        ids=ids,
        plm_h5_path=plm_h5_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_proteins,
    )


def save_results(
    dataset_name: str,
    output_dir: Path,
    results: Dict[str, AttributionResult],
    summary_filename: str,
    save_raw: bool,
) -> None:
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summary_report(results)
    if not summary_df.empty:
        summary_path = dataset_dir / summary_filename
        summary_df.to_csv(summary_path, index=False)
        print(f"[lip_interpretability] Wrote summary: {summary_path}")

    for method_name, result in results.items():
        if hasattr(result, "mean_per_feature"):
            per_feature = result.mean_per_feature()
            per_feature_path = dataset_dir / f"{method_name}_mean_per_feature.csv"
            per_feature.to_csv(per_feature_path, index=False)
            print(f"[lip_interpretability] Wrote per-feature table: {per_feature_path}")

        if hasattr(result, "to_frame"):
            frame = result.to_frame()
            frame_path = dataset_dir / f"{method_name}_raw.csv"
            frame.to_csv(frame_path, index=False)
            print(f"[lip_interpretability] Wrote raw table: {frame_path}")

    if save_raw:
        raw_path = dataset_dir / "results.pkl"
        with raw_path.open("wb") as handle:
            pickle.dump(results, handle)
        print(f"[lip_interpretability] Wrote pickle: {raw_path}")


def iter_dataset_paths(paths: Sequence[str]) -> Iterable[Tuple[str, str]]:
    for dataset_path in paths:
        yield Path(dataset_path).stem, dataset_path


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(args.config)
    print(f"Loaded config: {args.config}")

    model, checkpoint = load_checkpoint(args.model, device)
    model.eval()
    print(f"Loaded checkpoint: {args.model}")

    scalar_names, local_names, pairwise_names = extract_feature_names(
        config, checkpoint
    )
    print(
        "[lip_interpretability] Feature groups: "
        f"{len(scalar_names)} scalar, {len(local_names)} local, {len(pairwise_names)} pairwise"
    )

    method_kwargs = parse_method_kwargs(args.method_kwargs)

    for dataset_name, dataset_path in iter_dataset_paths(args.datasets):
        print(f"\n[lip_interpretability] Dataset: {dataset_path}")
        loader = build_loader(
            dataset_path=dataset_path,
            h5_path=args.h5,
            checkpoint=checkpoint,
            plm_h5_path=args.plm_h5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        results = run_all(
            model=model,
            loader=loader,
            device=device,
            scalar_names=scalar_names,
            local_names=local_names,
            pairwise_names=pairwise_names,
            methods=args.methods,
            method_kwargs=method_kwargs,
        )
        save_results(
            dataset_name=dataset_name,
            output_dir=output_dir,
            results=results,
            summary_filename=args.summary_name,
            save_raw=args.save_raw,
        )

        if checkpoint is not None:
            checkpoint_path = output_dir / dataset_name / "checkpoint_meta.json"
            try:
                serializable = json.loads(json.dumps(checkpoint, default=str))
                checkpoint_path.write_text(
                    json.dumps(serializable, indent=2), encoding="utf-8"
                )
                print(
                    f"[lip_interpretability] Wrote checkpoint metadata: {checkpoint_path}"
                )
            except Exception:
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()
