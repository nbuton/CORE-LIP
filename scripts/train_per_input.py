import argparse
import csv
import os
import copy
import yaml
from pathlib import Path
from core_lip.engine.trainer import CORE_LIP_Trainer, get_config

# Ensure FullConfig is imported from your definitions
# from your_config_module import FullConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the yaml config file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 1. Load and parse the base configuration
    base_cfg = get_config(args.config)

    # Mapping for the loop
    groups = {
        "SCALAR_FEATURES": {"input_key": "scalar_features", "dim_attr": "nb_scalar"},
        "LOCAL_FEATURES": {"input_key": "local_features", "dim_attr": "nb_local"},
        "PAIRWISE_FEATURES": {
            "input_key": "pairwise_features",
            "dim_attr": "nb_pairwise",
        },
    }

    results = []

    for list_name, meta in groups.items():
        # Get the list of features for this category from the BASE config
        feature_names = getattr(base_cfg.training, list_name)

        for feature in feature_names:
            print(f"\n{'='*60}")
            print(f"Testing individual feature: {feature}")
            print(f"{'='*60}")

            # Create a fresh copy
            run_cfg = base_cfg.model_copy(deep=True)

            # --- FIX: DO NOT EMPTY THE OTHER LISTS ---
            # We only modify the specific list we are testing to contain our single feature.
            # The other lists stay populated so the Scaler doesn't find '0' features.
            setattr(run_cfg.training, list_name, [feature])

            # --- ISOLATE AT MODEL LEVEL ---
            # We tell the model architecture to ONLY expect 1 feature from the target group
            # and 0 features from the other groups.
            run_cfg.model.nb_scalar = 0
            run_cfg.model.nb_local = 0
            run_cfg.model.nb_pairwise = 0

            setattr(run_cfg.model, meta["dim_attr"], 1)
            run_cfg.model.inputs_features = [meta["input_key"]]

            try:
                # Initialize and run
                trainer = CORE_LIP_Trainer(run_cfg, args.config, device=args.device)
                best_auc = trainer.run()

                results.append(
                    {
                        "feature_name": feature,
                        "group": meta["input_key"],
                        "best_auc": best_auc,
                    }
                )
                print(f"Result for {feature}: {best_auc}")

            except Exception as e:
                print(f"Error training feature {feature}: {e}")
                results.append(
                    {
                        "feature_name": feature,
                        "group": meta["input_key"],
                        "best_auc": "ERROR",
                    }
                )
            print(results)

    # 4. Save results to CSV
    output_path = Path("data/perf_per_inputs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_name", "group", "best_auc"])
        writer.header = writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Experiment complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
