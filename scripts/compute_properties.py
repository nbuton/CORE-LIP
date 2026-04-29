import argparse
import logging
import concurrent.futures
from pathlib import Path
import h5py
from tqdm import tqdm

from core_lip.data.properties_extraction import (
    process_single_protein,
    save_properties_to_h5,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute conformational MD properties."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    output_h5 = (
        Path("data/properties/") / f"{args.input_dir.stem}_derived_properties.h5"
    )

    directories = [d for d in args.input_dir.iterdir() if d.is_dir()]
    # 3. Filter out IDs already present in the HDF5 file
    if output_h5.exists():
        with h5py.File(output_h5, "r") as h5f:
            existing_ids = set(h5f.keys())

        # Filter directories: only keep those whose name (stem) isn't in the H5
        directories = [d for d in directories if d.stem not in existing_ids]

        print(f"Skipped {len(existing_ids)} proteins already present in the H5 file.")

    if not directories:
        print("All proteins are already processed. Exiting.")
        return

    print(
        f"Found {len(directories)} proteins. Processing with {args.workers} workers..."
    )

    results_dict = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Using as_completed allows us to update the progress bar as each finish
        futures = {executor.submit(process_single_protein, d): d for d in directories}

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                pid, props = future.result()
                results_dict[pid] = props
            except Exception as e:
                logging.error(f"Protein {futures[future].stem} failed: {e}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    save_properties_to_h5(results_dict, output_h5)
    print(f"Successfully saved {len(results_dict)} proteins to {output_h5}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
