import argparse
import h5py
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter protein dataset based on presence in an HDF5 file."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input txt file."
    )
    parser.add_argument(
        "-f",
        "--hdf5",
        required=True,
        help="Path to the HDF5 file containing valid IDs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    hdf5_path = Path(args.hdf5)

    if not input_path.exists() or not hdf5_path.exists():
        print("Error: Input file or HDF5 file not found.")
        return

    # Create output path: e.g., TR440.txt -> TR440_in_h5.txt
    output_path = input_path.parent / f"{input_path.stem}_in_h5{input_path.suffix}"

    total_count = 0
    kept_count = 0

    # Open HDF5 file once and get keys for O(1) lookup
    with h5py.File(hdf5_path, "r") as h5_file:
        # Store keys in a set for maximum speed
        valid_ids = set(h5_file.keys())

        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            # Filter out empty lines while reading
            lines = [line.strip() for line in infile if line.strip()]

            for i in range(0, len(lines), 3):
                if i + 2 < len(lines):
                    header = lines[i]
                    sequence = lines[i + 1]
                    features = lines[i + 2]

                    total_count += 1

                    # Extract ID from header (assumes header is '>ID' or similar)
                    # We strip the '>' if it exists to match standard H5 key naming
                    prot_id = header.lstrip(">")

                    if prot_id in valid_ids:
                        outfile.write(f"{header}\n{sequence}\n{features}\n")
                        kept_count += 1

    print(f"--- Filtering Complete ---")
    print(f"HDF5 reference: {hdf5_path.name} ({len(valid_ids)} IDs found)")
    print(f"Total read:     {total_count} proteins")
    print(f"Total kept:     {kept_count} proteins (present in H5)")
    print(f"Output file:    {output_path}")


if __name__ == "__main__":
    main()
