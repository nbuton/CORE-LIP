"""
Utility: Filter protein sequence text file by length
====================================================
Reads a 3-line format text file (Header, Sequence, Features)
and outputs a new file containing only proteins with a
sequence length strictly less than the specified maximum.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter protein dataset by sequence length."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input txt file (e.g., TR440.txt).",
    )
    parser.add_argument(
        "-m",
        "--max_length",
        type=int,
        required=True,
        help="Maximum sequence length (exclusive).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # Ensure the input file exists
    if not input_path.exists():
        print(f"Error: Could not find the file '{input_path}'.")
        return

    # Construct the output file path in the same folder
    # e.g., folder/TR440.txt -> folder/TR440_less_than_500.txt
    new_filename = f"{input_path.stem}_less_than_{args.max_length}{input_path.suffix}"
    output_path = input_path.parent / new_filename

    total_count = 0
    kept_count = 0

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        # Read lines, ignoring any completely blank lines
        lines = [line.strip() for line in infile if line.strip()]

        # Iterate through the file in chunks of 3 lines
        for i in range(0, len(lines), 3):
            # Make sure we have a complete set of 3 lines
            if i + 2 < len(lines):
                header = lines[i]
                sequence = lines[i + 1]
                features = lines[i + 2]

                total_count += 1

                # Check if sequence length is strictly less than max_length
                # Note: Change `<` to `<=` if you want it to include the exact length.
                if len(sequence) < args.max_length:
                    outfile.write(f"{header}\n{sequence}\n{features}\n")
                    kept_count += 1

    print(f"--- Filtering Complete ---")
    print(f"Input file:  {input_path}")
    print(f"Total read:  {total_count} proteins")
    print(f"Total kept:  {kept_count} proteins (length < {args.max_length})")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
