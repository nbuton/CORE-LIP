import os
import glob
import csv


def generate_prediction_csv(input_dir, output_file, threshold=0.5):
    """
    Parses .caid files and compiles them into a single CSV with
    comma-separated prediction strings.
    """
    # Find all .caid files in the specified directory
    caid_files = sorted(glob.glob(os.path.join(input_dir, "*.caid")))

    if not caid_files:
        print(f"No .caid files found in {input_dir}")
        return

    print(f"Found {len(caid_files)} files. Processing...")

    # Open the output CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the exact header you requested
        writer.writerow(["protein_id", "length", "predictions", "binary_predictions"])

        for file_path in caid_files:
            # Extract protein ID from the filename (e.g., "DP02000.caid" -> "DP02000")
            protein_id = os.path.basename(file_path).replace(".caid", "")

            scores = []
            binary_preds = []

            with open(file_path, "r") as infile:
                for line in infile:
                    line = line.strip()
                    # Skip empty lines or the FASTA/CAID header (e.g., >DP02000)
                    if not line or line.startswith(">"):
                        continue

                    # CAID format is typically: Index  Residue  Score
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            score_val = float(parts[2])
                            scores.append(str(score_val))

                            # Determine binary classification based on threshold
                            binary = "1" if score_val >= threshold else "0"
                            binary_preds.append(binary)

                        except ValueError:
                            # Skip lines where the score isn't a valid float
                            continue

            # Only write to CSV if we actually parsed sequence data
            if scores:
                length = len(scores)
                # Join the lists into comma-separated strings
                predictions_str = ",".join(scores)
                binary_preds_str = ",".join(binary_preds)

                # The csv module will automatically wrap strings with commas in double quotes
                writer.writerow([protein_id, length, predictions_str, binary_preds_str])

    print(f"Successfully generated formatted CSV at: {output_file}")


if __name__ == "__main__":
    # Define your paths here
    SOURCE_DIR = "data/predictions/output_MoRFchibi 2.0_MC2"
    TARGET_CSV = "data/predictions/MoRFchibi_TE440.csv"

    # You can change the threshold here if your specific evaluation requires it
    generate_prediction_csv(SOURCE_DIR, TARGET_CSV, threshold=0.95)
