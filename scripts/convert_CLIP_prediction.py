import csv


def convert_protein_file(input_file, output_file, threshold=0.5):
    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
        # Write the header
        writer.writerow(["protein_id", "length", "predictions", "binary_predictions"])

        lines = f_in.readlines()

        # Process in chunks of 4 lines
        # Line 0: >ID
        # Line 1: Sequence (discarded for content, used for length)
        # Line 2: True Values (discarded)
        # Line 3: CLIP Probabilities
        for i in range(0, len(lines), 4):
            if i + 3 >= len(lines):
                break

            # 1. Protein ID (remove '>' and whitespace)
            protein_id = lines[i].strip().replace(">", "")

            # 2. Sequence (used to determine length)
            sequence = lines[i + 1].strip()
            length = len(sequence)

            # 3. Predictions (CLIP probabilities)
            # We skip lines[i+2] as per your request to remove 'true value'
            raw_probs = lines[i + 3].strip().split(",")

            # Convert to float and create binary string based on threshold
            # Wrapping the strings in quotes is handled by the csv.writer
            predictions_str = ",".join(raw_probs)

            binary_list = []
            for val in raw_probs:
                if float(val) >= threshold:
                    binary_list.append("1")
                else:
                    binary_list.append("0")

            binary_predictions_str = ",".join(binary_list)

            # Write the row
            writer.writerow(
                [protein_id, length, predictions_str, binary_predictions_str]
            )


if __name__ == "__main__":
    MY_THRESHOLD = 0.2

    convert_protein_file(
        "data/predictions/CLIP/CLIP_TE440.txt",
        "data/predictions/CLIP_TE440.csv",
        threshold=MY_THRESHOLD,
    )
    print(f"Conversion complete using threshold: {MY_THRESHOLD}")
