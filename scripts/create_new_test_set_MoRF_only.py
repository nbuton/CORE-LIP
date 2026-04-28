#!/usr/bin/env python3
"""
Filter the Test-143 set to remove sequences with >= 30% sequence identity
to any sequence in TR1000 (training set), producing an independent test set.

Usage:
    python filter_test_set.py \
        --test  data/MC2-MoRF-Data/Test-143_unmasked.af \
        --train data/CLIP_dataset/TR1000.fasta \
        --out   data/MC2-MoRF-Data/Test-143_independent.af \
        --identity 0.30 \
        --threads 8
"""

import argparse
import os
import subprocess
import tempfile
import shutil
import sys


# ---------------------------------------------------------------------------
# Parsers for the two file formats
# ---------------------------------------------------------------------------


def parse_af_file(path):
    """
    Parse the joined .af file (Test-143 format):
        >accession
        amino-acid sequence
        MoRF annotation string
    Returns a list of (accession, sequence, annotation) tuples.
    """
    records = []
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            acc = lines[i][1:].strip()
            seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            ann = lines[i + 2].strip() if i + 2 < len(lines) else ""
            records.append((acc, seq, ann))
            i += 3
        else:
            i += 1
    return records


def parse_fasta(path):
    """Parse a plain FASTA file. Returns {accession: sequence}."""
    seqs = {}
    acc, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(buf)
                acc = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if acc is not None:
        seqs[acc] = "".join(buf)
    return seqs


# ---------------------------------------------------------------------------
# FASTA writers
# ---------------------------------------------------------------------------


def write_fasta(path, records):
    """Write (accession, sequence) pairs to a FASTA file."""
    with open(path, "w") as fh:
        for acc, seq in records:
            fh.write(f">{acc}\n{seq}\n")


# ---------------------------------------------------------------------------
# MMseqs2 identity search
# ---------------------------------------------------------------------------


def run_mmseqs2(test_fasta, train_fasta, identity_cutoff, threads, tmpdir):
    """
    Run MMseqs2 easy-search of test sequences against train sequences.
    Returns a set of test accessions that hit a train sequence at >= identity_cutoff.
    """
    result_tsv = os.path.join(tmpdir, "hits.tsv")
    mmseqs_tmp = os.path.join(tmpdir, "mmseqs_tmp")
    os.makedirs(mmseqs_tmp, exist_ok=True)

    cmd = [
        "mmseqs",
        "easy-search",
        test_fasta,
        train_fasta,
        result_tsv,
        mmseqs_tmp,
        "--min-seq-id",
        str(identity_cutoff),
        "--cov-mode",
        "0",  # coverage of both query and target
        "-c",
        "0.0",  # no coverage requirement (identity is the filter)
        "--alignment-mode",
        "3",  # alignment with identity
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen",
        "--threads",
        str(threads),
        "-s",
        "7.5",  # high sensitivity
        "-v",
        "1",  # minimal verbosity
    ]

    print(f"[MMseqs2] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[MMseqs2] STDERR:", result.stderr, file=sys.stderr)
        raise RuntimeError("MMseqs2 failed. See stderr above.")

    # Collect query accessions that have a hit
    contaminated = set()
    if os.path.exists(result_tsv) and os.path.getsize(result_tsv) > 0:
        with open(result_tsv) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                query_acc = parts[0]
                fident = float(parts[2]) if len(parts) > 2 else 0.0
                if fident >= identity_cutoff:
                    contaminated.add(query_acc)

    return contaminated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test", required=True, help="Path to Test-143 .af file")
    parser.add_argument("--train", required=True, help="Path to TR1000 FASTA file")
    parser.add_argument(
        "--out",
        required=True,
        help="Output filtered .af file (a .fasta file is also written alongside it)",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.30,
        help="Identity threshold to EXCLUDE (default: 0.30 = 30%%)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU threads for MMseqs2 (default: 8)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum sequence length to keep (default: 1024); use 0 to disable",
    )
    args = parser.parse_args()

    # ---- 1. Parse inputs ---------------------------------------------------
    print(f"[1/5] Parsing test file:  {args.test}")
    test_records = parse_af_file(args.test)
    print(f"      {len(test_records)} sequences loaded.")

    print(f"[2/5] Parsing train file: {args.train}")
    train_seqs = parse_fasta(args.train)
    print(f"      {len(train_seqs)} sequences loaded.")

    # ---- 2. Filter by sequence length -------------------------------------
    if args.max_len > 0:
        too_long = [
            (acc, seq, ann) for acc, seq, ann in test_records if len(seq) > args.max_len
        ]
        test_records = [r for r in test_records if len(r[1]) <= args.max_len]
        print(
            f"[2/6] Length filter (max {args.max_len} aa): "
            f"removed {len(too_long)}, {len(test_records)} remaining."
        )
        if too_long:
            for acc, _, _ in too_long:
                print(f"        too long: {acc}")
    else:
        too_long = []
        print("[2/6] Length filter: disabled.")

    # ---- 3. Write temporary FASTAs -----------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="filter_test_")
    try:
        test_fasta = os.path.join(tmpdir, "test.fasta")
        train_fasta = os.path.join(tmpdir, "train.fasta")

        write_fasta(test_fasta, [(acc, seq) for acc, seq, _ in test_records])
        write_fasta(train_fasta, list(train_seqs.items()))

        # ---- 3. Run MMseqs2 ------------------------------------------------
        print(f"[3/6] Running MMseqs2 (identity cutoff = {args.identity:.0%}) …")
        contaminated = run_mmseqs2(
            test_fasta, train_fasta, args.identity, args.threads, tmpdir
        )
        print(
            f"      {len(contaminated)} test sequence(s) exceed the identity threshold."
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- 4. Filter ---------------------------------------------------------
    print("[4/6] Filtering by identity …")
    kept = [r for r in test_records if r[0] not in contaminated]
    removed = [r for r in test_records if r[0] in contaminated]

    print(f"      Kept   : {len(kept)}")
    print(f"      Removed: {len(removed)}")
    if removed:
        print("      Removed accessions:")
        for acc, _, _ in removed:
            print(f"        {acc}")

    # ---- 5. Write outputs --------------------------------------------------
    base, _ = os.path.splitext(args.out)
    fasta_out = base + ".fasta"

    print(f"[5/6] Summary:")
    print(f"      Original       : {len(test_records) + len(too_long) + len(removed)}")
    print(f"      Removed (length): {len(too_long)}")
    print(f"      Removed (identity): {len(removed)}")
    print(f"      Final kept     : {len(kept)}")

    print(f"[6/6] Writing filtered .af  to: {args.out}")
    print(f"      Writing filtered .fasta to: {fasta_out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    csv_out = base + ".csv"

    print(f"      Writing filtered .csv   to: {csv_out}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w") as af_fh, open(fasta_out, "w") as fa_fh, open(
        csv_out, "w"
    ) as csv_fh:
        af_fh.write(f"# Sequences:\t{len(kept)}\n")
        af_fh.write("#\n")
        af_fh.write("# Format:\n")
        af_fh.write("#\t>accession\n")
        af_fh.write("#\tAmino acid sequence\n")
        af_fh.write("#\tMoRFTest annotation\n")
        af_fh.write("#\n")
        af_fh.write(
            f"# Filtered: removed {len(removed)} sequences "
            f"with >= {args.identity:.0%} identity to TR1000\n"
        )
        af_fh.write("#\n")

        csv_fh.write("test_case,sequence\n")

        for acc, seq, ann in kept:
            af_fh.write(f">{acc}\n{seq}\n{ann}\n")
            fa_fh.write(f">{acc}\n{seq}\n")
            csv_fh.write(f"{acc},{seq}\n")

    print("Done.")


if __name__ == "__main__":
    main()
