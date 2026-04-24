import argparse
import multiprocessing
from pathlib import Path
import h5py
from tqdm import tqdm

from core_lip.plm_embedding.embedding_lib import (
    EmbeddingManager,
    init_worker,
    process_sequence_task,
)
from core_lip.plm_embedding.utils import get_model_wrapper


def main():
    parser = argparse.ArgumentParser(description="FASTA to H5 Protein Embedding Tool")
    parser.add_argument("--input", required=True, help="Path to input FASTA")
    parser.add_argument("--model", default="esmc_600m", help="Model name")
    parser.add_argument("--token_file", default="data/forge_token.txt")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out_dir", default="data/embeddings")
    args = parser.parse_args()

    # Setup Paths
    input_path = Path(args.input)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_name = args.model.replace("/", "_")
    layer_suffix = f"_t{args.layer}" if args.layer != -1 else ""
    dest_path = output_dir / f"{clean_name}{layer_suffix}_{input_path.stem}.h5"

    # Identify work
    done_ids = EmbeddingManager.get_computed_ids(dest_path)
    todo_ids = [
        rec.id
        for rec in tqdm(
            h5py.File(dest_path, "r") if False else [], desc="Scanning..."
        )  # Mock scan logic
        if rec.id not in done_ids
    ]  # Simplified for brevity, use SeqIO.parse in production

    # Real ID scan
    print(f"Checking {input_path.name} for new sequences...")
    from Bio import SeqIO

    todo_ids = [
        rec.id for rec in SeqIO.parse(input_path, "fasta") if rec.id not in done_ids
    ]

    if not todo_ids:
        print("Done. No new sequences to process.")
        return

    pool = None
    try:
        with h5py.File(dest_path, "a") as f_out:
            if args.workers > 1:
                pool = multiprocessing.Pool(
                    processes=args.workers,
                    initializer=init_worker,
                    initargs=(
                        args.model,
                        args.device,
                        args.layer,
                        args.token_file,
                        get_model_wrapper,
                    ),
                )
                results = pool.imap_unordered(
                    process_sequence_task,
                    EmbeddingManager.fasta_generator(input_path, set(todo_ids)),
                    chunksize=5,
                )
            else:
                init_worker(
                    args.model,
                    args.device,
                    args.layer,
                    args.token_file,
                    get_model_wrapper,
                )
                results = map(
                    process_sequence_task,
                    EmbeddingManager.fasta_generator(input_path, set(todo_ids)),
                )

            for i, (uid, emb, err) in enumerate(
                tqdm(results, total=len(todo_ids), desc=args.model)
            ):
                if err:
                    if any(w in err.lower() for w in ["quota", "429", "limit"]):
                        print(f"\nStop: {err}")
                        break
                    continue

                EmbeddingManager.save_to_h5(f_out, uid, emb)
                if i % 20 == 0:
                    f_out.flush()

    except KeyboardInterrupt:
        if pool:
            pool.terminate()
    finally:
        if pool:
            pool.close()
            pool.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
