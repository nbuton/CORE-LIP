import sys
import multiprocessing
from pathlib import Path
import h5py
from tqdm import tqdm
from Bio import SeqIO

# Global variable for worker processes
worker_handler = None


def init_worker(model_name, device, layer, token_path, get_model_wrapper_fn):
    """Initializes the model wrapper inside each worker process."""
    global worker_handler
    try:
        worker_handler = get_model_wrapper_fn(
            model_name, device, wanted_layer=layer, token_path=token_path
        )
    except Exception as e:
        raise ValueError(f"Worker initialization failed: {e}")


def process_sequence_task(data):
    """Processes a single sequence record (uid, sequence)."""
    uid, sequence = data
    if worker_handler is None:
        return uid, None, "Worker handler not initialized"
    try:
        embedding = worker_handler.get_embedding(str(sequence))
        return uid, embedding, None
    except Exception as e:
        return uid, None, str(e)


class EmbeddingManager:
    """Handles the orchestration of embedding generation and H5 storage."""

    @staticmethod
    def get_computed_ids(dest_path):
        if not Path(dest_path).exists():
            return set()
        try:
            with h5py.File(dest_path, "r") as f:
                return set(f.keys())
        except Exception:
            return set()

    @staticmethod
    def save_to_h5(f_out, uid, embedding):
        grp = f_out.require_group(uid)
        if "embedding" in grp:
            del grp["embedding"]
        grp.create_dataset("embedding", data=embedding, compression="gzip")

    @staticmethod
    def fasta_generator(fasta_path, todo_ids):
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id in todo_ids:
                yield record.id, str(record.seq)
