from pathlib import Path
from typing import Any
import argparse
import hashlib
import json

import numpy as np


# ============================================================
# PSAI EMBEDDING PIPELINE v2
#
# Input:
#   <repository>/data_chunked/<doc_id>/<doc_id>.jsonl
#
# Output:
#   <repository>/data_embedded/<doc_id>/<doc_id>.npy
#   <repository>/data_embedded/<doc_id>/<doc_id>.meta.json
#   <repository>/data_embedded/<doc_id>/<doc_id>.manifest.json
#
# Model:
#   BAAI/bge-m3  (multilingual, hỗ trợ EN + VI, 1024 chiều)
#
# Alternatives (đổi MODEL_NAME):
#   intfloat/multilingual-e5-base   (768 chiều, nhẹ hơn)
#   BAAI/bge-small-en-v1.5          (384 chiều, chỉ EN)
#   text-embedding-3-small          (OpenAI, cần API)
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNK_DIR = BASE_DIR / "data_chunked"
EMBED_DIR = BASE_DIR / "data_embedded"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
DEVICE = "auto"  # Resolve only when embedding is explicitly requested.
NORMALIZE = True        # L2 normalize -> cosine = dot product

RESET_OUTPUT = False  # Never delete an existing dataset tree.


def resolve_device():
    if DEVICE != "auto":
        return DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    seen = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            for key in ("chunk_id", "doc_id", "chunk_index", "text"):
                if key not in chunk:
                    raise ValueError(f"Missing {key!r} in {jsonl_path}")
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise ValueError(f"Empty or invalid chunk text in {jsonl_path}")
            if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"]:
                raise ValueError(f"Invalid chunk ID in {jsonl_path}")
            if not isinstance(chunk["chunk_index"], int) or isinstance(chunk["chunk_index"], bool) or chunk["chunk_index"] < 0:
                raise ValueError(f"Invalid chunk index in {jsonl_path}")
            if chunk["doc_id"] != jsonl_path.parent.name:
                raise ValueError(f"Document ID mismatch in {jsonl_path}")
            if chunk["chunk_id"] in seen:
                raise ValueError(f"Duplicate chunk ID {chunk['chunk_id']!r}")
            expected_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            if chunk.get("hash") and chunk["hash"] not in (expected_hash, expected_hash[:16]):
                raise ValueError(f"Chunk hash mismatch for {chunk['chunk_id']!r}")
            seen.add(chunk["chunk_id"])
            chunks.append(chunk)
    return chunks


# ============================================================
# EMBEDDING
# ============================================================

def embed_chunks(model: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        dimension = model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Embedding model must expose a positive dimension for empty input.")
        return np.empty((0, dimension), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE,
        device=resolve_device(),
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1:
        raise ValueError("Embedding output shape does not match the input chunks.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding output contains non-finite values.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embedding output contains a zero vector.")
    if NORMALIZE and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Embedding model did not return normalized vectors.")
    return vectors


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(jsonl_path: Path, model: Any, *, output_dir=None, overwrite=False) -> dict:
    jsonl_path = Path(jsonl_path)
    doc_id = jsonl_path.parent.name
    out_folder = Path(output_dir or EMBED_DIR) / doc_id
    if out_folder.resolve() == jsonl_path.parent.resolve():
        raise ValueError("Embedding output must be separate from source chunks.")
    if out_folder.exists() and any(out_folder.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {out_folder}. Use --overwrite explicitly.")

    chunks = load_chunks(jsonl_path)
    if not chunks:
        raise ValueError(f"Document {doc_id!r} has no chunks; stale output must be handled explicitly.")

    texts = [c["text"] for c in chunks]

    print(f"[{doc_id}] embedding {len(texts)} chunks ...")
    vectors = embed_chunks(model, texts)

    out_folder.mkdir(parents=True, exist_ok=True)

    npy_file = out_folder / f"{doc_id}.npy"
    np.save(npy_file, vectors)

    meta = []
    for c in chunks:
        record = {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "section": c.get("section", ""),
            "heading_path": c.get("heading_path", []),
            "block_type": c.get("block_type", "section"),
            "token_count": c.get("token_count", 0),
            "hash": c.get("hash", ""),
            "text": c["text"],
        }
        for key in ("source_file", "source_sha256", "source_start_line", "source_end_line", "chunking_version", "unterminated_fence"):
            if key in c:
                record[key] = c[key]
        meta.append(record)

    meta_file = out_folder / f"{doc_id}.meta.json"
    meta_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "model": MODEL_NAME,
        "normalize": NORMALIZE,
        "npy_file": str(npy_file),
        "meta_file": str(meta_file),
        "source_jsonl": str(jsonl_path.resolve()),
        "source_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        "vectors_sha256": hashlib.sha256(npy_file.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(meta_file.read_bytes()).hexdigest(),
        "schema_version": "psai-embedding-v2",
    }
    (out_folder / f"{doc_id}.manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[{doc_id}] done. shape={vectors.shape}")

    return stats


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Embed validated chunks without deleting existing datasets.")
    parser.add_argument("--input-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--output-dir", type=Path, default=EMBED_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow model files to be downloaded.")
    args = parser.parse_args(argv)
    print("=" * 75)
    print("PSAI EMBEDDING PIPELINE v2")
    print("=" * 75)
    print(f"CHUNK_DIR : {args.input_dir}")
    print(f"EMBED_DIR : {args.output_dir}")
    print(f"MODEL     : {MODEL_NAME}")
    print(f"DEVICE    : {DEVICE}")
    print()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Chunk folder does not exist: {args.input_dir}")
    if args.output_dir.resolve() == args.input_dir.resolve() or args.output_dir.resolve() in args.input_dir.resolve().parents or args.input_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError("Input and output trees must be separate and non-overlapping.")
    jsonl_files = sorted(args.input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError("No .jsonl files found; no output was changed.")
    doc_ids = [path.parent.name for path in jsonl_files]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Multiple JSONL files map to the same document ID.")
    if (args.output_dir / "_summary.json").exists() and not args.overwrite:
        raise FileExistsError("Output summary already exists; choose a new output directory or --overwrite.")

    print(f"Found {len(jsonl_files)} chunk files.")
    print(f"Loading model {MODEL_NAME} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=resolve_device(), local_files_only=not args.allow_download)
    print("Model loaded.")
    print()

    all_stats = []
    failures = []
    for jsonl_file in jsonl_files:
        try:
            stats = process_document(jsonl_file, model, output_dir=args.output_dir, overwrite=args.overwrite)
            all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] {jsonl_file}: {e}")
            failures.append({"source": str(jsonl_file), "error": str(e)})

    total_chunks = sum(s.get("num_chunks", 0) for s in all_stats)

    summary = {
        "model": MODEL_NAME,
        "normalize": NORMALIZE,
        "dim": next((s["dim"] for s in all_stats if "dim" in s), None),
        "num_documents": len(all_stats),
        "total_chunks": total_chunks,
        "per_document": all_stats,
        "failures": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("EMBEDDING FINISHED")
    print("=" * 75)
    print(f"Documents : {len(all_stats)}")
    print(f"Chunks    : {total_chunks}")
    print(f"Dim       : {summary['dim']}")
    print(f"Output    : {args.output_dir}")
    print(f"Summary   : {summary_file}")
    if failures:
        raise RuntimeError("Embedding completed with errors; inspect the summary before indexing.")


if __name__ == "__main__":
    main()
