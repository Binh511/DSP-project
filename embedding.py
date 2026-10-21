from pathlib import Path
import json

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ============================================================
# PSAI EMBEDDING PIPELINE v1
#
# Input:
#   D:\fall26\DSP_PROJECT\data_chunked\<doc_id>\<doc_id>.jsonl
#
# Output:
#   D:\fall26\DSP_PROJECT\data_embedded\<doc_id>\<doc_id>.npy
#   D:\fall26\DSP_PROJECT\data_embedded\<doc_id>\<doc_id>.meta.json
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE = True        # L2 normalize -> cosine = dot product

RESET_OUTPUT = True


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


# ============================================================
# EMBEDDING
# ============================================================

def embed_chunks(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE,
        device=DEVICE,
    )
    return vectors.astype(np.float32)


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(jsonl_path: Path, model: SentenceTransformer) -> dict:
    doc_id = jsonl_path.parent.name

    chunks = load_chunks(jsonl_path)
    if not chunks:
        print(f"[{doc_id}] no chunks, skipped.")
        return {"doc_id": doc_id, "num_chunks": 0}

    texts = [c["text"] for c in chunks]

    print(f"[{doc_id}] embedding {len(texts)} chunks ...")
    vectors = embed_chunks(model, texts)

    out_folder = EMBED_DIR / doc_id
    out_folder.mkdir(parents=True, exist_ok=True)

    npy_file = out_folder / f"{doc_id}.npy"
    np.save(npy_file, vectors)

    meta = []
    for c in chunks:
        meta.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
            "section": c.get("section", ""),
            "heading_path": c.get("heading_path", []),
            "block_type": c.get("block_type", "section"),
            "token_count": c.get("token_count", 0),
            "hash": c.get("hash", ""),
            "text": c["text"],
        })

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
    }
    print(f"[{doc_id}] done. shape={vectors.shape}")

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PSAI EMBEDDING PIPELINE v1")
    print("=" * 75)
    print(f"CHUNK_DIR : {CHUNK_DIR}")
    print(f"EMBED_DIR : {EMBED_DIR}")
    print(f"MODEL     : {MODEL_NAME}")
    print(f"DEVICE    : {DEVICE}")
    print()

    if not CHUNK_DIR.exists():
        print("ERROR: chunk folder does not exist.")
        return

    if RESET_OUTPUT and EMBED_DIR.exists():
        import shutil
        shutil.rmtree(EMBED_DIR)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(CHUNK_DIR.rglob("*.jsonl"))
    if not jsonl_files:
        print("ERROR: no .jsonl files found.")
        return

    print(f"Found {len(jsonl_files)} chunk files.")
    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("Model loaded.")
    print()

    all_stats = []
    for jsonl_file in tqdm(jsonl_files, desc="Documents"):
        try:
            stats = process_document(jsonl_file, model)
            all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] {jsonl_file}: {e}")

    total_chunks = sum(s.get("num_chunks", 0) for s in all_stats)

    summary = {
        "model": MODEL_NAME,
        "normalize": NORMALIZE,
        "dim": next((s["dim"] for s in all_stats if "dim" in s), None),
        "num_documents": len(all_stats),
        "total_chunks": total_chunks,
        "per_document": all_stats,
    }

    summary_file = EMBED_DIR / "_summary.json"
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
    print(f"Output    : {EMBED_DIR}")
    print(f"Summary   : {summary_file}")


if __name__ == "__main__":
    main()