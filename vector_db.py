from pathlib import Path
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


# ============================================================
# PSAI VECTOR DB PIPELINE v1
#
# Input:
#   D:\fall26\DSP_PROJECT\data_embedded\<doc_id>\<doc_id>.npy
#   D:\fall26\DSP_PROJECT\data_embedded\<doc_id>\<doc_id>.meta.json
#
# Output:
#   D:\fall26\DSP_PROJECT\chroma_db\
#
# Collection:
#   name     = "psai_chunks"
#   distance = "cosine"
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EMBED_DIR = BASE_DIR / "data_embedded"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "psai_chunks"
BATCH_ADD = 128

RESET_COLLECTION = False

_MODEL = None


def get_model():
    """Nạp BGE-M3 đúng một lần."""
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        import embedding as embedding_config

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = SentenceTransformer(embedding_config.MODEL_NAME, device=device)
    return _MODEL


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    meta_files = sorted(EMBED_DIR.rglob("*.meta.json"))
    if not meta_files:
        return

    for meta_file in meta_files:
        doc_id = meta_file.parent.name
        npy_file = meta_file.parent / f"{doc_id}.npy"

        if not npy_file.exists():
            print(f"[SKIP] {doc_id}: missing .npy")
            continue

        vectors = np.load(npy_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        if len(vectors) != len(meta):
            print(f"[SKIP] {doc_id}: vectors/meta mismatch")
            continue

        yield doc_id, vectors, meta


# ============================================================
# BUILD COLLECTION
# ============================================================

def build_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_added = 0

    for doc_id, vectors, meta in tqdm(load_embeddings(), desc="Documents"):
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for vec, m in zip(vectors, meta):
            ids.append(m["chunk_id"])
            embeddings.append(vec.tolist())
            metadatas.append({
                "doc_id": m.get("doc_id", doc_id),
                "chunk_index": int(m.get("chunk_index", 0)),
                "section": m.get("section", ""),
                "heading_path": " > ".join(m.get("heading_path", [])),
                "block_type": m.get("block_type", "section"),
                "token_count": int(m.get("token_count", 0)),
                "hash": m.get("hash", ""),
            })
            documents.append(m["text"])

        # xóa chunk cũ của tài liệu này (tránh còn sót id cũ khi chunk lại)
        collection.delete(where={"doc_id": doc_id})

        # add in batches
        for i in range(0, len(ids), BATCH_ADD):
            collection.upsert(
                ids=ids[i:i + BATCH_ADD],
                embeddings=embeddings[i:i + BATCH_ADD],
                metadatas=metadatas[i:i + BATCH_ADD],
                documents=documents[i:i + BATCH_ADD],
            )
            total_added += len(ids[i:i + BATCH_ADD])

        print(f"[{doc_id}] added {len(ids)} chunks")

    print()
    print(f"Total chunks in DB: {collection.count()}")

    return collection


def retrieve_context_batch(
    queries: list[tuple[str, str]], top_k: int = 5
) -> dict[str, list[dict]]:
    """Embed queries with BGE-M3 and retrieve nearest chunks per document."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not queries:
        return {}
    if any(not document_id or not query.strip() for document_id, query in queries):
        raise ValueError("Every retrieval query must have a document ID and non-empty text.")

    import embedding as embedding_config

    model = get_model()
    vectors = model.encode(
        [query for _, query in queries],
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=embedding_config.NORMALIZE,
        show_progress_bar=False,
    )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            raise ValueError(f"Vector collection {COLLECTION_NAME!r} is empty.")
        results = {}
        for (document_id, _), vector in zip(queries, vectors):
            matches = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=min(top_k, count),
                where={"doc_id": document_id},
                include=["documents", "metadatas", "distances"],
            )
            results[document_id] = [
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                }
                for chunk_id, text, metadata, distance in zip(
                    matches["ids"][0],
                    matches["documents"][0],
                    matches["metadatas"][0],
                    matches["distances"][0],
                )
            ]
    finally:
        client.close()
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PSAI VECTOR DB PIPELINE v1")
    print("=" * 75)
    print(f"EMBED_DIR : {EMBED_DIR}")
    print(f"CHROMA_DIR: {CHROMA_DIR}")
    print(f"COLLECTION: {COLLECTION_NAME}")
    print()

    if not EMBED_DIR.exists():
        print("ERROR: embedding folder does not exist.")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    collection = build_collection()

    print()
    print("=" * 75)
    print("VECTOR DB FINISHED")
    print("=" * 75)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Chunks     : {collection.count()}")
    print(f"Path       : {CHROMA_DIR}")


if __name__ == "__main__":
    main()