from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import warnings


# ============================================================
# PSAI VECTOR DB PIPELINE v2
#
# Input:
#   <repository>/data_embedded/<doc_id>/<doc_id>.npy
#   <repository>/data_embedded/<doc_id>/<doc_id>.meta.json
#
# Output:
#   <repository>/chroma_db/
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
        _MODEL = SentenceTransformer(embedding_config.MODEL_NAME, device=device, local_files_only=True)
    return _MODEL


def _open_client(path=None):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=str(path or CHROMA_DIR), settings=Settings(anonymized_telemetry=False),
    )


def _close_client(client):
    close = getattr(client, "close", None)
    if close is not None:
        close()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(embed_dir=None, *, allow_legacy=False):
    import embedding as embedding_config
    embed_dir = Path(embed_dir or EMBED_DIR)
    if not embed_dir.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {embed_dir}")
    meta_files = sorted(embed_dir.rglob("*.meta.json"))
    if not meta_files:
        raise ValueError("No embedding metadata found; no database was changed.")

    for meta_file in meta_files:
        doc_id = meta_file.parent.name
        npy_file = meta_file.parent / f"{doc_id}.npy"

        if not npy_file.exists():
            raise FileNotFoundError(f"Missing vectors: {npy_file}")

        vectors = np.load(npy_file)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        if not isinstance(meta, list) or vectors.ndim != 2 or vectors.shape[0] != len(meta) or vectors.shape[1] < 1 or not meta:
            raise ValueError(f"Invalid vectors/metadata shape for {doc_id!r}")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError(f"Non-finite or zero vectors for {doc_id!r}")
        ids = [item.get("chunk_id") for item in meta]
        if any(not isinstance(item_id, str) or not item_id for item_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Missing or duplicate chunk IDs in {meta_file}")
        if any(item.get("doc_id") != doc_id or not isinstance(item.get("text"), str) or not item["text"].strip() for item in meta):
            raise ValueError(f"Invalid chunk document/text provenance in {meta_file}")
        for item in meta:
            if not isinstance(item.get("chunk_index"), int) or isinstance(item.get("chunk_index"), bool) or item["chunk_index"] < 0:
                raise ValueError(f"Invalid chunk index in {meta_file}")
            if not isinstance(item.get("section", ""), str) or not isinstance(item.get("block_type", "section"), str):
                raise ValueError(f"Invalid section/block metadata in {meta_file}")
            if not isinstance(item.get("heading_path", []), list) or any(not isinstance(heading, str) for heading in item.get("heading_path", [])):
                raise ValueError(f"Invalid heading path in {meta_file}")
        manifest_file = meta_file.parent / f"{doc_id}.manifest.json"
        if not manifest_file.exists():
            if not allow_legacy:
                raise ValueError(f"Missing embedding manifest: {manifest_file}. Re-embed into a new directory, or explicitly allow legacy inputs.")
            warnings.warn(f"Legacy embeddings lack verified model/hash provenance: {doc_id}", RuntimeWarning)
        else:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            expected = {"model": embedding_config.MODEL_NAME, "normalize": embedding_config.NORMALIZE,
                        "dim": int(vectors.shape[1]), "num_chunks": len(meta), "doc_id": doc_id}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Embedding model/config/shape manifest mismatch: {doc_id}")
            if manifest.get("vectors_sha256") != hashlib.sha256(npy_file.read_bytes()).hexdigest() or manifest.get("metadata_sha256") != hashlib.sha256(meta_file.read_bytes()).hexdigest():
                raise ValueError(f"Embedding file hash mismatch: {doc_id}")
        if embedding_config.NORMALIZE and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError(f"Expected normalized embeddings for {doc_id}")

        yield doc_id, vectors, meta


# ============================================================
# BUILD COLLECTION
# ============================================================

def build_collection(*, embed_dir=None, chroma_dir=None, collection_name=None, allow_legacy=False):
    import embedding as embedding_config
    # Validate all inputs before opening or mutating the persisted database.
    documents = list(load_embeddings(embed_dir, allow_legacy=allow_legacy))
    all_ids = [item["chunk_id"] for _, _, metadata in documents for item in metadata]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Chunk IDs collide across documents.")
    if len({vectors.shape[1] for _, vectors, _ in documents}) != 1:
        raise ValueError("Cannot index mixed embedding dimensions.")
    collection_name = collection_name or COLLECTION_NAME
    dimension = int(documents[0][1].shape[1])
    has_legacy = any(not (Path(embed_dir or EMBED_DIR) / doc_id / f"{doc_id}.manifest.json").exists() for doc_id, _, _ in documents)
    client = _open_client(chroma_dir)

    if RESET_COLLECTION:
        _close_client(client)
        raise ValueError("Destructive collection reset is disabled; choose a new collection name.")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_config.MODEL_NAME,
                  "embedding_normalize": embedding_config.NORMALIZE, "embedding_dimension": dimension,
                  "embedding_provenance": "legacy_unverified" if has_legacy else "manifest_verified"},
    )
    metadata = collection.metadata or {}
    if metadata.get("hnsw:space", "l2") != "cosine":
        raise ValueError("Collection distance must be cosine.")
    if collection.count() and (metadata.get("embedding_model") != embedding_config.MODEL_NAME or metadata.get("embedding_normalize") != embedding_config.NORMALIZE):
        raise ValueError("Existing collection embedding provenance differs or is unknown; use a new collection name.")
    if metadata.get("embedding_dimension", dimension) != dimension:
        raise ValueError("Existing collection dimension differs; use a new collection name.")

    total_added = 0

    for doc_id, vectors, meta in documents:
        ids = []
        embeddings = []
        metadatas = []
        texts = []

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
            for key in ("source_file", "source_sha256", "source_start_line", "source_end_line", "chunking_version", "unterminated_fence"):
                if key in m:
                    metadatas[-1][key] = m[key]
            texts.append(m["text"])

        # xóa chunk cũ của tài liệu này (tránh còn sót id cũ khi chunk lại)
        old_ids = collection.get(where={"doc_id": doc_id}, include=[])["ids"]

        # add in batches
        for i in range(0, len(ids), BATCH_ADD):
            collection.upsert(
                ids=ids[i:i + BATCH_ADD],
                embeddings=embeddings[i:i + BATCH_ADD],
                metadatas=metadatas[i:i + BATCH_ADD],
                documents=texts[i:i + BATCH_ADD],
            )
            total_added += len(ids[i:i + BATCH_ADD])

        # Only remove obsolete IDs after every replacement batch succeeded.
        stale_ids = sorted(set(old_ids) - set(ids))
        if stale_ids:
            collection.delete(ids=stale_ids)

        print(f"[{doc_id}] added {len(ids)} chunks")

    print()
    print(f"Total chunks in DB: {collection.count()}")

    return collection


def retrieve_context_batch(
    queries: list[tuple[str, str]], top_k: int = 5
) -> dict[str, list[dict]]:
    """Embed queries with BGE-M3 and retrieve nearest chunks per document."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not queries:
        return {}
    if any(not document_id or not query.strip() for document_id, query in queries):
        raise ValueError("Every retrieval query must have a document ID and non-empty text.")
    if len({document_id for document_id, _ in queries}) != len(queries):
        raise ValueError("Only one retrieval query per document ID is supported; duplicate IDs would overwrite results.")

    import embedding as embedding_config

    client = _open_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            raise ValueError(f"Vector collection {COLLECTION_NAME!r} is empty.")
        metadata = collection.metadata or {}
        if metadata.get("hnsw:space", "l2") != "cosine":
            raise ValueError("Retrieval requires a cosine collection.")
        if "embedding_model" in metadata:
            if metadata["embedding_model"] != embedding_config.MODEL_NAME or metadata.get("embedding_normalize") != embedding_config.NORMALIZE:
                raise ValueError("Query model and indexed embedding provenance do not match.")
            if metadata.get("embedding_provenance") == "legacy_unverified":
                warnings.warn("Indexed vectors came from legacy inputs without verified embedding manifests.", RuntimeWarning)
        else:
            warnings.warn("Legacy collection has no verified embedding model metadata.", RuntimeWarning)
        model = get_model()
        vectors = np.asarray(model.encode(
            [query for _, query in queries], batch_size=16, convert_to_numpy=True,
            normalize_embeddings=embedding_config.NORMALIZE, show_progress_bar=False,
        ))
        if vectors.ndim != 2 or len(vectors) != len(queries) or not np.isfinite(vectors).all():
            raise ValueError("Query encoder returned invalid embeddings.")
        if vectors.shape[1] < 1 or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Query encoder returned empty or zero vectors.")
        if metadata.get("embedding_dimension", vectors.shape[1]) != vectors.shape[1]:
            raise ValueError("Query vector dimension does not match the indexed collection.")
        results = {}
        for (document_id, _), vector in zip(queries, vectors):
            document_count = len(collection.get(where={"doc_id": document_id}, include=[])["ids"])
            if document_count == 0:
                results[document_id] = []
                continue
            matches = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=min(top_k, document_count),
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
        _close_client(client)
    return results


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate embedding provenance and build a cosine collection.")
    parser.add_argument("--input-dir", type=Path, default=EMBED_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--allow-legacy", action="store_true", help="Accept legacy embeddings without verified model/hash manifests.")
    args = parser.parse_args(argv)
    print("=" * 75)
    print("PSAI VECTOR DB PIPELINE v2")
    print("=" * 75)
    print(f"EMBED_DIR : {args.input_dir}")
    print(f"CHROMA_DIR: {args.chroma_dir}")
    print(f"COLLECTION: {args.collection_name}")
    print()

    collection = build_collection(embed_dir=args.input_dir, chroma_dir=args.chroma_dir,
                                  collection_name=args.collection_name, allow_legacy=args.allow_legacy)

    print()
    print("=" * 75)
    print("VECTOR DB FINISHED")
    print("=" * 75)
    print(f"Collection : {args.collection_name}")
    print(f"Chunks     : {collection.count()}")
    print(f"Path       : {args.chroma_dir}")


if __name__ == "__main__":
    main()
