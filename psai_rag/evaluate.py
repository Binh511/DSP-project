"""Isolated retrieval evaluation against source-chunk pseudopositives.

Only NumPy and the standard library are imported at module import time. Model
dependencies are loaded only by ``evaluate_retriever``. These metrics measure
weak source-chunk labels, never human relevance or end-to-end planning quality.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable, Mapping, Sequence
import unicodedata

import numpy as np


EVALUATION_VERSION = "psai-rag-retrieval-evaluation-v1"
WEAK_LABEL_NOTICE = (
    "Original source chunks are pseudopositives (weak labels), not human ground "
    "truth. Retrieval scores do not establish planning or PDF/OCR quality."
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def tokenize(text: str) -> list[str]:
    """Unicode word tokens for the reproducible lexical baseline."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


class BM25:
    """Small deterministic BM25 index over chunk records or plain strings.

    ``scores``/``get_scores`` return scores in input order. ``rank`` and
    ``negative_candidates`` return copies of records with ``bm25_score`` added.
    Candidate IDs always refer to ``chunk_id`` (not document IDs).
    """

    def __init__(self, documents: Sequence[Mapping | str], *, k1: float = 1.5,
                 b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = float(k1), float(b)
        self.documents = [
            {"chunk_id": str(index), "text": value}
            if isinstance(value, str) else dict(value)
            for index, value in enumerate(documents)
        ]
        if any(not isinstance(row.get("chunk_id"), str) or not row["chunk_id"]
               or not isinstance(row.get("text"), str) for row in self.documents):
            raise ValueError("BM25 records need string chunk_id and text fields")
        self.ids = [row["chunk_id"] for row in self.documents]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("BM25 chunk IDs must be unique")
        self.id_to_index = {value: index for index, value in enumerate(self.ids)}
        self.term_frequencies = [Counter(tokenize(row["text"])) for row in self.documents]
        self.lengths = np.asarray([sum(row.values()) for row in self.term_frequencies], dtype=np.float64)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        document_frequency = Counter(term for row in self.term_frequencies for term in row)
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._postings = defaultdict(list)
        for index, counts in enumerate(self.term_frequencies):
            for term, frequency in counts.items():
                self._postings[term].append((index, frequency))

    def scores(self, query: str | Sequence[str]) -> np.ndarray:
        terms = tokenize(query) if isinstance(query, str) else list(query)
        scores = np.zeros(len(self.documents), dtype=np.float64)
        if not self.average_length:
            return scores
        normalizers = self.k1 * (1.0 - self.b + self.b * self.lengths / self.average_length)
        for term in set(terms):
            for index, frequency in self._postings.get(term, ()):
                scores[index] += self.idf[term] * frequency * (self.k1 + 1) / (frequency + normalizers[index])
        return scores

    get_scores = scores

    def rank(self, query: str, top_k: int | None = None,
             candidate_ids: Iterable[str] | None = None) -> list[dict]:
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        candidates = set(candidate_ids) if candidate_ids is not None else None
        scores = self.scores(query)
        indices = [index for index, value in enumerate(self.ids)
                   if candidates is None or value in candidates]
        indices.sort(key=lambda index: (-float(scores[index]), self.ids[index]))
        if top_k is not None:
            indices = indices[:top_k]
        return [{**self.documents[index], "bm25_score": float(scores[index])} for index in indices]

    def negative_candidates(self, query: str, positive_id: str,
                            known_positive_ids: Iterable[str] = (), *, top_k: int = 10,
                            candidate_ids: Iterable[str] | None = None) -> list[dict]:
        """Mine lexical negatives without treating known/duplicate positives as negatives."""
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        positives = {positive_id, *known_positive_ids}
        if positive_id not in self.id_to_index:
            raise ValueError(f"Unknown positive chunk: {positive_id}")
        positive_texts = {
            _normalized_text(self.documents[self.id_to_index[value]]["text"])
            for value in positives if value in self.id_to_index
        }
        selected, seen_texts = [], set(positive_texts)
        for row in self.rank(query, candidate_ids=candidate_ids):
            key = _normalized_text(row["text"])
            if row["chunk_id"] in positives or key in seen_texts:
                continue
            selected.append(row)
            seen_texts.add(key)
            if len(selected) >= top_k:
                break
        return selected if top_k else []


def ranking_metrics(ranked_ids: Sequence[str], positive_ids: Iterable[str]) -> dict[str, float]:
    """Binary-relevance recall, truncated MRR, and nDCG; supports multiple positives."""
    positives = set(positive_ids)
    if not positives:
        raise ValueError("At least one positive is required")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("A ranking must not repeat a chunk")
    result = {f"Recall@{k}": len(set(ranked_ids[:k]) & positives) / len(positives)
              for k in (1, 5, 10)}
    ranks = [rank for rank, value in enumerate(ranked_ids[:10], 1) if value in positives]
    result["MRR@10"] = 1.0 / ranks[0] if ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(positives)) + 1))
    result["nDCG@10"] = dcg / ideal
    return result


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    if type(k) is not int or k < 1:
        raise ValueError("RRF k must be a positive integer")
    scores = defaultdict(float)
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("RRF inputs must not repeat chunks")
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _directory_provenance(path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt"}
    files = {item.relative_to(path).as_posix(): _sha256(item)
             for item in sorted(path.rglob("*")) if item.is_file() and item.suffix in suffixes}
    if not files:
        raise ValueError(f"No model/tokenizer artifacts found under {path}")
    return {"path": str(path), "files": files, "sha256": _digest_json(files)}


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield row


def _load_validation_inputs(corpus_path: Path, pairs_path: Path) -> tuple[list[dict], list[dict]]:
    corpus = [row for row in _read_jsonl(corpus_path) if row.get("split") == "validation"]
    if not corpus:
        raise ValueError("No validation chunks; train/test chunks are never evaluated")
    required = ("chunk_id", "doc_id", "source_group_id", "text")
    if any(any(not isinstance(row.get(key), str) or not row[key].strip() for key in required)
           for row in corpus):
        raise ValueError("Validation chunks require chunk_id, doc_id, source_group_id and text")
    corpus.sort(key=lambda row: row["chunk_id"])
    by_id = {row["chunk_id"]: row for row in corpus}
    if len(by_id) != len(corpus):
        raise ValueError("Duplicate validation chunk IDs")
    grouped = {}
    for row in _read_jsonl(pairs_path):
        if row.get("split", "validation") != "validation":
            raise ValueError("The validation query file contains a non-validation query")
        for key in ("query_id", "query", "positive_id", "doc_id", "source_group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Validation query requires non-empty {key}")
        positive = by_id.get(row["positive_id"])
        if positive is None:
            raise ValueError(f"Positive {row['positive_id']} is outside the validation corpus")
        if any(row[key] != positive[key] for key in ("doc_id", "source_group_id")):
            raise ValueError("Query/positive document or source group does not match")
        for key in ("positive", "positive_text"):
            if key in row and row[key] != positive["text"]:
                raise ValueError("Query positive text differs from the source chunk")
        known = row.get("known_positive_ids", row.get("positive_ids", []))
        if not isinstance(known, list) or any(not isinstance(value, str) for value in known):
            raise ValueError("Known positive IDs must be a list of strings")
        positives = {row["positive_id"], *known}
        if not positives.issubset(by_id):
            raise ValueError("Known positives must all belong to the validation corpus")
        if any(by_id[value]["doc_id"] != row["doc_id"] for value in positives):
            raise ValueError("Known positives must belong to the query document for within_doc evaluation")
        current = grouped.get(row["query_id"])
        if current is None:
            grouped[row["query_id"]] = {**row, "positive_ids": sorted(positives)}
        else:
            if any(current[key] != row[key] for key in ("query", "doc_id", "source_group_id")):
                raise ValueError("A query ID refers to inconsistent queries or source groups")
            current["positive_ids"] = sorted(set(current["positive_ids"]) | positives)
    if not grouped:
        raise ValueError("No validation queries")
    return corpus, [grouped[key] for key in sorted(grouped)]


class _DenseEncoder:
    def __init__(self, base_model_dir: Path, adapter_dir: Path | None, config: dict):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        requested = config.get("device", "auto")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Evaluation device must be auto, cuda or cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This evaluation requires BF16-capable CUDA")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.batch_size = int(config.get("eval_batch_size", 1))
        self.max_length = int(config.get("max_length", config.get("max_seq_length", 256)))
        if self.batch_size not in (1, 4) or not 1 <= self.max_length <= 256:
            raise ValueError("Use evaluation batch size 1 or 4 and max length <= 256")
        pool_config = Path(base_model_dir) / "1_Pooling" / "config.json"
        if pool_config.is_file():
            pooling = json.loads(pool_config.read_text(encoding="utf-8"))
            if not pooling.get("pooling_mode_cls_token") or any(
                value for key, value in pooling.items()
                if key.startswith("pooling_mode_") and key != "pooling_mode_cls_token"
            ):
                raise ValueError("Pinned SentenceTransformer must use CLS-only pooling")
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModel.from_pretrained(str(base_model_dir), local_files_only=True,
                                              trust_remote_code=False, torch_dtype=self.dtype)
        if adapter_dir is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=False,
                                                   local_files_only=True)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arrays = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(list(texts[start:start + self.batch_size]), padding=True,
                                        truncation=True, max_length=self.max_length, return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                result = self.model(**tokens, return_dict=True).last_hidden_state[:, 0].float()
                result = self.torch.nn.functional.normalize(result, p=2, dim=1)
                arrays.append(result.cpu().numpy())
        return np.concatenate(arrays, axis=0)

    def summary(self) -> dict:
        return {
            "device": self.device, "dtype": str(self.dtype), "pooling": "normalized_CLS",
            "normalization_dtype": "float32", "cls_cast_before_normalization": True,
            "batch_size": self.batch_size, "max_length": self.max_length,
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated() if self.device == "cuda" else 0,
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved() if self.device == "cuda" else 0,
        }

    def close(self):
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def _make_encoder(base_model_dir, adapter_dir, config):
    return _DenseEncoder(base_model_dir, adapter_dir, config)


def _validate_vectors(vectors: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or len(values) != count or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Invalid embedding array")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-3):
        raise ValueError("Embeddings must be normalized nonzero vectors")
    return values


def _cached_embeddings(cache_dir: Path, kind: str, texts: Sequence[str], provenance: dict,
                       encoder) -> tuple[np.ndarray, dict]:
    key = _digest_json({**provenance, "texts_sha256": _digest_json(list(texts)), "kind": kind})
    path = cache_dir / f"{kind}-{key}.npy"
    manifest_path = path.with_suffix(".json")
    if path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("key") == key and manifest.get("vectors_sha256") == _sha256(path):
                vectors = _validate_vectors(np.load(path, allow_pickle=False), len(texts))
                return vectors, {"path": str(path), "key": key, "reused": True, "seconds": 0.0}
        except (OSError, ValueError, TypeError):
            pass
    encoder.synchronize()
    started = time.perf_counter()
    vectors = _validate_vectors(encoder.encode(texts), len(texts))
    encoder.synchronize()
    seconds = time.perf_counter() - started
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors, allow_pickle=False)
    manifest = {"key": key, "provenance": provenance, "kind": kind, "rows": len(texts),
                "dimension": vectors.shape[1], "vectors_sha256": _sha256(path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors, {"path": str(path), "key": key, "reused": False, "seconds": seconds}


def _rank_indices(scores: np.ndarray, indices: Sequence[int], ids: Sequence[str]) -> list[int]:
    return sorted(indices, key=lambda index: (-float(scores[index]), ids[index]))


def _evaluate_rankings(corpus: list[dict], queries: list[dict], corpus_vectors: np.ndarray,
                       query_vectors: np.ndarray, *, rrf_k: int = 60) -> tuple[dict, list[dict], dict]:
    corpus_vectors = _validate_vectors(corpus_vectors, len(corpus))
    query_vectors = _validate_vectors(query_vectors, len(queries))
    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Query and passage embedding dimensions differ")
    bm25 = BM25(corpus)
    ids = bm25.ids
    by_doc = defaultdict(list)
    for index, row in enumerate(corpus):
        by_doc[row["doc_id"]].append(index)
    metrics, predictions, timings = {}, [], defaultdict(list)
    for scope in ("within_doc", "global_validation"):
        per_method = defaultdict(list)
        for query_index, query in enumerate(queries):
            indices = by_doc[query["doc_id"]] if scope == "within_doc" else list(range(len(corpus)))
            if not indices:
                raise ValueError(f"Query {query['query_id']} has no candidates in {scope}")
            started = time.perf_counter()
            lexical_scores = bm25.scores(query["query"])
            lexical = _rank_indices(lexical_scores, indices, ids)
            timings[f"{scope}/bm25"].append(time.perf_counter() - started)
            started = time.perf_counter()
            dense_scores = corpus_vectors @ query_vectors[query_index]
            dense = _rank_indices(dense_scores, indices, ids)
            timings[f"{scope}/dense"].append(time.perf_counter() - started)
            started = time.perf_counter()
            fusion = reciprocal_rank_fusion([[ids[index] for index in lexical],
                                              [ids[index] for index in dense]], k=rrf_k)
            fused = sorted(indices, key=lambda index: (-fusion[ids[index]], ids[index]))
            timings[f"{scope}/rrf_fusion_only"].append(time.perf_counter() - started)
            rankings = {"bm25": (lexical, lexical_scores), "dense": (dense, dense_scores),
                        "bm25_dense_rrf": (fused, fusion)}
            for method, (ranking, scores) in rankings.items():
                chunk_ids = [ids[index] for index in ranking[:10]]
                values = ranking_metrics(chunk_ids, query["positive_ids"])
                per_method[method].append(values)
                predictions.append({
                    "query_id": query["query_id"], "query": query["query"], "doc_id": query["doc_id"],
                    "source_group_id": query["source_group_id"], "scope": scope, "method": method,
                    "positive_ids": query["positive_ids"], "candidate_count": len(indices), "metrics": values,
                    "ranking": [{"rank": rank, "chunk_id": ids[index],
                                 "score": float(scores[ids[index]] if method == "bm25_dense_rrf" else scores[index]),
                                 "relevant": ids[index] in query["positive_ids"]}
                                for rank, index in enumerate(ranking[:10], 1)],
                })
        metrics[scope] = {method: {name: float(np.mean([row[name] for row in rows]))
                                  for name in rows[0]} for method, rows in per_method.items()}
    latency = {key: {"mean_ms": float(np.mean(values) * 1000),
                     "p95_ms": float(np.percentile(values, 95) * 1000), "query_count": len(values)}
               for key, values in timings.items()}
    return metrics, predictions, latency


def evaluate_retriever(root: Path, study_dir: Path, base_model_dir: Path,
                       adapter_dir: Path | None, config: dict) -> dict:
    """Evaluate BM25, normalized-CLS dense retrieval, and their RRF combination.

    All candidates are from ``split == 'validation'``. Each query's original
    source chunk is a weak positive. Nothing reads or writes the production
    index, classifier checkpoints, or final-test evaluation artifacts.
    """
    root, study_dir = Path(root).resolve(), Path(study_dir).resolve()
    base_model_dir = Path(base_model_dir).resolve()
    adapter_dir = Path(adapter_dir).resolve() if adapter_dir is not None else None
    if root != study_dir and root not in study_dir.parents:
        raise ValueError("The retrieval study must be inside the workspace")
    corpus_path, pairs_path = study_dir / "corpus.jsonl", study_dir / "validation_pairs.jsonl"
    corpus, queries = _load_validation_inputs(corpus_path, pairs_path)
    base = _directory_provenance(base_model_dir)
    adapter = _directory_provenance(adapter_dir) if adapter_dir is not None else None
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "peft", "numpy"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    variant = "adapter" if adapter_dir is not None else "base"
    rrf_k = config.get("rrf_k", 60)
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    start = time.perf_counter()
    encoder = _make_encoder(base_model_dir, adapter_dir, config)
    encoder.synchronize()
    model_load_seconds = time.perf_counter() - start
    try:
        runtime = encoder.summary()
        inference = {key: runtime[key] for key in ("device", "dtype", "pooling", "batch_size", "max_length",
                                                  "normalization_dtype", "cls_cast_before_normalization")}
        provenance = {"evaluation_version": EVALUATION_VERSION, "evaluation_code_sha256": _sha256(Path(__file__)),
                      "base_sha256": base["sha256"],
                      "adapter_sha256": adapter["sha256"] if adapter else None,
                      "validation_corpus_sha256": _digest_json(corpus), "inference": inference,
                      "package_versions": versions}
        passage_vectors, passage_cache = _cached_embeddings(
            study_dir / "cache", f"{variant}-validation-passages", [row["text"] for row in corpus], provenance, encoder)
        query_vectors, query_cache = _cached_embeddings(
            study_dir / "cache", f"{variant}-validation-queries", [row["query"] for row in queries], provenance, encoder)
        runtime = encoder.summary()
    finally:
        encoder.close()
    metrics, predictions, latency = _evaluate_rankings(corpus, queries, passage_vectors,
                                                       query_vectors, rrf_k=rrf_k)
    query_encoding_mean_ms = query_cache["seconds"] * 1000 / len(queries)
    retrieval_mean_ms = {}
    for scope in ("within_doc", "global_validation"):
        lexical_ms = latency[f"{scope}/bm25"]["mean_ms"]
        dense_ms = latency[f"{scope}/dense"]["mean_ms"]
        fusion_ms = latency[f"{scope}/rrf_fusion_only"]["mean_ms"]
        retrieval_mean_ms[scope] = {
            "bm25": lexical_ms,
            "dense": query_encoding_mean_ms + dense_ms,
            "bm25_dense_rrf": query_encoding_mean_ms + lexical_ms + dense_ms + fusion_ms,
        }
    prediction_path = study_dir / f"{variant}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = study_dir / f"{variant}_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "doc_id", "scope", "method", "rank",
                                                    "chunk_id", "score", "relevant", "positive_ids"])
        writer.writeheader()
        for row in predictions:
            for hit in row["ranking"]:
                writer.writerow({**{key: row[key] for key in ("query_id", "doc_id", "scope", "method")},
                                 **hit, "positive_ids": json.dumps(row["positive_ids"], ensure_ascii=False)})
    report = {
        "evaluation_version": EVALUATION_VERSION, "variant": variant, "label_kind": "source_chunk_pseudopositive",
        "weak_label_notice": WEAK_LABEL_NOTICE, "primary_scope": "within_doc",
        "candidate_split": "validation", "validation_chunks": len(corpus), "validation_queries": len(queries),
        "validation_documents": len({row["doc_id"] for row in corpus}), "rrf_k": rrf_k,
        "metrics": metrics, "latency": {"ranking_cpu": latency, "model_load_seconds": model_load_seconds,
                                          "passage_encoding_seconds": passage_cache["seconds"],
                                          "query_encoding_seconds": query_cache["seconds"],
                                          "query_encoding_mean_ms": query_encoding_mean_ms,
                                          "retrieval_mean_ms": retrieval_mean_ms,
                                          "note": "Retrieval means add amortized query encoding to CPU ranking; "
                                                  "exclude model loading and offline passage encoding. "
                                                  "Cached query encoding time is zero and is not an inference benchmark."},
        "runtime": runtime, "base_model": base, "adapter": adapter, "cache_provenance": provenance,
        "inputs": {"corpus_file_sha256": _sha256(corpus_path), "validation_pairs_sha256": _sha256(pairs_path)},
        "embedding_cache": {"passages": passage_cache, "queries": query_cache},
        "outputs": {"predictions_jsonl": str(prediction_path), "predictions_csv": str(csv_path)},
    }
    metrics_path = study_dir / f"{variant}_metrics.json"
    report["outputs"]["metrics_json"] = str(metrics_path)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
