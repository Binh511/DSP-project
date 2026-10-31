"""Source-grounded retrieval with immutable local index artifacts.

BM25 uses only the standard library. Dense retrieval and reranking are explicit,
local-only operations, and never load or update the historical Chroma index.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_right
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable, Sequence
import unicodedata


VERSION = "psai-retrieval-runtime-v2"
BASE_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
_HUB_CACHE = Path(os.environ.get('HF_HUB_CACHE') or
                  (Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
DEFAULT_BASE = Path(os.environ.get('PSAI_BGE_MODEL_DIR') or
                    (_HUB_CACHE / 'models--BAAI--bge-m3' / 'snapshots' / BASE_REVISION))
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_MODES = {"bm25", "dense", "hybrid"}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    return _hash_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _tokens(value: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFKC", value).casefold())


def _positive_integer(value, name: str, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _implementation() -> dict:
    from psai_rag import data
    return {"runtime_sha256": _file_hash(Path(__file__)),
            "block_parser_sha256": _file_hash(Path(data.__file__))}


class BM25:
    """Deterministic standard-library lexical scoring in original record order."""
    def __init__(self, texts: Sequence[str], *, k1=1.5, b=0.75):
        self.counts = [Counter(_tokens(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.k1, self.b = k1, b
        frequencies = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log(1 + (len(texts) - frequency + .5) / (frequency + .5))
                    for term, frequency in frequencies.items()}
        self.postings = defaultdict(list)
        for index, counts in enumerate(self.counts):
            for term, count in counts.items():
                self.postings[term].append((index, count))

    def scores(self, query: str) -> list[float]:
        result = [0.0] * len(self.counts)
        if not self.average:
            return result
        for term in set(_tokens(query)):
            for index, count in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                result[index] += self.idf[term] * count * (self.k1 + 1) / (count + norm)
        return result


def _source(text: str, doc_id: str, *, role="document", source_file: Path | None = None) -> dict:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise ValueError("A source must contain nonempty text without NUL characters")
    if not isinstance(doc_id, str) or not doc_id.strip() or any(ord(char) < 32 for char in doc_id):
        raise ValueError("A source needs a nonempty document ID without control characters")
    if role not in {"document", "resource"}:
        raise ValueError("Source role must be document or resource")
    source_file = Path(source_file).resolve() if source_file is not None else None
    return {"doc_id": doc_id, "role": role, "text": text,
            "source_sha256": _hash_bytes(text.encode("utf-8")),
            "source_file": str(source_file) if source_file else None,
            "source_file_sha256": _file_hash(source_file) if source_file else None,
            "source_encoding": "utf-8-sig", "source_newlines": "universal"}


def _read_sources(path: Path, role: str) -> list[dict]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("Sources must be clean Markdown or UTF-8 text")
        paths, directory = [path], path.parent
    else:
        paths = sorted(item for item in path.rglob("*")
                       if item.is_file() and item.suffix.lower() in {".md", ".txt"})
        directory = path
    if not paths:
        raise ValueError(f"No Markdown/text sources found: {path}")
    result = []
    for file in paths:
        resolved = file.resolve()
        if resolved != directory and directory not in resolved.parents:
            raise ValueError("A source symlink escapes its explicitly supplied source directory")
        relative = file.relative_to(directory).with_suffix("").as_posix()
        doc_id = "resource:" + relative if role == "resource" else relative
        result.append(_source(file.read_text(encoding="utf-8-sig"), doc_id, role=role, source_file=resolved))
    return result


def _validate_sources(sources: list[dict]) -> None:
    if not sources or len({row.get("doc_id") for row in sources}) != len(sources):
        raise ValueError("Sources must be nonempty with unique document IDs")
    for row in sources:
        expected = _source(row.get("text"), row.get("doc_id"), role=row.get("role"))
        if row.get("source_sha256") != expected["source_sha256"]:
            raise ValueError(f"Source text hash mismatch: {row['doc_id']}")
        if row.get("source_file"):
            path = Path(row["source_file"])
            if not path.is_file() or _file_hash(path) != row.get("source_file_sha256"):
                raise ValueError(f"Source file changed or disappeared; rebuild the index: {path}")
            if path.read_text(encoding="utf-8-sig") != row["text"]:
                raise ValueError(f"Source file/text alignment differs: {path}")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    value = text[start:end]
    left = len(value) - len(value.lstrip())
    right = len(value) - len(value.rstrip())
    return start + left, end - right


def _dense_heading_prefix(heading_path: list[str], tokenizer, max_tokens=254) -> str:
    """Match the study heading protocol without altering any quoted source."""
    prefix = '# ' + ' > '.join(heading_path) + '\n\n' if heading_path else ''
    ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    if len(ids) > max_tokens // 3:
        prefix = tokenizer.decode(ids[:max_tokens // 3], skip_special_tokens=True).strip() + '\n\n'
    return prefix


def _chunk_sources(sources: list[dict], *, tokenizer=None, max_chars=1100, max_tokens=254) -> list[dict]:
    # Reuse structure only. Training chunks prefix headings and cannot serve as exact quotes.
    from psai_rag.data import _markdown_blocks
    _positive_integer(max_chars, "max_chars", minimum=64)
    chunks = []
    document_counts = Counter()
    for source in sources:
        text = source["text"]
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        blocks = _markdown_blocks(text)
        if not blocks:  # A heading-only document is still a legitimate source.
            blocks = [{"source_start_line": 1, "source_end_line": len(offsets) - 1,
                       "heading_path": [], "block_type": "section"}]
        for block in blocks:
            start, end = _trim_span(text, offsets[block["source_start_line"] - 1],
                                    offsets[block["source_end_line"]])
            body = text[start:end]
            if not body:
                continue
            windows = []
            prefix = _dense_heading_prefix(block['heading_path'], tokenizer, max_tokens) if tokenizer is not None else ''
            if tokenizer is None:
                cursor = 0
                while cursor < len(body):
                    stop = min(cursor + max_chars, len(body))
                    if stop < len(body):
                        boundary = max(body.rfind("\n", cursor + max_chars // 2, stop),
                                       body.rfind(" ", cursor + max_chars // 2, stop))
                        if boundary > cursor:
                            stop = boundary
                    windows.append((cursor, stop))
                    if stop == len(body):
                        break
                    cursor = max(cursor + 1, stop - min(160, (stop - cursor) // 4))
            else:
                encoded = tokenizer.encode(body, add_special_tokens=False)
                budget = max(1, max_tokens - len(tokenizer.encode(prefix, add_special_tokens=False).ids) - 2)
                cursor = 0
                while cursor < len(encoded.ids):
                    stop = min(cursor + budget, len(encoded.ids))
                    while stop > cursor:
                        left, right = encoded.offsets[cursor][0], encoded.offsets[stop - 1][1]
                        if len(tokenizer.encode(prefix + body[left:right].strip(), add_special_tokens=False).ids) <= max_tokens:
                            break
                        stop -= 1
                    if stop <= cursor:
                        raise ValueError("Cannot fit a source token within the native tokenizer budget")
                    windows.append((left, right))
                    if stop == len(encoded.ids):
                        break
                    cursor = stop - min(32, (stop - cursor) // 4)
            for left, right in windows:
                char_start, char_end = _trim_span(text, start + left, start + right)
                quote = text[char_start:char_end]
                if not quote:
                    continue
                index = document_counts[source["doc_id"]]
                document_counts[source["doc_id"]] += 1
                evidence_id = f"{source['doc_id']}:R{index:05d}:{source['source_sha256'][:12]}"
                chunks.append({"chunk_id": evidence_id, "evidence_id": evidence_id,
                               "doc_id": source["doc_id"], "role": source["role"],
                               "quote": quote, "text": quote, "char_start": char_start, "char_end": char_end,
                               "source_sha256": source["source_sha256"], "quote_sha256": _hash_bytes(quote.encode("utf-8")),
                               "heading_path": block["heading_path"], "block_type": block["block_type"],
                               "source_start_line": bisect_right(offsets, char_start),
                               "source_end_line": bisect_right(offsets, char_end - 1)})
                if tokenizer is not None:
                    chunks[-1]['embedding_prefix'] = prefix
                    chunks[-1]['embedding_text'] = prefix + quote
                    chunks[-1]['embedding_tokens_without_special_tokens'] = len(
                        tokenizer.encode(prefix + quote, add_special_tokens=False).ids)
    if not chunks:
        raise ValueError("No nonempty source chunks")
    return chunks


def _validate_chunks(chunks: list[dict], sources: list[dict]) -> None:
    by_id = {source["doc_id"]: source for source in sources}
    if not chunks or len({row.get("chunk_id") for row in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique and the corpus nonempty")
    for row in chunks:
        source = by_id.get(row.get("doc_id"))
        start, end = row.get("char_start"), row.get("char_end")
        if (source is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source["text"])
                or row.get("quote") != source["text"][start:end] or row.get("text") != row.get("quote")
                or row.get("source_sha256") != source["source_sha256"] or row.get("role") != source["role"]
                or row.get("quote_sha256") != _hash_bytes(row["quote"].encode("utf-8"))
                or row.get("evidence_id") != row.get("chunk_id")
                or not row["evidence_id"].startswith(source["doc_id"] + ":R")
                or not isinstance(row.get("heading_path"), list)
                or any(not isinstance(value, str) for value in row["heading_path"])):
            raise ValueError("Chunk source, citation, text hash or character spans do not align")
        if 'embedding_text' in row and (not isinstance(row.get('embedding_prefix'), str)
                or row['embedding_text'] != row['embedding_prefix'] + row['quote']
                or type(row.get('embedding_tokens_without_special_tokens')) is not int
                or not 1 <= row['embedding_tokens_without_special_tokens'] <= 254):
            raise ValueError('Dense embedding input differs from its declared heading and source quote')


def _model_files(path: Path, *, adapter=False, teacher=False) -> dict:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if not adapter and not re.fullmatch(r"[0-9a-f]{40}", path.name):
        raise ValueError("Models must use a pinned local 40-character snapshot revision")
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "sentencepiece.bpe.model", "vocab.txt", "modules.json", "sentence_bert_config.json",
             "1_Pooling/config.json", "2_Normalize/config.json"}
    if adapter:
        names = {"adapter_config.json", "base_provenance.json"}
        weights = list(path.glob("adapter_model*.safetensors"))
        if not weights or not (path / "adapter_config.json").is_file():
            raise ValueError("Adapter must contain adapter_config.json and adapter-only safetensors weights")
    else:
        required = {"config.json", "tokenizer_config.json"}
        if not teacher:
            required |= {"tokenizer.json", "modules.json", "1_Pooling/config.json"}
        if any(not (path / name).is_file() for name in required):
            raise ValueError("Pinned model/tokenizer snapshot is incomplete")
        weights = list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))
        if not weights:
            raise ValueError("Pinned model snapshot contains no weights")
        configuration = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if teacher:
            if configuration.get("model_type") != "bert" or configuration.get("hidden_size") != 384 or configuration.get("num_hidden_layers") != 6:
                raise ValueError("Reranking requires the explicitly pinned MiniLM-L6 MS MARCO cross-encoder")
        else:
            pooling = json.loads((path / "1_Pooling/config.json").read_text(encoding="utf-8"))
            modules = json.loads((path / "modules.json").read_text(encoding="utf-8"))
            if (configuration.get("model_type") != "xlm-roberta" or configuration.get("hidden_size") != 1024
                    or not pooling.get("pooling_mode_cls_token") or any(value for key, value in pooling.items()
                        if key.startswith("pooling_mode_") and key != "pooling_mode_cls_token")
                    or not any(row.get("type", "").endswith(".Normalize") for row in modules)):
                raise ValueError("Dense retrieval requires normalized CLS BGE-M3, dimension 1024")
    files = {name: _file_hash(path / name) for name in sorted(names) if (path / name).is_file()}
    files.update({file.name: _file_hash(file) for file in sorted(weights)})
    return {"path": str(path), "revision": None if adapter else path.name,
            "files_sha256": files, "sha256": _digest(files)}


def _encoder_provenance(base_model_dir: Path | None, adapter_dir: Path | None) -> dict:
    base_path = Path(base_model_dir or DEFAULT_BASE).resolve()
    base = _model_files(base_path)
    adapter = _model_files(Path(adapter_dir), adapter=True) if adapter_dir is not None else None
    if adapter is not None:
        configuration = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
        declared = configuration.get("base_model_name_or_path")
        binding_path = Path(adapter_dir) / 'base_provenance.json'
        if binding_path.is_file():
            binding = json.loads(binding_path.read_text(encoding='utf-8'))
            if (binding.get('model_id') != 'BAAI/bge-m3' or binding.get('revision') != base['revision']
                    or binding.get('files_sha256') != base['files_sha256']):
                raise ValueError('Portable adapter base hashes do not match the supplied local encoder')
        elif not isinstance(declared, str) or Path(declared).resolve() != base_path:
            raise ValueError("Adapter base identity does not match; a relocated export requires base_provenance.json")
        if configuration.get("peft_type") != "LORA" or configuration.get("task_type") != "FEATURE_EXTRACTION":
            raise ValueError("Only an explicit feature-extraction LoRA adapter is supported")
    from psai_rag import evaluate
    return {"base": base, "adapter": adapter, "dimension": 1024, "max_length": 256,
            "pooling": "CLS", "normalization_dtype": "float32", "cls_cast_before_normalization": True,
            "dense_input": "study_v1_heading_prefix_then_exact_source_quote",
            "encoder_code_sha256": _file_hash(Path(evaluate.__file__))}


def _validate_vectors(values, rows: int):
    import numpy as np
    array = np.asarray(values, dtype=np.float32)
    if (array.shape != (rows, 1024) or not np.isfinite(array).all()
            or not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-3)):
        raise ValueError("Dense vectors must align with chunks, be finite, normalized and 1024-dimensional")
    return array


def _encode(texts: list[str], encoder: dict, device: str) -> tuple[object, dict]:
    from psai_rag.evaluate import _DenseEncoder
    model = _DenseEncoder(Path(encoder["base"]["path"]),
                          Path(encoder["adapter"]["path"]) if encoder["adapter"] else None,
                          {"device": device, "eval_batch_size": 1, "max_length": 256})
    try:
        values = _validate_vectors(model.encode(texts), len(texts))
        model.synchronize()
        return values, model.summary()
    finally:
        model.close()
        gc.collect()


def _create(sources: list[dict], *, mode: str, base_model_dir=None, adapter_dir=None,
            device="auto", max_chars=1100, prepared_queries=None):
    if mode not in _MODES:
        raise ValueError("mode must be bm25, dense or hybrid")
    if mode == "bm25" and adapter_dir is not None:
        raise ValueError("An adapter is only used by explicit dense/hybrid retrieval")
    _validate_sources(sources)
    encoder, tokenizer = None, None
    if mode != "bm25":
        encoder = _encoder_provenance(base_model_dir, adapter_dir)
        from psai_rag.data import _load_tokenizer
        tokenizer = _load_tokenizer(Path(encoder["base"]["path"]) / "tokenizer.json")
    chunks = _chunk_sources(sources, tokenizer=tokenizer, max_chars=max_chars)
    _validate_chunks(chunks, sources)
    vectors, runtime, prepared = None, None, None
    if encoder is not None:
        values, runtime = _encode([row["embedding_text"] for row in chunks] + list(prepared_queries or []), encoder, device)
        vectors = values[:len(chunks)]
        if prepared_queries is not None:
            prepared = (list(prepared_queries), values[len(chunks):], runtime)
    manifest = {"schema_version": 1, "version": VERSION, "implementation": _implementation(),
                "source_count": len(sources), "chunk_count": len(chunks), "corpus_sha256": _digest(chunks),
                "sources_sha256": _digest(sources), "encoder": encoder, "encoding_runtime": runtime,
                "chunking": {"kind": "unicode_character_windows" if encoder is None else "native_bge_token_windows",
                             "max_chars": max_chars if encoder is None else None,
                             "max_tokens_without_special_tokens": 254 if encoder is not None else None,
                             "quotes_are_exact_source_spans": True},
                "lexical": {"algorithm": "BM25", "k1": 1.5, "b": .75,
                            "tokenizer": "NFKC_casefold_unicode_words_v1",
                            "document": "heading_path_then_exact_source_quote"}}
    manifest["manifest_sha256"] = _digest(manifest)
    return RetrievalIndex(sources, chunks, manifest, vectors=vectors, prepared=prepared)


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Index JSONL records must be objects")
    return rows


def _verify_artifacts(directory: Path, manifest: dict) -> None:
    expected_names = {"sources.jsonl", "chunks.jsonl"} | ({"vectors.npy"} if manifest.get("encoder") else set())
    files = manifest.get("files_sha256", {})
    if set(files) != expected_names:
        raise ValueError("Index artifact manifest has missing or unexpected files")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or _file_hash(path) != expected:
            raise ValueError(f"Index artifact SHA256 mismatch: {name}")


def build_index(input_path: Path, output_dir: Path, *, resource_dir: Path | None = None,
                mode="bm25", base_model_dir=None, adapter_dir=None, device="auto", max_chars=1100) -> dict:
    """Build a fresh index from a Markdown file/directory and optional resources."""
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a new index output directory; existing indexes are never overwritten")
    for source_path in [input_path] + ([Path(resource_dir).resolve()] if resource_dir is not None else []):
        if output_dir == source_path or output_dir in source_path.parents or source_path in output_dir.parents:
            raise ValueError("Index output and input source trees must not overlap")
    sources = _read_sources(input_path, "document")
    if resource_dir is not None:
        sources += _read_sources(Path(resource_dir), "resource")
    index = _create(sources, mode=mode, base_model_dir=base_model_dir, adapter_dir=adapter_dir,
                    device=device, max_chars=max_chars)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".psai-retrieval-", dir=output_dir.parent))
    try:
        for name, rows in (("sources.jsonl", index.sources), ("chunks.jsonl", index.chunks)):
            with (stage / name).open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        files = ["sources.jsonl", "chunks.jsonl"]
        if index.vectors is not None:
            import numpy as np
            np.save(stage / "vectors.npy", index.vectors, allow_pickle=False)
            files.append("vectors.npy")
        manifest = {**index.manifest, "files_sha256": {name: _file_hash(stage / name) for name in files}}
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = _digest(manifest)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            if stage.resolve().parent != output_dir.parent or not stage.name.startswith(".psai-retrieval-"):
                raise RuntimeError("Refusing cleanup outside the verified index staging directory")
            shutil.rmtree(stage)
    return {"index_dir": str(output_dir), "manifest": manifest}


def load_index(index_dir: Path):
    """Check artifact hashes, exact citation spans and the live source files."""
    index_dir = Path(index_dir).resolve()
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _digest(unsigned):
        raise ValueError("Index manifest SHA256 mismatch")
    if manifest.get("version") != VERSION or manifest.get("implementation") != _implementation():
        raise ValueError("Index/runtime implementation differs; rebuild the index")
    _verify_artifacts(index_dir, manifest)
    sources, chunks = _jsonl(index_dir / "sources.jsonl"), _jsonl(index_dir / "chunks.jsonl")
    _validate_sources(sources)
    _validate_chunks(chunks, sources)
    if (len(sources) != manifest.get("source_count") or len(chunks) != manifest.get("chunk_count")
            or _digest(sources) != manifest.get("sources_sha256") or _digest(chunks) != manifest.get("corpus_sha256")):
        raise ValueError("Index corpus/source manifest does not align")
    vectors = None
    if manifest.get("encoder"):
        import numpy as np
        vectors = _validate_vectors(np.load(index_dir / "vectors.npy", allow_pickle=False), len(chunks))
    return RetrievalIndex(sources, chunks, manifest, vectors=vectors, index_dir=index_dir,
                          manifest_file_sha256=_file_hash(manifest_path))


def _queries(queries: str | Sequence[str]) -> list[str]:
    values = [queries] if isinstance(queries, str) else list(queries)
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Provide at least one nonempty retrieval query")
    result, seen = [], set()
    for value in values:
        value = value.strip()
        key = " ".join(value.casefold().split())
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _rerank(pairs: list[tuple[str, str]], model_dir: Path) -> tuple[list[float], dict]:
    provenance = _model_files(model_dir, teacher=True)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True,
                                                               trust_remote_code=False, torch_dtype=torch.float32)
    model.to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 8):
                current = pairs[start:start + 8]
                tokens = tokenizer([pair[0] for pair in current], [pair[1] for pair in current],
                                   padding=True, truncation=True, max_length=384, return_tensors="pt")
                logits = model(**tokens).logits
                if logits.ndim != 2 or logits.shape[1] != 1 or not torch.isfinite(logits).all():
                    raise ValueError("Reranker must return one finite relevance logit per query/passage")
                scores.extend(logits[:, 0].float().tolist())
        return scores, {**provenance, "device": "cpu", "dtype": "float32", "max_length": 384,
                        "score_kind": "uncalibrated_cross_encoder_logit", "language_scope": "English_MS_MARCO"}
    finally:
        del model, tokenizer
        gc.collect()


class RetrievalIndex:
    def __init__(self, sources, chunks, manifest, *, vectors=None, index_dir=None, manifest_file_sha256=None, prepared=None):
        self.sources, self.chunks, self.manifest = sources, chunks, manifest
        self.vectors, self.index_dir = vectors, index_dir
        self.manifest_file_sha256 = manifest_file_sha256
        self._prepared = prepared
        self.lexical = BM25(["\n".join(row["heading_path"] + [row["quote"]]) for row in chunks])

    def search(self, queries: str | Sequence[str], *, top_k=3, mode="bm25",
               document_ids: Iterable[str] | None = None, include_resources=False,
               base_model_dir=None, adapter_dir=None, device="auto", rerank_model_dir=None,
               minimum_dense_score=0.0) -> dict:
        _positive_integer(top_k, "top_k")
        if mode not in _MODES:
            raise ValueError("mode must be bm25, dense or hybrid")
        if not isinstance(minimum_dense_score, (int, float)) or not math.isfinite(minimum_dense_score):
            raise ValueError("minimum_dense_score must be finite")
        queries = _queries(queries)
        _validate_sources(self.sources)
        _validate_chunks(self.chunks, self.sources)
        if self.index_dir is not None:
            if _file_hash(self.index_dir / "manifest.json") != self.manifest_file_sha256:
                raise ValueError("Index manifest changed since loading")
            _verify_artifacts(self.index_dir, self.manifest)
        requested = ({document_ids} if isinstance(document_ids, str) else set(document_ids)) if document_ids is not None else None
        known = {row["doc_id"] for row in self.sources if row["role"] == "document"}
        if requested is not None and (not requested or not requested.issubset(known)):
            raise ValueError("Unknown or empty document filter; unrelated documents are never substituted")
        eligible = [index for index, row in enumerate(self.chunks)
                    if ((row["role"] == "document" and (requested is None or row["doc_id"] in requested))
                        or (include_resources and row["role"] == "resource"))]
        if not eligible:
            raise ValueError("No chunks match the requested document/resource scope")
        dense_scores, runtime = None, None
        if mode != "bm25":
            indexed = self.manifest.get("encoder")
            if indexed is None or self.vectors is None:
                raise ValueError("This lexical-only index has no aligned dense vectors; build a fresh dense index")
            prepared = self._prepared if self._prepared is not None and self._prepared[0] == queries else None
            if prepared is not None:
                if (Path(base_model_dir or indexed["base"]["path"]).resolve() != Path(indexed["base"]["path"])
                        or (str(Path(adapter_dir).resolve()) if adapter_dir is not None else None)
                        != (indexed["adapter"]["path"] if indexed["adapter"] else None)):
                    raise ValueError("Query encoder paths differ from the prepared in-memory encoder")
                actual = indexed
            else:
                actual = _encoder_provenance(base_model_dir or indexed["base"]["path"], adapter_dir)
            # An indexed adapter must be supplied explicitly at search, never silently activated.
            if (actual["base"]["sha256"] != indexed["base"]["sha256"]
                    or (actual["adapter"]["sha256"] if actual["adapter"] else None)
                    != (indexed["adapter"]["sha256"] if indexed["adapter"] else None)
                    or actual["encoder_code_sha256"] != indexed["encoder_code_sha256"]):
                raise ValueError("Query model/tokenizer/adapter does not match the indexed vectors")
            query_vectors, runtime = (prepared[1], prepared[2]) if prepared is not None else _encode(queries, actual, device)
            dense_scores = query_vectors @ self.vectors.T
        elif adapter_dir is not None:
            raise ValueError("An adapter is unused in BM25 mode; request dense or hybrid explicitly")
        rankings = []
        pool_size = max(top_k, min(40, top_k * 4)) if rerank_model_dir is not None else top_k
        for query_index, query in enumerate(queries):
            lexical = self.lexical.scores(query)
            lexical_order = sorted((i for i in eligible if lexical[i] > 0),
                                   key=lambda i: (-lexical[i], self.chunks[i]["chunk_id"]))
            dense = dense_scores[query_index] if dense_scores is not None else None
            dense_order = sorted((i for i in eligible if float(dense[i]) >= minimum_dense_score),
                                 key=lambda i: (-float(dense[i]), self.chunks[i]["chunk_id"])) if dense is not None else []
            if mode == "bm25":
                scores, order = {i: lexical[i] for i in lexical_order}, lexical_order
            elif mode == "dense":
                scores, order = {i: float(dense[i]) for i in dense_order}, dense_order
            else:
                scores = defaultdict(float)
                for values in (lexical_order, dense_order):
                    for rank, index in enumerate(values, 1):
                        scores[index] += 1 / (60 + rank)
                order = sorted(scores, key=lambda i: (-scores[i], self.chunks[i]["chunk_id"]))
            rankings.append([{"index": i, "score": float(scores[i]), "bm25_score": lexical[i],
                              "dense_score": float(dense[i]) if dense is not None else None,
                              "rerank_score": None, "query": query, "query_index": query_index}
                             for i in order[:pool_size]])
        reranker = None
        if rerank_model_dir is not None and any(rankings):
            flat = [row for rows in rankings for row in rows]
            scores, reranker = _rerank([(row["query"], self.chunks[row["index"]]["quote"]) for row in flat], Path(rerank_model_dir))
            if len(scores) != len(flat) or any(not math.isfinite(value) for value in scores):
                raise ValueError('Reranker result count and finite scores must match every candidate')
            for row, score in zip(flat, scores):
                row["rerank_score"], row["score"] = score, score
            for rows in rankings:
                rows.sort(key=lambda row: (-row["score"], self.chunks[row["index"]]["chunk_id"]))
        combined, matches = defaultdict(float), defaultdict(list)
        for rows in rankings:
            for rank, row in enumerate(rows[:top_k], 1):
                combined[row["index"]] += 1 / (60 + rank) if len(queries) > 1 else row["score"]
                matches[row["index"]].append({key: value for key, value in {**row, "rank": rank}.items() if key != "index"})
        selected = sorted(combined, key=lambda i: (-combined[i], self.chunks[i]["chunk_id"]))[:top_k]
        by_id = {row["doc_id"]: row for row in self.sources}
        hits = []
        for index in selected:
            row = self.chunks[index]
            source = by_id[row["doc_id"]]
            hits.append({**row, "score": combined[index], "query_matches": matches[index],
                         "source_file": source["source_file"], "source_file_sha256": source["source_file_sha256"],
                         "source_text_sha256": source["source_sha256"], "source_encoding": source["source_encoding"],
                         "source_newlines": source["source_newlines"], "verified": True,
                         "source_grounding_verified": True, "relevance_verified": False})
        return {"mode": mode, "queries": queries, "hits": hits,
                "scope": {"document_ids": sorted(requested) if requested is not None else sorted(known),
                          "include_resources": bool(include_resources), "candidate_count": len(eligible)},
                "provenance": {"version": VERSION, "index_sha256": self.manifest["manifest_sha256"],
                               "corpus_sha256": self.manifest["corpus_sha256"], "encoder": self.manifest["encoder"] if mode != "bm25" else None,
                               "index_encoding_runtime": self.manifest.get('encoding_runtime') if mode != 'bm25' else None,
                               "encoding_runtime": runtime, "reranker": reranker,
                               "query_fusion": "RRF_k60" if len(queries) > 1 else "single_query",
                               "note": "verified proves exact source grounding; relevance is an unverified retrieval estimate."}}


def search_index(index_dir: Path, queries: str | Sequence[str], **kwargs) -> dict:
    return load_index(index_dir).search(queries, **kwargs)


def retrieve_document(text: str, doc_id: str, *, queries: str | Sequence[str], top_k=3,
                      mode="bm25", resource_dir: Path | None = None, base_model_dir=None,
                      adapter_dir=None, device="auto", rerank_model_dir=None) -> dict:
    """Retrieve exact-source evidence from one input and optional local resources.

    This API writes no files. Source SHA256 and character offsets refer to the
    exact decoded input text, with external resources identified separately.
    """
    queries = _queries(queries)
    sources = [_source(text, doc_id)]
    if resource_dir is not None:
        sources += _read_sources(Path(resource_dir), "resource")
    index = _create(sources, mode=mode, base_model_dir=base_model_dir, adapter_dir=adapter_dir, device=device,
                    prepared_queries=queries)
    return index.search(queries, top_k=top_k, mode=mode, document_ids=[doc_id],
                        include_resources=resource_dir is not None, base_model_dir=base_model_dir,
                        adapter_dir=adapter_dir, device=device, rerank_model_dir=rerank_model_dir)


def build_evidence(text: str, document_id: str, queries: str | Sequence[str], *, backend="bm25", **kwargs) -> dict:
    """Compatibility spelling for callers that describe retrieval as evidence building."""
    return retrieve_document(text, document_id, queries=queries, mode=backend, **kwargs)
