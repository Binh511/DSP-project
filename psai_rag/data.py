"""Prepare isolated, source-grouped retrieval data without loading any model.

Preparation is explicit: importing this module neither writes files nor reads a
test-label artifact. All outputs belong to the supplied study directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NEAR_DUPLICATE_THRESHOLD = 0.85
NEAR_DUPLICATE_LENGTH_RATIO = 0.8
SHINGLE_WORDS = 5
OVERLAP_TOKENS = 32
MIN_CANDIDATE_TOKENS = 32


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _load_tokenizer(path: Path) -> Any:
    # Rust tokenizers is deliberately the only optional dependency here.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def _word_shingles(text: str) -> tuple[int, frozenset[bytes]]:
    words = re.findall(r"\w+", _normalize(text))
    width = min(SHINGLE_WORDS, len(words))
    if not width:
        return 0, frozenset()
    shingles = frozenset(
        hashlib.blake2b(
            "\x1f".join(words[index:index + width]).encode("utf-8"),
            digest_size=16,
        ).digest()
        for index in range(len(words) - width + 1)
    )
    return len(words), shingles


def _source_groups(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group exact sources first, then exact Jaccard comparisons of word shingles."""
    dsu = _DisjointSet(len(sources))
    exact: dict[str, int] = {}
    for index, source in enumerate(sources):
        fingerprint = source["normalized_sha256"]
        if fingerprint in exact:
            dsu.union(index, exact[fingerprint])
        else:
            exact[fingerprint] = index

    representatives = sorted(exact.values())
    features = {index: _word_shingles(sources[index]["_text"]) for index in representatives}
    matches = []
    for offset, left in enumerate(representatives):
        left_length, left_set = features[left]
        for right in representatives[offset + 1:]:
            right_length, right_set = features[right]
            if not left_length or not right_length:
                continue
            length_ratio = min(left_length, right_length) / max(left_length, right_length)
            if length_ratio < NEAR_DUPLICATE_LENGTH_RATIO:
                continue
            # This upper bound avoids unnecessary intersections without missing
            # any pair that can reach the specified Jaccard threshold.
            if min(len(left_set), len(right_set)) / max(len(left_set), len(right_set)) < NEAR_DUPLICATE_THRESHOLD:
                continue
            intersection = len(left_set.intersection(right_set))
            similarity = intersection / (len(left_set) + len(right_set) - intersection)
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                dsu.union(left, right)
                matches.append({
                    "left_doc_id": sources[left]["doc_id"],
                    "right_doc_id": sources[right]["doc_id"],
                    "jaccard": similarity,
                    "word_count_ratio": length_ratio,
                })

    members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, source in enumerate(sources):
        members[dsu.find(index)].append(source)
    groups = []
    for group_sources in members.values():
        group_sources.sort(key=lambda source: source["doc_id"])
        fingerprint = "\n".join(sorted({source["normalized_sha256"] for source in group_sources}))
        groups.append({
            "source_group_id": "source-" + _sha256(fingerprint.encode("utf-8"))[:20],
            "canonical_doc_id": group_sources[0]["doc_id"],
            "alias_doc_ids": [source["doc_id"] for source in group_sources],
            "_sources": group_sources,
        })
    groups.sort(key=lambda group: group["canonical_doc_id"])
    return groups, matches


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    """Return source blocks and their enclosing, one-based source-line spans."""
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, heading.group(2)))
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        start = index
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        block_type = "section"
        if fence:
            block_type = "code"
            marker = fence.group(1)[0]
            minimum = len(fence.group(1))
            index += 1
            while index < len(lines):
                current = lines[index]
                index += 1
                closing = re.match(r"^\s*([`~]+)\s*$", current)
                if closing and set(closing.group(1)) == {marker} and len(closing.group(1)) >= minimum:
                    break
        elif line.count("|") >= 2:
            block_type = "table_rows"
            index += 1
            while index < len(lines) and lines[index].strip() and lines[index].count("|") >= 2:
                index += 1
        else:
            index += 1
            while index < len(lines):
                current = lines[index]
                if (not current.strip() or re.match(r"^\s{0,3}#{1,6}\s+", current)
                        or re.match(r"^\s*(`{3,}|~{3,})", current) or current.count("|") >= 2):
                    break
                index += 1
        body = "\n".join(lines[start:index]).strip()
        if body:
            blocks.append({
                "text": body,
                "heading_path": [title for _, title in headings],
                "block_type": block_type,
                "source_start_line": start + 1,
                "source_end_line": index,
            })
    return blocks


def _token_chunks(block: dict[str, Any], tokenizer: Any, max_tokens: int) -> list[dict[str, Any]]:
    body = block["text"]
    encoded = tokenizer.encode(body, add_special_tokens=False)
    if not encoded.ids:
        return []
    prefix = "# " + " > ".join(block["heading_path"]) + "\n\n" if block["heading_path"] else ""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    if len(prefix_ids) > max_tokens // 3:
        prefix = tokenizer.decode(prefix_ids[:max_tokens // 3], skip_special_tokens=True).strip() + "\n\n"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    budget = max(1, max_tokens - len(prefix_ids) - 2)
    output = []
    start = 0
    while start < len(encoded.ids):
        end = min(start + budget, len(encoded.ids))
        while True:
            char_start = encoded.offsets[start][0]
            char_end = encoded.offsets[end - 1][1]
            body_window = body[char_start:char_end].strip()
            value = prefix + body_window
            tokens = len(tokenizer.encode(value, add_special_tokens=False).ids)
            if tokens <= max_tokens:
                break
            end -= 1
            if end <= start:
                raise ValueError("The tokenizer cannot fit a source token and heading within the token budget.")
        if value.strip():
            output.append({
                **block,
                "text": value,
                "token_count": tokens,
                "body_token_count": len(tokenizer.encode(body_window, add_special_tokens=False).ids),
            })
        if end == len(encoded.ids):
            break
        # Keep progress for small test budgets and exceptionally long headings.
        overlap = min(OVERLAP_TOKENS, max(0, (end - start) // 4))
        start = end - overlap
    return output


def _natural_passage(record: dict[str, Any]) -> bool:
    if record["block_type"] != "section" or record["body_token_count"] < MIN_CANDIDATE_TOKENS:
        return False
    text = record["text"]
    nonspace = [char for char in text if not char.isspace()]
    return bool(nonspace) and sum(char.isalpha() for char in nonspace) / len(nonspace) >= 0.55


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_dataset(
    root: Path,
    study_dir: Path,
    tokenizer_path: Path,
    seed: int = 42,
    max_tokens: int = 254,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """Write a fresh study corpus and pseudo-supervision candidates.

    The corpus includes every retained split for later partition-specific
    indexing. Candidates contain training and validation passages only. They
    are potential inputs to synthetic-query generation, never relevance gold.
    Source line spans enclose the original block; the added heading and token
    window mean that a chunk must not be described as an exact source quote.
    """
    root, study_dir, tokenizer_path = Path(root).resolve(), Path(study_dir).resolve(), Path(tokenizer_path).resolve()
    if not (math.isfinite(train_ratio) and math.isfinite(val_ratio)
            and train_ratio > 0 and val_ratio > 0 and train_ratio + val_ratio < 1):
        raise ValueError("Training and validation ratios must be positive and leave a test partition.")
    if max_tokens < MIN_CANDIDATE_TOKENS:
        raise ValueError("max_tokens must be at least 32 native BGE tokens.")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    protected = [root / name for name in ["data_cleaned", "data_chunked", "data_embedded", "data_extracted", "data_labeled", "models", "chroma_db", "BAAI_bge_m3"]]
    if study_dir == root or any(study_dir == path or path in study_dir.parents for path in protected):
        raise ValueError("Study output must be isolated from existing source, model, embedding and index directories.")
    source_paths = sorted((root / "data_cleaned").glob("*/*.md"))
    if not source_paths:
        raise ValueError("No cleaned Markdown sources were found.")
    sources = []
    for path in source_paths:
        raw = path.read_bytes()
        text = path.read_text(encoding="utf-8-sig")
        sources.append({
            "doc_id": path.parent.name,
            "source_file": str(path.resolve()),
            "source_sha256": _sha256(raw),
            "source_text_sha256": _sha256(text.encode("utf-8")),
            "normalized_sha256": _sha256(_normalize(text).encode("utf-8")),
            "source_bytes": len(raw),
            "_text": text,
        })
    groups, near_matches = _source_groups(sources)
    if len(groups) < 3:
        raise ValueError("At least three independent source groups are required for train/validation/test.")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    training_count = min(max(1, int(len(groups) * train_ratio)), len(groups) - 2)
    validation_count = min(max(1, int(len(groups) * val_ratio)), len(groups) - training_count - 1)
    for index, group in enumerate(shuffled):
        group["split"] = "train" if index < training_count else "validation" if index < training_count + validation_count else "test"

    tokenizer = _load_tokenizer(tokenizer_path)
    records = []
    within_group_duplicate_count = 0
    for group in groups:
        source = group["_sources"][0]
        seen: set[str] = set()
        index = 0
        for block in _markdown_blocks(source["_text"]):
            for window in _token_chunks(block, tokenizer, max_tokens):
                fingerprint = _sha256(_normalize(window["text"]).encode("utf-8"))
                if fingerprint in seen:
                    within_group_duplicate_count += 1
                    continue
                seen.add(fingerprint)
                records.append({
                    **window,
                    "chunk_id": f"{source['doc_id']}-RAG{index:05d}",
                    "doc_id": source["doc_id"],
                    "canonical_doc_id": source["doc_id"],
                    "source_group_id": group["source_group_id"],
                    "alias_doc_ids": group["alias_doc_ids"],
                    "split": group["split"],
                    "hash": _sha256(window["text"].encode("utf-8")),
                    "normalized_text_sha256": fingerprint,
                    "source_file": source["source_file"],
                    "source_sha256": source["source_sha256"],
                    "source_text_sha256": source["source_text_sha256"],
                    "source_span_kind": "enclosing_block_lines_heading_prefixed_token_window",
                    "chunking_version": "native-bge-source-groups-v1",
                })
                index += 1

    hash_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        hash_groups[record["normalized_text_sha256"]].add(record["source_group_id"])
    shared = {fingerprint for fingerprint, group_ids in hash_groups.items() if len(group_ids) > 1}
    shared_records = [record for record in records if record["normalized_text_sha256"] in shared]
    corpus = [record for record in records if record["normalized_text_sha256"] not in shared]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in corpus:
        if _natural_passage(record):
            by_group[record["source_group_id"]].append(record)
    candidates = []
    for group in groups:
        split = group["split"]
        if split == "test":
            continue
        eligible = list(by_group[group["source_group_id"]])
        local_seed = int(_sha256(f"{seed}:{group['source_group_id']}".encode("utf-8"))[:16], 16)
        random.Random(local_seed).shuffle(eligible)
        for record in sorted(eligible[:8 if split == "train" else 2], key=lambda row: row["chunk_id"]):
            candidates.append({
                **record,
                "candidate_id": "cand-" + record["chunk_id"],
                "supervision": "synthetic_query_generation_candidate_not_human_relevance",
            })
    counts = {
        "source_documents": len(sources),
        "normalized_unique_sources": len({source["normalized_sha256"] for source in sources}),
        "source_groups": len(groups),
        "canonical_sources": len(groups),
        "alias_sources_omitted": len(sources) - len(groups),
        "near_duplicate_pairs": len(near_matches),
        "chunks_before_shared_text_exclusion": len(records),
        "within_group_duplicate_chunks_omitted": within_group_duplicate_count,
        "shared_text_hashes_excluded": len(shared),
        "shared_text_chunks_excluded": len(shared_records),
        "corpus_chunks": len(corpus),
        "candidate_passages": len(candidates),
    }
    split_groups = {split: sorted(group["source_group_id"] for group in groups if group["split"] == split) for split in ["train", "validation", "test"]}
    split_details = {}
    for split, group_ids in split_groups.items():
        selected = [group for group in groups if group["split"] == split]
        split_details[split] = {
            "source_group_ids": group_ids,
            "document_ids": sorted(doc for group in selected for doc in group["alias_doc_ids"]),
            "canonical_document_ids": sorted(group["canonical_doc_id"] for group in selected),
            "groups": len(group_ids),
            "chunks": sum(row["split"] == split for row in corpus),
            "candidate_passages": sum(row["split"] == split for row in candidates),
            "groups_without_eligible_passages": sorted(group["source_group_id"] for group in selected if not by_group[group["source_group_id"]]),
        }
    grouping = {
        "exact": "sha256-utf8-unicode-nfc-casefold-collapse-whitespace",
        "near_duplicate": "single-link-components-of-word-shingle-jaccard",
        "shingle_words": SHINGLE_WORDS,
        "shingle_hash": "blake2b-128",
        "minimum_jaccard": NEAR_DUPLICATE_THRESHOLD,
        "minimum_word_count_ratio": NEAR_DUPLICATE_LENGTH_RATIO,
        "canonical_selection": "lexicographically_first_document_id",
        "shared_chunk_policy": "exclude_normalized_exact_chunk_text_shared_across_source_groups_from_every_partition",
    }
    public_groups = [{k: v for k, v in group.items() if not k.startswith("_")} for group in groups]
    source_manifest = {
        "schema_version": 1,
        "sources": [{k: v for k, v in source.items() if not k.startswith("_")} for source in sources],
        "groups": public_groups,
        "grouping": grouping,
        "near_duplicate_pairs": near_matches,
        "source_sha256_semantics": "sha256_of_original_file_bytes",
        "source_text_sha256_semantics": "sha256_of_utf8_BOM_decoded_universal_newline_text",
        "language_labels_available": False,
    }
    split_manifest = {
        "schema_version": 1,
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": _sha256(tokenizer_path.read_bytes()),
        "max_tokens_without_special_tokens": max_tokens,
        "token_window_overlap": "min(32, quarter_of_current_body_window)",
        "minimum_candidate_tokens": MIN_CANDIDATE_TOKENS,
        "candidate_limit_per_training_group": 8,
        "candidate_limit_per_validation_group": 2,
        "supervision": "synthetic_candidates_only_no_human_relevance_labels",
        "test_labels_read": False,
        "grouping": grouping,
        "split_groups": split_groups,
        "splits": split_details,
        "counts": counts,
        "shared_text_exclusions_by_split": dict(Counter(row["split"] for row in shared_records)),
        "shared_text_hashes": sorted(shared),
        "limitations": [
            "Synthetic queries and teacher filtering are weak supervision, not human relevance gold.",
            "Near-duplicate grouping is a high-threshold word-shingle heuristic; it cannot certify semantic independence.",
            "Source line spans enclose the source block; generated heading prefixes are not verbatim source quotes.",
            "The complete corpus includes test-source passages; evaluation and negative mining must filter by split.",
        ],
    }
    study_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: study_dir / name for name in ["corpus.jsonl", "candidate_passages.jsonl", "source_manifest.json", "split_manifest.json"]}
    _write_jsonl(paths["corpus.jsonl"], corpus)
    _write_jsonl(paths["candidate_passages.jsonl"], candidates)
    _write_json(paths["source_manifest.json"], source_manifest)
    split_manifest["artifact_sha256"] = {name: _sha256(path.read_bytes()) for name, path in paths.items() if name != "split_manifest.json"}
    _write_json(paths["split_manifest.json"], split_manifest)
    return {
        "status": "prepared",
        "study_dir": str(study_dir),
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": counts,
        "split_groups": split_groups,
        "splits": split_details,
    }
