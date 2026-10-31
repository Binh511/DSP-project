"""Resumable GenQ weak supervision, with sequential GPU model lifetimes.

Importing this module is lightweight and never starts model inference. Query
generation and teacher filtering are explicit steps in ``generate_pairs``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import unicodedata
from typing import Any


VERSION = "psai-genq-teacher-filter-v1"
SUPERVISION = "synthetic_genq_teacher_filtered"
DEFAULTS = {
    "seed": 42, "generation_batch_size": 4, "input_max_length": 256,
    "max_new_tokens": 40, "min_new_tokens": 4, "queries_per_passage": 2,
    "top_p": 0.95, "top_k": 50, "no_repeat_ngram_size": 3,
    "teacher_batch_size": 16, "teacher_max_length": 384,
    "teacher_margin": 0.5, "negative_candidates": 2,
    "max_train_pairs": 1536, "max_validation_pairs": 64,
}


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _status(study_dir: Path, phase: str, **details: Any) -> None:
    _atomic_json(study_dir / "generation_status.json", {
        "phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), **details,
    })


def _settings(config: dict) -> dict:
    values = dict(DEFAULTS)
    for source in [config, config.get("query_generation", {}), config.get("teacher_filter", {})]:
        if not isinstance(source, dict):
            raise ValueError("Generation and teacher settings must be objects")
        values.update({key: value for key, value in source.items() if key in DEFAULTS})
    for key in DEFAULTS:
        if key in {"top_p", "teacher_margin"}:
            if isinstance(values[key], bool) or not isinstance(values[key], (int, float)) or not math.isfinite(values[key]):
                raise ValueError(f"Invalid setting {key}")
        elif type(values[key]) is not int or values[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid positive integer setting {key}")
    if not 0 < values["top_p"] <= 1 or values["teacher_margin"] < 0.5:
        raise ValueError("top_p must be in (0,1], and the predeclared teacher margin cannot be below 0.5")
    if values["queries_per_passage"] != 2 or values["negative_candidates"] != 2:
        raise ValueError("This experiment requires two queries and two mined negative candidates per passage")
    if values["min_new_tokens"] > values["max_new_tokens"]:
        raise ValueError("min_new_tokens exceeds max_new_tokens")
    if values["max_train_pairs"] < 64 or values["max_validation_pairs"] < 16:
        raise ValueError("Pair caps must permit at least 64 training and 16 validation pairs")
    return values


def _model_bindings(auxiliary_models: dict) -> dict[str, dict]:
    entries = auxiliary_models.get("models", [])
    models = {}
    for role, expected in [("query_generator", "doc2query/msmarco-t5-small-v1"),
                           ("pseudo_label_teacher", "cross-encoder/ms-marco-MiniLM-L6-v2")]:
        matches = [entry for entry in entries if entry.get("role") == role]
        if len(matches) != 1:
            raise ValueError(f"Exactly one pinned {role} model is required")
        entry = matches[0]
        if entry.get("repo_id") != expected or not re.fullmatch(r"[0-9a-fA-F]{40}", str(entry.get("revision", ""))):
            raise ValueError(f"Unexpected or unpinned {role} model")
        path = Path(entry["local_path"]).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        files = {item.name: _file_hash(item) for item in sorted(path.iterdir())
                 if item.is_file() and item.suffix in {".json", ".bin", ".safetensors", ".model", ".txt"}}
        if "config.json" not in files or not any(name.endswith((".bin", ".safetensors")) for name in files):
            raise ValueError(f"Incomplete local auxiliary model: {path}")
        models[role] = {"repo_id": expected, "revision": entry["revision"],
                        "local_path": str(path), "files_sha256": files}
    return models


def _load_inputs(study_dir: Path) -> tuple[list[dict], list[dict], dict]:
    corpus = _read_jsonl(study_dir / "corpus.jsonl")
    candidates = _read_jsonl(study_dir / "candidate_passages.jsonl")
    manifest = json.loads((study_dir / "split_manifest.json").read_text("utf-8-sig"))
    by_id = {row["chunk_id"]: row for row in corpus}
    if len(by_id) != len(corpus):
        raise ValueError("Duplicate corpus chunk IDs")
    split_groups = manifest["split_groups"]
    sets = [set(split_groups[split]) for split in ["train", "validation", "test"]]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Source groups overlap corpus partitions")
    seen = set()
    for row in candidates:
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("Query generation must never use a test-source passage")
        reference = by_id.get(row.get("chunk_id"))
        if reference is None or any(row.get(key) != reference.get(key) for key in
                                    ["text", "doc_id", "source_group_id", "split", "source_sha256"]):
            raise ValueError("Candidate does not exactly match its source corpus record")
        if row["source_group_id"] not in split_groups[row["split"]]:
            raise ValueError("Candidate source group does not match the split manifest")
        if not isinstance(row.get("candidate_id"), str) or row["candidate_id"] in seen:
            raise ValueError("Candidate IDs must be present and unique")
        seen.add(row["candidate_id"])
    if not all(any(row["split"] == split for row in candidates) for split in ["train", "validation"]):
        raise ValueError("Both train and validation need query-generation candidates")
    return corpus, sorted(candidates, key=lambda row: row["candidate_id"]), manifest


def _journal(path: Path, fingerprint: str, valid_ids: set[str], id_key: str) -> dict[str, dict]:
    """Recover an interrupted final write; reject every other journal inconsistency."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    recovered = bool(raw and not raw.endswith(b"\n"))
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                recovered = True
                break
            raise ValueError(f"Corrupt journal record in {path}")
        if row.get("fingerprint") != fingerprint or row.get(id_key) not in valid_ids:
            raise ValueError(f"Journal fingerprint or source identity mismatch: {path}")
        rows.append(row)
    result = {row[id_key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate journal identity in {path}")
    if recovered:
        _atomic_jsonl(path, rows)
    return result


class _QueryGenerator:
    def __init__(self, entry: dict, settings: dict):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("GenQ inference requires the authorized CUDA device")
        self.torch, self.settings = torch, settings
        self.tokenizer = AutoTokenizer.from_pretrained(entry["local_path"], local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(entry["local_path"], local_files_only=True,
                                                         torch_dtype=torch.float32).to("cuda").eval()

    def generate(self, passages: list[str], seed: int) -> list[list[str]]:
        torch, cfg = self.torch, self.settings
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = self.tokenizer(passages, padding=True, truncation=True,
                                max_length=cfg["input_max_length"], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=True, top_p=cfg["top_p"], top_k=cfg["top_k"],
                                         num_return_sequences=2, max_new_tokens=cfg["max_new_tokens"],
                                         min_new_tokens=cfg["min_new_tokens"],
                                         no_repeat_ngram_size=cfg["no_repeat_ngram_size"])
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [decoded[index:index + 2] for index in range(0, len(decoded), 2)]

    def close(self) -> None:
        del self.model, self.tokenizer
        gc.collect()
        self.torch.cuda.empty_cache()


class _Teacher:
    def __init__(self, entry: dict, settings: dict):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Teacher filtering requires the authorized CUDA device")
        self.torch, self.settings = torch, settings
        self.tokenizer = AutoTokenizer.from_pretrained(entry["local_path"], local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            entry["local_path"], local_files_only=True, torch_dtype=torch.float32).to("cuda").eval()
        if self.model.config.num_labels != 1:
            raise ValueError("Expected a single-logit MS MARCO cross encoder")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        inputs = self.tokenizer([query for query, _ in pairs], [passage for _, passage in pairs],
                                padding=True, truncation=True, max_length=self.settings["teacher_max_length"],
                                return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            values = self.model(**inputs).logits.reshape(-1).float().cpu().tolist()
        return [float(value) for value in values]

    def close(self) -> None:
        del self.model, self.tokenizer
        gc.collect()
        self.torch.cuda.empty_cache()


def _round_robin_cap(rows: list[dict], limit: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["query_id"])
    result = []
    for offset in range(max((len(group) for group in groups.values()), default=0)):
        for key in sorted(groups):
            if offset < len(groups[key]):
                result.append(groups[key][offset])
                if len(result) == limit:
                    return result
    return result


def _deduplicate_queries(generated: list[dict], candidates: dict[str, dict]) -> tuple[list[dict], Counter]:
    by_query = defaultdict(list)
    counts = Counter()
    for entry in generated:
        candidate = candidates[entry["candidate_id"]]
        queries = entry.get("queries")
        if not isinstance(queries, list) or len(queries) != 2 or any(not isinstance(query, str) for query in queries):
            raise ValueError("Every generated candidate needs exactly two query strings")
        for index, raw in enumerate(queries):
            query = " ".join(raw.split()).strip()
            counts["generated_query_strings"] += 1
            query_words = re.findall(r"\w+", _norm(query))
            passage_words = re.findall(r"\w+", _norm(candidate["text"]))
            if not 4 <= len(query_words) <= 40:
                counts["queries_rejected_word_count_4_to_40"] += 1
                continue
            copied_span = (query_words and len(query_words) >= 0.8 * len(passage_words)
                           and (" " + " ".join(query_words) + " ") in (" " + " ".join(passage_words) + " "))
            if _norm(query) == _norm(candidate["text"]) or copied_span:
                counts["queries_rejected_verbatim_or_long_source_span"] += 1
                continue
            by_query[_norm(query)].append({"query": query, "candidate": candidate, "generation_index": index})
    unique = []
    for normalized, rows in sorted(by_query.items()):
        if len({row["candidate"]["split"] for row in rows}) > 1:
            counts["cross_partition_duplicate_queries_removed"] += len(rows)
            continue
        if len({row["candidate"]["doc_id"] for row in rows}) > 1:
            counts["cross_document_duplicate_queries_removed"] += len(rows)
            continue
        rows.sort(key=lambda row: (row["candidate"]["candidate_id"], row["generation_index"]))
        selected = rows[0]
        selected["query_id"] = "genq-" + _digest(normalized)[:24]
        selected["known_positive_ids"] = sorted({row["candidate"]["chunk_id"] for row in rows})
        counts["within_partition_duplicate_queries_removed"] += len(rows) - 1
        unique.append(selected)
    return unique, counts


def _language_flags(text: str, provided: Any = None) -> list[str]:
    flags = []
    if provided and str(provided).lower() not in {"en", "eng", "english"}:
        flags.append("provided_non_english_language")
    if re.search(r"[\u0103\u0111\u01a1\u01b0\u1ea0-\u1ef9]", text.casefold()):
        flags.append("vietnamese_diacritic_characters_heuristic")
    if re.search(r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]", text):
        flags.append("non_latin_script_heuristic")
    return flags


def generate_pairs(root: Path, study_dir: Path, auxiliary_models: dict, config: dict) -> dict:
    """Generate and filter isolated synthetic pairs; never interpret them as gold."""
    root, study_dir = Path(root).resolve(), Path(study_dir).resolve()
    if root == study_dir or root not in study_dir.parents:
        raise ValueError("Generation output must be an isolated study directory under the project")
    cfg = _settings(config)
    corpus, candidates, split_manifest = _load_inputs(study_dir)
    models = _model_bindings(auxiliary_models)
    source_files = ["candidate_passages.jsonl", "corpus.jsonl", "split_manifest.json", "source_manifest.json"]
    binding = {"version": VERSION, "implementation_sha256": _file_hash(Path(__file__)),
               "sources_sha256": {name: _file_hash(study_dir / name) for name in source_files},
               "models": models, "settings": cfg,
               "negative_policy": "same_document_and_source_group_only_drop_if_none",
               "deduplication": "remove_all_normalized_query_duplicates_across_partitions_or_documents",
               "query_quality": {"minimum_words": 4, "maximum_words": 40,
                                 "reject_verbatim_passage": True, "reject_copied_source_span_minimum_passage_fraction": 0.8}}
    fingerprint = _digest(binding)
    manifest_path = study_dir / "query_generation_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text("utf-8-sig"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("Source, model, implementation or configuration changed; use a new study directory")
    else:
        if (study_dir / "generated_queries.jsonl").exists() or (study_dir / "teacher_scores.jsonl").exists():
            raise ValueError("An existing journal has no trustworthy generation manifest")
        _atomic_json(manifest_path, {"fingerprint": fingerprint, "binding": binding,
                                     "supervision": SUPERVISION, "human_relevance_labels": False})
    journal_path = study_dir / "generated_queries.jsonl"
    candidate_ids = {row["candidate_id"] for row in candidates}
    generated = _journal(journal_path, fingerprint, candidate_ids, "candidate_id")
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    for identity, entry in generated.items():
        if any(entry.get(key) != candidate_by_id[identity].get(key)
               for key in ["chunk_id", "doc_id", "source_group_id", "split"]):
            raise ValueError("Query journal source metadata differs from its bound candidate")
    _status(study_dir, "query_generation", completed_candidates=len(generated), total_candidates=len(candidates), fingerprint=fingerprint)
    generator = None
    try:
        with journal_path.open("a", encoding="utf-8", newline="\n") as stream:
            size = cfg["generation_batch_size"]
            for batch_index, start in enumerate(range(0, len(candidates), size)):
                batch = candidates[start:start + size]
                if all(row["candidate_id"] in generated for row in batch):
                    continue
                if generator is None:
                    print("Loading pinned T5-small query generator on CUDA (float32)", flush=True)
                    generator = _QueryGenerator(models["query_generator"], cfg)
                # Regenerate the complete fixed batch after a partial journal
                # append so an interrupted run uses identical sampling inputs.
                batch_seed = int(_digest([cfg["seed"], [row["candidate_id"] for row in batch]])[:8], 16)
                outputs = generator.generate([row["text"] for row in batch], batch_seed)
                if len(outputs) != len(batch) or any(len(queries) != 2 for queries in outputs):
                    raise ValueError("Query generator returned an unexpected batch shape")
                for row, queries in zip(batch, outputs):
                    identity = row["candidate_id"]
                    if identity in generated:
                        if generated[identity]["queries"] != queries:
                            raise ValueError("Partial-batch resume did not reproduce existing sampled queries")
                        continue
                    entry = {"candidate_id": identity, "chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
                             "source_group_id": row["source_group_id"], "split": row["split"],
                             "queries": queries, "generation_seed": batch_seed,
                             "generator_repo_id": models["query_generator"]["repo_id"],
                             "generator_revision": models["query_generator"]["revision"], "fingerprint": fingerprint}
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    generated[identity] = entry
                stream.flush()
                os.fsync(stream.fileno())
                if (batch_index + 1) % 10 == 0 or len(generated) == len(candidates):
                    print(f"GenQ candidates {len(generated)}/{len(candidates)}", flush=True)
                    _status(study_dir, "query_generation", completed_candidates=len(generated), total_candidates=len(candidates), fingerprint=fingerprint)
    finally:
        if generator is not None:
            generator.close()
    unique, counts = _deduplicate_queries(list(generated.values()), {row["candidate_id"]: row for row in candidates})
    counts["completed_candidates"] = len(generated)
    counts["unique_queries_before_negative_mining"] = len(unique)

    from psai_rag.evaluate import BM25
    indices = {split: BM25([row for row in corpus if row["split"] == split]) for split in ["train", "validation"]}
    doc_chunks = defaultdict(list)
    for row in corpus:
        if row["split"] in indices:
            doc_chunks[row["split"], row["doc_id"], row["source_group_id"]].append(row["chunk_id"])
    mined = []
    for query in unique:
        positive = query["candidate"]
        negative_ids = doc_chunks[positive["split"], positive["doc_id"], positive["source_group_id"]]
        negatives = indices[positive["split"]].negative_candidates(
            query["query"], positive["chunk_id"], query["known_positive_ids"],
            top_k=2, candidate_ids=negative_ids)
        if not negatives:
            counts["queries_without_same_document_negatives"] += 1
            continue
        mined.append({**query, "negatives": negatives})
    counts["queries_with_mined_negatives"] = len(mined)
    teacher_journal_path = study_dir / "teacher_scores.jsonl"
    scored = _journal(teacher_journal_path, fingerprint, {row["query_id"] for row in mined}, "query_id")
    teacher = None
    try:
        with teacher_journal_path.open("a", encoding="utf-8", newline="\n") as stream:
            pending = [row for row in mined if row["query_id"] not in scored]
            # Pack at most floor(batch/3) complete queries, retaining each
            # positive and its one/two negative scores in a durable record.
            query_batch = max(1, cfg["teacher_batch_size"] // 3)
            _status(study_dir, "teacher_filtering", completed_queries=len(scored), total_queries=len(mined), fingerprint=fingerprint)
            for batch_index, start in enumerate(range(0, len(pending), query_batch)):
                batch = pending[start:start + query_batch]
                if teacher is None:
                    print("Loading pinned MiniLM cross-encoder teacher on CUDA (float32)", flush=True)
                    teacher = _Teacher(models["pseudo_label_teacher"], cfg)
                pairs = [(row["query"], passage["text"]) for row in batch
                         for passage in [row["candidate"], *row["negatives"]]]
                scores = teacher.score(pairs)
                if len(scores) != len(pairs) or any(not math.isfinite(score) for score in scores):
                    raise ValueError("Teacher returned invalid relevance logits")
                offset = 0
                for row in batch:
                    count = 1 + len(row["negatives"])
                    entry = {"query_id": row["query_id"], "positive_id": row["candidate"]["chunk_id"],
                             "positive_score": scores[offset],
                             "negative_scores": [{"chunk_id": negative["chunk_id"], "score": scores[offset + index + 1]}
                                                 for index, negative in enumerate(row["negatives"])],
                             "fingerprint": fingerprint}
                    offset += count
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    scored[row["query_id"]] = entry
                stream.flush()
                os.fsync(stream.fileno())
                if (batch_index + 1) % 10 == 0 or len(scored) == len(mined):
                    print(f"Teacher queries {len(scored)}/{len(mined)}", flush=True)
                    _status(study_dir, "teacher_filtering", completed_queries=len(scored), total_queries=len(mined), fingerprint=fingerprint)
    finally:
        if teacher is not None:
            teacher.close()

    accepted = []
    for row in mined:
        result = scored[row["query_id"]]
        positive = row["candidate"]
        if result["positive_id"] != positive["chunk_id"] or {value["chunk_id"] for value in result["negative_scores"]} != {value["chunk_id"] for value in row["negatives"]}:
            raise ValueError("Teacher journal positive/negative identities differ from mined records")
        eligible = [value for value in result["negative_scores"] if result["positive_score"] - value["score"] >= cfg["teacher_margin"]]
        counts["negative_candidates_rejected_teacher_margin"] += len(result["negative_scores"]) - len(eligible)
        if not eligible:
            counts["queries_rejected_teacher_margin"] += 1
            continue
        chosen = max(eligible, key=lambda value: (value["score"], value["chunk_id"]))
        negative = next(value for value in row["negatives"] if value["chunk_id"] == chosen["chunk_id"])
        flags = _language_flags(positive["text"], positive.get("language"))
        accepted.append({
            "query_id": row["query_id"], "query": row["query"],
            "query_word_count": len(re.findall(r"\w+", row["query"])),
            "positive": positive["text"], "negative": negative["text"],
            "positive_id": positive["chunk_id"], "negative_id": negative["chunk_id"],
            "doc_id": positive["doc_id"], "source_group_id": positive["source_group_id"],
            "split": positive["split"], "supervision": SUPERVISION,
            "known_positive_ids": row["known_positive_ids"],
            "source_sha256": positive.get("source_sha256"), "source_file": positive.get("source_file"),
            "generator_repo_id": models["query_generator"]["repo_id"], "generator_revision": models["query_generator"]["revision"],
            "teacher_repo_id": models["pseudo_label_teacher"]["repo_id"], "teacher_revision": models["pseudo_label_teacher"]["revision"],
            "teacher_positive_score": result["positive_score"], "teacher_negative_score": chosen["score"],
            "teacher_margin": result["positive_score"] - chosen["score"],
            "minimum_teacher_margin": cfg["teacher_margin"], "negative_bm25_score": negative["bm25_score"],
            "language_risk_flags": flags, "human_relevance_verified": False,
            "generation_fingerprint": fingerprint,
        })
    counts["teacher_accepted_pairs_before_caps"] = len(accepted)
    selected = {split: _round_robin_cap([row for row in accepted if row["split"] == split], cfg[limit])
                for split, limit in [("train", "max_train_pairs"), ("validation", "max_validation_pairs")]}
    coverage = {}
    for split in ["train", "validation"]:
        rows = selected[split]
        all_groups = set(split_manifest["split_groups"][split])
        candidate_groups = {row["source_group_id"] for row in candidates if row["split"] == split}
        passed_groups = {row["source_group_id"] for row in accepted if row["split"] == split}
        selected_groups = {row["source_group_id"] for row in rows}
        coverage[split] = {"partition_source_groups": len(all_groups), "candidate_source_groups": len(candidate_groups),
                           "teacher_accepted_source_groups": len(passed_groups), "selected_source_groups": len(selected_groups),
                           "selected_pair_count": len(rows), "selected_document_count": len({row["doc_id"] for row in rows}),
                           "groups_without_candidates": sorted(all_groups - candidate_groups),
                           "candidate_groups_without_accepted_pairs": sorted(candidate_groups - passed_groups),
                           "groups_omitted_by_cap": sorted(passed_groups - selected_groups),
                           "pairs_per_source_group": dict(Counter(row["source_group_id"] for row in rows)),
                           "language_risk_flagged_pairs": sum(bool(row["language_risk_flags"]) for row in rows)}
    paths = {"train_pairs.jsonl": study_dir / "train_pairs.jsonl", "validation_pairs.jsonl": study_dir / "validation_pairs.jsonl",
             "generation_report.json": study_dir / "generation_report.json"}
    enough = len(selected["train"]) >= 64 and len(selected["validation"]) >= 16
    report = {"status": "ready" if enough else "insufficient_filtered_supervision", "version": VERSION,
              "fingerprint": fingerprint, "settings": cfg, "counts": dict(counts), "coverage": coverage,
              "models": models, "human_relevance_ground_truth": False, "test_sources_used": False,
              "language_coverage": {"models": "English-oriented MS MARCO query generator and teacher",
                                    "language_labels_available": False, "non_english_detection": "script/diacritic heuristic only; not language certification",
                                    "vietnamese_quality_verified": False},
              "limitations": ["All pairs are synthetic weak supervision, not human relevance judgments.",
                              "A positive-negative logit margin can pass even when both passages are irrelevant; this heuristic does not establish absolute relevance or calibrated human relevance.",
                              "English-oriented auxiliary models do not establish Vietnamese retrieval quality.",
                              "Validation synthetic labels must not be presented as independent human evaluation."]}
    _atomic_json(paths["generation_report.json"], report)
    if not enough:
        _status(study_dir, "insufficient_filtered_supervision", coverage=coverage, fingerprint=fingerprint)
        raise ValueError(f"Need >=64 training and >=16 validation pairs; retained {len(selected['train'])}/{len(selected['validation'])}; see generation_report.json")
    _atomic_jsonl(paths["train_pairs.jsonl"], selected["train"])
    _atomic_jsonl(paths["validation_pairs.jsonl"], selected["validation"])
    report["artifact_sha256"] = {name: _file_hash(path) for name, path in paths.items() if name.endswith(".jsonl")}
    _atomic_json(paths["generation_report.json"], report)
    _status(study_dir, "complete", coverage=coverage, fingerprint=fingerprint)
    return {"status": "ready", "paths": {name: str(path) for name, path in paths.items()},
            "counts": dict(counts), "coverage": coverage, "fingerprint": fingerprint}
