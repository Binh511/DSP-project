"""Blinded pooled validation judgments and strict, non-inferential CSV import."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from .analysis import digest_file, read_jsonl


IDENTITY_FIELDS = ["item_id", "query_id", "doc_id", "source_group_id", "query", "chunk_id",
                   "passage_doc_id", "passage_source_group_id", "text", "source_file",
                   "source_start_line", "source_end_line"]
LABEL_FIELDS = ["query_realism", "source_quality", "relevance", "reviewer", "notes"]
FORM_FIELDS = IDENTITY_FIELDS + LABEL_FIELDS
CHOICES = {"query_realism": {"realistic", "malformed", "uncertain"},
           "source_quality": {"readable", "corrupt", "uncertain"}, "relevance": {"0", "1", "2", "unknown"}}


def _safe_cell(value) -> str:
    value = str(value)
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_cell(row.get(key, "")) for key in FORM_FIELDS})


def make_review(study_dir: Path, output_dir: Path, *, depth: int = 5, seed: int = 42,
                priority: bool = False) -> dict:
    if not 1 <= depth <= 10:
        raise ValueError("Pool depth must be between one and ten")
    study_dir, output_dir = Path(study_dir), Path(output_dir)
    corpus = {row["chunk_id"]: row for row in read_jsonl(study_dir / "corpus.jsonl") if row["split"] == "validation"}
    queries = {row["query_id"]: row for row in read_jsonl(study_dir / "validation_pairs.jsonl")}
    if any(row.get("split") != "validation" for row in queries.values()):
        raise ValueError("Only validation queries are permitted in a review packet")
    total_queries = len(queries)
    if priority:
        depth = 3
        base = {row["query_id"]: row for row in read_jsonl(study_dir / "base_predictions.jsonl")
                if row["scope"] == "within_doc" and row["method"] == "dense"}
        adapter = {row["query_id"]: row for row in read_jsonl(study_dir / "adapter_predictions.jsonl")
                   if row["scope"] == "within_doc" and row["method"] == "dense"}
        changed = {identity for identity in queries if abs(base[identity]["metrics"]["MRR@10"] - adapter[identity]["metrics"]["MRR@10"]) > 1e-12}
        risk = {identity for identity, row in queries.items() if row["teacher_positive_score"] < 0}
        queries = {identity: row for identity, row in queries.items() if identity in changed | risk}
    pool = defaultdict(set)
    coordinator = defaultdict(list)
    for variant in ["base", "adapter"]:
        for prediction in read_jsonl(study_dir / f"{variant}_predictions.jsonl"):
            identity = prediction["query_id"]
            if identity not in queries:
                if priority:
                    continue
                raise ValueError("A prediction refers to an unknown validation query")
            if priority and (prediction["scope"] != "within_doc" or prediction["method"] != "dense"):
                continue
            for hit in prediction["ranking"][:depth]:
                pool[identity].add(hit["chunk_id"])
                coordinator[identity, hit["chunk_id"]].append({"variant": variant, "scope": prediction["scope"],
                                                              "method": prediction["method"], "rank": hit["rank"]})
    for identity, query in queries.items():
        pool[identity].update(query.get("known_positive_ids", [query["positive_id"]]))
        pool[identity].add(query["positive_id"])
        if query.get("negative_id") and not priority:
            pool[identity].add(query["negative_id"])
    items = []
    for identity in sorted(queries):
        query = queries[identity]
        for chunk_id in sorted(pool[identity]):
            if chunk_id not in corpus:
                raise ValueError("Pooled passage is not in the validation corpus")
            passage = corpus[chunk_id]
            item_id = hashlib.sha256((identity + "\n" + chunk_id).encode()).hexdigest()[:24]
            items.append({
                "item_id": item_id, "query_id": identity, "doc_id": query["doc_id"],
                "source_group_id": query["source_group_id"], "query": query["query"], "chunk_id": chunk_id,
                "passage_doc_id": passage["doc_id"], "passage_source_group_id": passage["source_group_id"],
                "text": passage["text"], "source_file": passage.get("source_file", ""),
                "source_start_line": str(passage.get("source_start_line", "")),
                "source_end_line": str(passage.get("source_end_line", "")),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    # Rerunning packet construction must not erase any entered human labels.
    for name in ["reviewer_A.csv", "reviewer_B.csv", "items.jsonl", "packet_manifest.json"]:
        if (output_dir / name).exists():
            raise FileExistsError("Use a new packet directory; existing review artifacts are preserved")
    with (output_dir / "items.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    for index, name in enumerate(["reviewer_A.csv", "reviewer_B.csv"]):
        shuffled = list(items)
        random.Random(seed + index).shuffle(shuffled)
        _write_csv(output_dir / name, shuffled)
    key = [{"item_id": item["item_id"], "provenance": coordinator[item["query_id"], item["chunk_id"]],
            "synthetic_positive": item["chunk_id"] in set(queries[item["query_id"]].get("known_positive_ids", []))
                or item["chunk_id"] == queries[item["query_id"]]["positive_id"]}
           for item in items]
    (output_dir / "coordinator_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "psai-validation-pooled-human-review-v1", "split": "validation",
                "query_origin": "synthetic_genq_not_real_user_queries", "queries": len(queries), "items": len(items),
                "total_validation_queries": total_queries,
                "review_scope": "prioritized_diagnostic_subset" if priority else "full_validation_pooled",
                "selection_policy": "union of changed within_doc dense MRR queries and negative-positive-teacher-logit queries; outcome-informed diagnostic sample" if priority else "all validation queries",
                "candidate_pool_policy": "within_doc dense base/adapter top3 plus synthetic positives" if priority else "all methods/scopes/variants top-k plus synthetic positives/negatives",
                "pool_depth": depth, "seed": seed, "human_labels_prefilled": False,
                "items_sha256": digest_file(output_dir / "items.jsonl"),
                "sources_sha256": {name: digest_file(study_dir / name) for name in
                    ["corpus.jsonl", "validation_pairs.jsonl", "base_predictions.jsonl", "adapter_predictions.jsonl"]},
                "blinding": "Forms hide model, rank, score and pseudo-positive status; keep coordinator key from raters.",
                "coverage_limit": "Pool combines top-k from all recorded systems and pseudo-positive/negative passages; unpooled relevance remains unknown.",
                "allowed_metrics_after_review": [f"Precision@{depth}", f"nDCG@{depth}_pooled_ideal"],
                "human_mrr_at_10_ready": depth == 10, "final_test_scored": False}
    (output_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_form(path: Path, items: dict[str, dict]) -> dict[str, dict]:
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(FORM_FIELDS) or len(reader.fieldnames or []) != len(FORM_FIELDS):
            raise ValueError("Review CSV columns differ from the declared form schema")
        for row in reader:
            if None in row or row.get("item_id") not in items or row["item_id"] in result:
                raise ValueError("Unknown/repeated item or malformed CSV row")
            expected = items[row["item_id"]]
            if any(row[key] != _safe_cell(expected[key]) for key in IDENTITY_FIELDS):
                raise ValueError("A query, passage, source reference or identity was edited")
            for key, choices in CHOICES.items():
                if row[key] and row[key] not in choices:
                    raise ValueError(f"Unsupported human label for {key}; blank remains missing")
            if any(row[key] for key in CHOICES) and not row["reviewer"].strip():
                raise ValueError("A human reviewer identity is required for entered labels")
            result[row["item_id"]] = row
    if set(result) != set(items):
        raise ValueError("Review CSV is missing packet items")
    return result


def import_reviews(packet_dir: Path, reviewer_a: Path, reviewer_b: Path, output_dir: Path,
                   *, adjudication_path: Path | None = None) -> dict:
    """Validate two forms and export only fully reviewed validation input.

    Missing labels are reported, never filled by a model or inferred from the
    original synthetic positive. No retrieval evaluation is performed here.
    """
    packet_dir, output_dir = Path(packet_dir), Path(output_dir)
    manifest = json.loads((packet_dir / "packet_manifest.json").read_text("utf-8-sig"))
    if manifest.get("split") != "validation" or digest_file(packet_dir / "items.jsonl") != manifest["items_sha256"]:
        raise ValueError("Packet provenance or validation-only scope is invalid")
    items = {row["item_id"]: row for row in read_jsonl(packet_dir / "items.jsonl")}
    a, b = _read_form(reviewer_a, items), _read_form(reviewer_b, items)
    complete = lambda row: all(row[key] for key in CHOICES) and bool(row["reviewer"].strip())
    missing = sum(not complete(row) for row in [*a.values(), *b.values()])
    needs, resolved = {}, {}
    for identity, item in items.items():
        left, right = a[identity], b[identity]
        if complete(left) and complete(right) and left["reviewer"].strip().casefold() == right["reviewer"].strip().casefold():
            raise ValueError("Independent rater identities must differ")
        agreement = complete(left) and complete(right) and all(left[key] == right[key] for key in CHOICES)
        certain = left["query_realism"] != "uncertain" and left["source_quality"] != "uncertain" and left["relevance"] != "unknown"
        if agreement and certain:
            resolved[identity] = {**item, **{key: left[key] for key in CHOICES},
                                  "reviewers": [left["reviewer"], right["reviewer"]]}
        else:
            needs[identity] = {**item, "notes": "A: " + str({key: left[key] for key in CHOICES})
                                                + "; B: " + str({key: right[key] for key in CHOICES})}
    # Query realism is a query-level decision, and must agree across its rows.
    query_labels = defaultdict(set)
    for row in resolved.values():
        query_labels[row["query_id"]].add(row["query_realism"])
    inconsistent_queries = {identity for identity, labels in query_labels.items() if len(labels) > 1}
    for identity in list(resolved):
        if resolved[identity]["query_id"] in inconsistent_queries:
            needs[identity] = items[identity]
            del resolved[identity]
        elif resolved[identity]["source_quality"] == "corrupt" and resolved[identity]["relevance"] != "0":
            needs[identity] = {**items[identity], "notes": "Source judged corrupt but marked relevant; adjudicate this contradiction."}
            del resolved[identity]
    initially_disputed = len(needs)
    if adjudication_path is not None:
        if missing:
            raise ValueError("Complete both independent forms before adjudicating disagreements")
        adjudicated = _read_form(adjudication_path, {identity: items[identity] for identity in needs})
        for identity, row in adjudicated.items():
            if not complete(row) or row["query_realism"] == "uncertain" or row["source_quality"] == "uncertain" or row["relevance"] == "unknown":
                continue
            if row["source_quality"] == "corrupt" and row["relevance"] != "0":
                continue
            if row["reviewer"].strip().casefold() in {a[identity]["reviewer"].strip().casefold(), b[identity]["reviewer"].strip().casefold()}:
                raise ValueError("Adjudicator must differ from both initial raters")
            resolved[identity] = {**items[identity], **{key: row[key] for key in CHOICES},
                                  "reviewers": [a[identity]["reviewer"], b[identity]["reviewer"], row["reviewer"]]}
            del needs[identity]
    by_query = defaultdict(list)
    for row in resolved.values():
        by_query[row["query_id"]].append(row)
    consistency = all(len({row["query_realism"] for row in rows}) == 1 for rows in by_query.values())
    corrupt_positive = [row["item_id"] for row in resolved.values() if row["source_quality"] == "corrupt" and row["relevance"] != "0"]
    ready = not missing and not needs and consistency and not corrupt_positive
    if any((output_dir / name).exists() for name in
           ["adjudication_to_complete.csv", "review_import_report.json", "human_validation_queries.jsonl"]):
        raise FileExistsError("Use a new import-output directory; preserve prior judgments and receipts")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "adjudication_to_complete.csv", list(needs.values()))
    report = {"status": "ready_for_validation_export" if ready else "needs_human_review", "items": len(items),
              "missing_rater_item_judgments": missing, "initial_disputed_or_uncertain_items": initially_disputed,
              "unresolved_items": len(needs), "resolved_items": len(resolved),
              "query_realism_consistent": consistency, "corrupt_passages_marked_relevant": corrupt_positive,
              "human_validation_exported": ready, "final_test_scored": False,
              "review_scope": manifest.get("review_scope", "full_validation_pooled"),
              "full_validation_review_complete": ready and manifest.get("review_scope") == "full_validation_pooled",
              "packet_items_sha256": manifest["items_sha256"],
              "limitation": "Pooled human judgments do not establish exhaustive corpus relevance or real-user query quality."}
    if ready:
        rows = []
        for identity, judgments in sorted(by_query.items()):
            if judgments[0]["query_realism"] != "realistic":
                continue
            relevance = {row["chunk_id"]: int(row["relevance"]) for row in judgments if row["source_quality"] == "readable"}
            rows.append({"query_id": identity, "query": judgments[0]["query"], "doc_id": judgments[0]["doc_id"],
                         "source_group_id": judgments[0]["source_group_id"], "split": "validation",
                         "label_kind": "human_pooled_relevance", "query_origin": manifest["query_origin"],
                         "review_scope": manifest.get("review_scope", "full_validation_pooled"),
                         "positive_ids": sorted(key for key, grade in relevance.items() if grade > 0),
                         "relevance_grades": relevance, "unjudged_policy": "unknown_not_automatically_irrelevant",
                         "pool_depth": manifest["pool_depth"], "reviewers": sorted({name for row in judgments for name in row["reviewers"]})})
        with (output_dir / "human_validation_queries.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["exported_realistic_queries"] = len(rows)
    (output_dir / "review_import_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
