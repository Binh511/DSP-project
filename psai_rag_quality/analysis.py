"""Paired retrieval diagnostics with source-cluster uncertainty estimates."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


METRICS = ("Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(ranked_ids: list[str], positive_ids: list[str]) -> dict[str, float]:
    if not positive_ids or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Metrics require positives and a ranking without repeated IDs")
    positives = set(positive_ids)
    ranks = [rank for rank, value in enumerate(ranked_ids[:10], 1) if value in positives]
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(positives)) + 1))
    return {
        **{f"Recall@{k}": len(set(ranked_ids[:k]) & positives) / len(positives) for k in [1, 5, 10]},
        "MRR@10": 1 / ranks[0] if ranks else 0.0,
        "nDCG@10": dcg / ideal,
    }


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def grouped_bootstrap(rows: list[dict], *, samples: int = 10000, seed: int = 42) -> dict:
    """Resample whole source groups, preserving all paired queries in each draw.

    Each replicate reports a query-weighted mean delta. Groups are sampled with
    replacement; their query counts travel with them, rather than treating 54
    correlated queries as independent observations.
    """
    if not rows or samples < 100:
        raise ValueError("Bootstrap requires observations and at least 100 resamples")
    groups = defaultdict(list)
    for row in rows:
        value = float(row["delta"])
        if not math.isfinite(value) or not row.get("source_group_id"):
            raise ValueError("Finite paired deltas and source groups are required")
        groups[row["source_group_id"]].append(value)
    if len(groups) < 2:
        raise ValueError("At least two independent source groups are needed")
    values = [(sum(group), len(group)) for _, group in sorted(groups.items())]
    rng = random.Random(seed)
    replicates = []
    for _ in range(samples):
        selected = [values[rng.randrange(len(values))] for _ in values]
        replicates.append(sum(value for value, _ in selected) / sum(count for _, count in selected))
    low, high = _quantile(replicates, 0.025), _quantile(replicates, 0.975)
    return {
        "estimate": statistics.mean(float(row["delta"]) for row in rows),
        "source_macro_delta": statistics.mean(total / count for total, count in values),
        "ci95_low": low, "ci95_high": high, "ci_includes_zero": low <= 0 <= high,
        "source_groups": len(values), "queries": len(rows), "samples": samples, "seed": seed,
        "method": "paired-source-cluster-percentile-bootstrap-query-weighted",
        "limitation": "Sampling uncertainty conditional on this synthetic dataset; excludes label error and training-seed uncertainty.",
    }


def _prediction_index(path: Path) -> dict:
    result = {}
    for row in read_jsonl(path):
        key = (row["query_id"], row["scope"], row["method"])
        if key in result:
            raise ValueError("Repeated prediction identity")
        recalculated = metrics([hit["chunk_id"] for hit in row["ranking"]], row["positive_ids"])
        if any(abs(recalculated[name] - row["metrics"][name]) > 1e-10 for name in METRICS):
            raise ValueError("Stored retrieval metrics differ from the recorded ranking")
        result[key] = row
    return result


def analyze(study_dir: Path, output_dir: Path, *, samples: int = 10000, seed: int = 42) -> dict:
    study_dir, output_dir = Path(study_dir), Path(output_dir)
    left = _prediction_index(study_dir / "base_predictions.jsonl")
    right = _prediction_index(study_dir / "adapter_predictions.jsonl")
    if set(left) != set(right):
        raise ValueError("Base/adapter must have exactly paired query/scope/method identities")
    queries = {row["query_id"]: row for row in read_jsonl(study_dir / "validation_pairs.jsonl")}
    if any(row.get("split") != "validation" for row in queries.values()):
        raise ValueError("Only validation queries can be analyzed")
    paired, grouped = [], defaultdict(list)
    for key in sorted(left):
        base, adapter = left[key], right[key]
        reference = queries.get(key[0])
        if reference is None or any(base[name] != adapter[name] or base[name] != reference[name]
                                    for name in ["query", "doc_id", "source_group_id"]):
            raise ValueError("Prediction/source identities are inconsistent")
        if set(base["positive_ids"]) != set(adapter["positive_ids"]):
            raise ValueError("Base/adapter positive sets differ")
        delta = adapter["metrics"]["MRR@10"] - base["metrics"]["MRR@10"]
        row = {"query_id": key[0], "query": base["query"], "doc_id": base["doc_id"],
               "source_group_id": base["source_group_id"], "scope": key[1], "method": key[2],
               "base_mrr": base["metrics"]["MRR@10"], "adapter_mrr": adapter["metrics"]["MRR@10"],
               "delta": delta, "outcome": "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie",
               "base_positive_rank": next((hit["rank"] for hit in base["ranking"] if hit["chunk_id"] in base["positive_ids"]), None),
               "adapter_positive_rank": next((hit["rank"] for hit in adapter["ranking"] if hit["chunk_id"] in adapter["positive_ids"]), None),
               "base_top1": base["ranking"][0]["chunk_id"], "adapter_top1": adapter["ranking"][0]["chunk_id"]}
        paired.append(row)
        grouped[key[1], key[2]].append(row)
    summary, per_document = {}, []
    for (scope, method), rows in sorted(grouped.items()):
        key = scope + "/" + method
        keys = [(row["query_id"], scope, method) for row in rows]
        summary[key] = {
            "scope": scope, "method": method,
            "wins": sum(row["outcome"] == "win" for row in rows),
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "ties": sum(row["outcome"] == "tie" for row in rows),
            "base": {name: statistics.mean(left[item]["metrics"][name] for item in keys) for name in METRICS},
            "adapter": {name: statistics.mean(right[item]["metrics"][name] for item in keys) for name in METRICS},
            "mrr_delta_bootstrap": grouped_bootstrap(rows, samples=samples, seed=seed),
        }
        by_doc = defaultdict(list)
        for row in rows:
            by_doc[row["doc_id"]].append(row)
        for doc_id, doc_rows in sorted(by_doc.items()):
            per_document.append({"doc_id": doc_id, "source_group_id": doc_rows[0]["source_group_id"],
                                 "scope": scope, "method": method, "queries": len(doc_rows),
                                 "base_mrr": statistics.mean(row["base_mrr"] for row in doc_rows),
                                 "adapter_mrr": statistics.mean(row["adapter_mrr"] for row in doc_rows),
                                 "delta": statistics.mean(row["delta"] for row in doc_rows),
                                 "wins": sum(row["outcome"] == "win" for row in doc_rows),
                                 "losses": sum(row["outcome"] == "loss" for row in doc_rows)})
    report = {"label_kind": "synthetic_source_chunk_pseudopositives", "final_test_scored": False,
              "queries": len(queries), "source_groups": len({row["source_group_id"] for row in queries.values()}),
              "summary": summary, "inputs_sha256": {name: digest_file(study_dir / name) for name in
                  ["base_predictions.jsonl", "adapter_predictions.jsonl", "validation_pairs.jsonl"]},
              "limitations": ["Six related comparisons are exploratory, not six independent confirmatory tests.",
                              "Confidence intervals do not validate the synthetic relevance labels.",
                              "One adapter seed and 22 validation sources do not support broad superiority claims."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, rows in [("paired_queries.csv", paired), ("paired_documents.csv", per_document)]:
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report
