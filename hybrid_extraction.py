"""Merge source-grounded extracted evidence with retrieved related context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            document_id = row.get("document_id")
            if not isinstance(document_id, str) or document_id in rows:
                raise ValueError(f"{path}:{number}: invalid or duplicate document_id")
            rows[document_id] = row
    return rows


def run(
    *,
    cleaned_dir: Path,
    rules_path: Path,
    bert_path: Path | None,
    t5_path: Path | None,
    output_path: Path,
    contexts_path: Path,
    document_ids: set[str],
) -> None:
    del cleaned_dir, bert_path, t5_path
    rules = _read_jsonl(rules_path)
    contexts = _read_jsonl(contexts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for document_id in sorted(document_ids):
            if document_id not in rules:
                raise ValueError(f"Missing extracted evidence for {document_id}.")
            rule = rules[document_id]
            context = contexts.get(document_id, {})
            stream.write(json.dumps({
                "document_id": document_id,
                "source_sha256": rule["source_sha256"],
                "fields": rule["fields"],
                "related_context": context.get("hits", []),
                "merge_method": "verified_source_evidence_plus_bge_m3_retrieval",
            }, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
