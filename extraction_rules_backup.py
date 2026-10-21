"""Conservative, source-grounded evidence extraction for cleaned documents."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterator

FIELD_CUES = {
    "objectives": re.compile(r"\b(want to|aim to|goal|objective|intend to|keep .+ as)\b", re.I),
    "requirements": re.compile(r"\b(must|should|requirement|support|need to|has to|without|no reliance)\b", re.I),
    "activities": re.compile(r"\b(develop|design|collect|study|implement|build|add|deploy|launch|provide|create)\b", re.I),
    "deliverables": re.compile(r"\b(dashboard|script|library|release|documentation|tool|module|application|report)\b", re.I),
    "expected_results": re.compile(r"\b(result|benefit|enable|allow|users will|will be|to keep|improve)\b", re.I),
    "technologies": re.compile(r"\b(Python|Prometheus|Grafana|Docker|API|Lotus|Filecoin|ChromaDB)\b", re.I),
    "resources": re.compile(r"\b(hours?|costs?|budget|team|community|miners?|users?)\b", re.I),
}


def iter_documents(cleaned_dir: Path) -> Iterator[tuple[str, Path]]:
    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Cleaned document directory not found: {cleaned_dir}")
    paths = sorted((*cleaned_dir.rglob("*.md"), *cleaned_dir.rglob("*.txt")))
    seen: set[str] = set()
    for path in paths:
        document_id = path.parent.name
        if document_id in seen:
            raise ValueError(f"Multiple cleaned source files found for {document_id}.")
        seen.add(document_id)
        yield document_id, path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _sentences(text: str) -> Iterator[str]:
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = part.strip()
        if sentence and not sentence.startswith("#"):
            yield sentence


def extract_document(document_id: str, text: str) -> dict[str, Any]:
    fields: dict[str, list[dict[str, Any]]] = {name: [] for name in FIELD_CUES}
    evidence_index = 0
    for sentence in _sentences(text):
        matched = [name for name, cue in FIELD_CUES.items() if cue.search(sentence)]
        for field in matched:
            evidence_id = f"{document_id}:E{evidence_index:05d}"
            evidence_index += 1
            fields[field].append({
                "value": sentence,
                "quote": sentence,
                "evidence_id": evidence_id,
                "verified": True,
                "verification": "verbatim_source_match",
            })
    return {
        "document_id": document_id,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "extraction_method": "source_cue_rules",
        "fields": fields,
    }
