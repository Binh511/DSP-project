"""Create evidence-cited project plans without inventing unsupported tasks."""

from __future__ import annotations

from typing import Any

import hashlib
import math
import re

from extraction import _clean_value, _PAST

PLAN_MIN_SCORE = 0.80   # chặt hơn ngưỡng trích xuất, vì mỗi bước sai là một việc sai

_LEAD = re.compile(r"^[\s\-●*•]*(?:\d+[.)]\s*)*")
_STATUS_TAIL = re.compile(r"\s*\((?:reviewed|done|completed|in progress)[^)]*\)\s*$", re.I)
_COMPLETED_TAIL = re.compile(r"\((?:reviewed|done|completed)\b[^)]*\)\s*$", re.I)
_OPTION = re.compile(r"^\W*(?:we could|another option|alternatively|one option|it uses|it is)\b", re.I)
_FRAGMENT = re.compile(r"\.\.\s*\w*\s*$")
_CUE = re.compile(r"\b(will|need to|needs to|must|should|shall|plan to|going to)\b", re.I)
_IMPER_START = re.compile(
    r"^\W*(?:\d+[.)]\s*)*(?:[○●•-]\s*)*(implement|add|create|build|develop|design|fix|write|update|"
    r"integrate|test|deploy|refactor|provide|release|document|complete|improve|support|publish|"
    r"migrate|define|extend|expand|establish|finalize|finalise)\b", re.I)
_DATE = re.compile(
    r"\b(week\s*\d+|\d{1,2}\s+\w{3,9}\s+\d{4}|\w{3,9}\s+\d{1,2},\s*\d{4}|q[1-4]|milestone)\b", re.I)
_PLAN_SECTION = re.compile(r"timeline|schedule|milestone|deliverable|work\s*plan|roadmap", re.I)


def _plan_key(text: str) -> str:
    text = _STATUS_TAIL.sub("", _LEAD.sub("", text))
    return re.sub(r"\W+", " ", text).casefold().strip()


def _plan_worthy(item: dict[str, Any]) -> bool:
    value = item["value"].strip()
    if item.get("label_score", 1.0) < PLAN_MIN_SCORE:   # deliverables không có điểm: giữ
        return False
    if (_OPTION.match(value) or _FRAGMENT.search(value) or _COMPLETED_TAIL.search(value)
            or _PAST.match(value)):
        return False
    # chỉ giữ câu có dấu hiệu của một việc cần làm
    return bool(
        _CUE.search(value)
        or _IMPER_START.match(value)
        or _DATE.search(value)
        or _PLAN_SECTION.search(item.get("section", ""))
    )


def _verified_items(record: dict[str, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        item
        for field in fields
        for item in record.get("fields", {}).get(field, [])
        if item.get("verified") is True and item.get("evidence_id")
    ]


def _evidence_index(record: dict[str, Any], source_text: str | None) -> dict[str, dict]:
    """Validate stored evidence, optionally against the actual cleaned source.

    Without source_text this checks internal consistency only; it cannot prove
    that a caller-supplied verified flag matches a real source document.
    """
    doc = record.get("document_id")
    if not isinstance(doc, str) or not doc.strip():
        raise ValueError("Extraction record needs a non-empty document_id")
    if source_text is not None:
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if digest != record.get("source_sha256"):
            raise ValueError("Extraction source hash does not match the provided source")
    fields = record.get("fields", {})
    if not isinstance(fields, dict):
        raise ValueError("Extraction fields must be a mapping")
    evidence = {}
    for entries in fields.values():
        if not isinstance(entries, list):
            raise ValueError("Each extraction field must be a list")
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError("Evidence entries must be objects")
            if item.get("verified") is not True:
                continue
            key, quote, value = (item.get(k) for k in ("evidence_id", "quote", "value"))
            start, end = item.get("char_start"), item.get("char_end")
            if (not isinstance(key, str) or not key.startswith(doc + ":")
                    or not isinstance(quote, str) or not quote.strip()
                    or not isinstance(value, str) or value.strip() != _clean_value(quote).strip()
                    or type(start) is not int or type(end) is not int
                    or start < 0 or end != start + len(quote)):
                raise ValueError("Verified evidence has invalid IDs, values or source offsets")
            if source_text is not None and (end > len(source_text) or source_text[start:end] != quote):
                raise ValueError(f"Evidence {key} does not match its source span")
            score = item.get("label_score", 1.0)
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(f"Evidence {key} has an invalid confidence")
            if key in evidence:
                old = evidence[key]
                if any(old.get(k) != item.get(k) for k in ("quote", "value", "char_start", "char_end")):
                    raise ValueError(f"Evidence ID {key} refers to conflicting source items")
            evidence[key] = item
    return evidence


def generate_plan(record: dict[str, Any], *, source_text: str | None = None) -> dict[str, Any]:
    evidence = _evidence_index(record, source_text)
    candidates = _verified_items(record, ("activities", "deliverables"))
    # sắp theo thứ tự xuất hiện trong tài liệu gốc
    candidates.sort(key=lambda it: it.get("char_start", 10**9))

    seen: set[str] = set()
    steps = []
    for item in candidates:
        if not _plan_worthy(item):
            continue
        description = item["value"].strip()
        key = _plan_key(description)
        if not description or not key or key in seen:
            continue
        seen.add(key)
        steps.append({
            "step": len(steps) + 1,
            "description": description,
            "evidence_ids": [item["evidence_id"]],
            "status": "proposed",
        })
    plan = {
        "document_id": record["document_id"],
        "source_sha256": record.get("source_sha256"),
        "planning_method": "evidence_ordered_source_items",
        "grounding_validation": "source_hash_and_spans" if source_text is not None else "record_consistency_only",
        "steps": steps,
        "related_context": record.get("related_context", []),
    }
    validate_plan(plan, set(evidence), evidence_by_id=evidence)
    return plan


def validate_plan(plan: dict[str, Any], allowed_evidence: set[str], *,
                  evidence_by_id: dict[str, dict] | None = None) -> None:
    """Validate citations; pass evidence_by_id to also enforce source wording."""
    if (not isinstance(plan.get("document_id"), str) or not plan["document_id"].strip()
            or not isinstance(plan.get("steps"), list)):
        raise ValueError("Plan must contain a document_id and a steps list.")
    for index, step in enumerate(plan["steps"], 1):
        if (not isinstance(step, dict) or type(step.get("step")) is not int
                or step["step"] != index or not isinstance(step.get("description"), str)
                or not step["description"].strip()):
            raise ValueError(f"Invalid plan step at position {index}.")
        evidence_ids = step.get("evidence_ids")
        if (not isinstance(evidence_ids, list) or not evidence_ids
                or any(not isinstance(e, str) for e in evidence_ids)
                or len(set(evidence_ids)) != len(evidence_ids)):
            raise ValueError(f"Plan step {index} must cite evidence.")
        if not set(evidence_ids).issubset(allowed_evidence):
            raise ValueError(f"Plan step {index} cites unknown or unverified evidence.")
        if evidence_by_id is not None:
            if any(e not in evidence_by_id for e in evidence_ids):
                raise ValueError(f"Plan step {index} has no corresponding evidence entry")
            if not any(step["description"].strip() == evidence_by_id[e]["value"].strip()
                       for e in evidence_ids):
                raise ValueError(f"Plan step {index} adds wording unsupported by its cited evidence")
