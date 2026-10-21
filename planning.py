"""Create evidence-cited project plans without inventing unsupported tasks."""

from __future__ import annotations

from typing import Any


def _verified_items(record: dict[str, Any], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        item
        for field in fields
        for item in record.get("fields", {}).get(field, [])
        if item.get("verified") is True and item.get("evidence_id")
    ]


def generate_plan(record: dict[str, Any]) -> dict[str, Any]:
    candidates = _verified_items(record, ("activities", "deliverables"))
    # sắp theo thứ tự xuất hiện trong tài liệu gốc
    candidates.sort(key=lambda it: it.get("char_start", 10**9))

    seen: set[str] = set()
    steps = []
    for item in candidates:
        description = item["value"].strip()
        if not description or description.casefold() in seen:
            continue
        seen.add(description.casefold())
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
        "steps": steps,
        "related_context": record.get("related_context", []),
    }
    validate_plan(plan, {
        item["evidence_id"]
        for values in record.get("fields", {}).values()
        for item in values
        if item.get("verified") is True and item.get("evidence_id")
    })
    return plan


def validate_plan(plan: dict[str, Any], allowed_evidence: set[str]) -> None:
    if not isinstance(plan.get("document_id"), str) or not isinstance(plan.get("steps"), list):
        raise ValueError("Plan must contain a document_id and a steps list.")
    for index, step in enumerate(plan["steps"], 1):
        if step.get("step") != index or not isinstance(step.get("description"), str):
            raise ValueError(f"Invalid plan step at position {index}.")
        evidence_ids = step.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError(f"Plan step {index} must cite evidence.")
        if not set(evidence_ids).issubset(allowed_evidence):
            raise ValueError(f"Plan step {index} cites unknown or unverified evidence.")
