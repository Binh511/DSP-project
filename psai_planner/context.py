"""Literal source context and explicitly unreviewed lexical task associations.

This layer does not correct classifier labels or certify requirement coverage.
It makes potentially missed source material visible without model inference.
"""
from __future__ import annotations

import re
from collections import defaultdict


SCHEMA_VERSION = "psai-review-context-v1"
MAX_EVIDENCE = 512
MAX_LINKS_PER_ITEM = 4
CATEGORIES = ("requirements", "acceptance", "dependency_constraints", "risks",
              "personnel", "measures", "open_questions")
_STOP = set("a an the this that these those we will must should shall may can could be been "
            "is are was were to of and or in on at as by for from with without not no only "
            "it its their they our has have had do does did project system proposed proposal "
            "source document task activity activities need needs include including each all "
            "before after using use used make final em va và là của cho các một những cần phải "
            "được không trong với theo từ khi sẽ dự án hệ thống".split())
_CUES = {
    "requirements": re.compile(r"\b(?:must|shall|should|required|requires?|need(?:s)? to)\b|phải|yêu cầu", re.I),
    "acceptance": re.compile(r"acceptance|nghiệm thu|tiêu chí", re.I),
    "dependency_constraints": re.compile(r"\b(?:precede[sd]?|depends? on|dependent on|prerequisite|before|after)\b|phụ thuộc|trước khi|sau khi", re.I),
    "risks": re.compile(r"\b(?:risk\s*(?:r?\d+)?|mitigation)\b|rủi ro|giảm thiểu", re.I),
    "personnel": re.compile(r"\b(?:team|staffing|personnel|developers?|engineers?|staff|operators?|lead|reviewers?)\b|nhân sự|kỹ sư|đội ngũ", re.I),
    "measures": re.compile(r"\b(?:target|percent|percentage|measur\w*|minutes?|seconds?)\b|\d\s*%|mục tiêu|chỉ số|phần trăm|phút|giây", re.I),
    "open_questions": re.compile(r"\b(?:pending|unresolved|not (?:yet )?(?:approved|confirmed)|remain(?:s)? (?:unknown|open)|not assign|not reported as passed|future controlled)\b|chưa xác nhận|chưa duyệt|chưa phê duyệt|chưa xác định|còn bỏ ngỏ", re.I),
}


def _terms(text: str) -> set[str]:
    result = set()
    for word in re.findall(r"[^\W\d_]+", text.casefold()):
        if len(word) < 3 or word in _STOP:
            continue
        # Limited morphology is a matching heuristic, never semantic entailment.
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            word = word[:-1]
        for suffix in ("ing", "ment", "ed"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[:-len(suffix)]
                break
        if word.endswith("e") and len(word) > 4:
            word = word[:-1]
        result.add(word)
    return result


def _source_segments(text: str):
    """Yield untouched sentence/line spans with the nearest Markdown heading."""
    section = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
        elif stripped and not re.fullmatch(r"[\s|:\-]+", stripped):
            # Never normalize or rebuild wording; offsets refer to original text.
            for match in re.finditer(r"\S(?:.*?)(?:(?<=[.!?])(?=\s+[A-ZÀ-Ỹ])|$)", line.rstrip("\r\n")):
                start, end = match.span()
                while start < end and line[start] in " \t-*•":
                    start += 1
                while end > start and line[end - 1].isspace():
                    end -= 1
                if end > start:
                    yield offset + start, offset + end, section
        offset += len(line)


def _categories(quote: str, section: str) -> list[str]:
    found = {key for key, pattern in _CUES.items() if pattern.search(quote)}
    if re.search(r"requirements?|yêu cầu", section, re.I):
        found.add("requirements")
    if _CUES["acceptance"].search(section):
        found.add("acceptance")
    # Merely mentioning an operator does not establish a staffing fact.
    if "personnel" in found and not re.search(
            r"\b(?:team|engineers?|developers?|consists? of|part.time|responsible|assigned|staffing (?:is|will|includes?|requires?)|personnel (?:is|will|includes?|requires?))\b|nhân sự|kỹ sư|đội ngũ|phụ trách", quote, re.I):
        found.remove("personnel")
    return [key for key in CATEGORIES if key in found]


def _document_hits(record: dict, text: str, retrieval: dict | None) -> list[dict]:
    from .planner import PlannerValidationError
    if retrieval is None:
        return []
    if not isinstance(retrieval, dict) or not isinstance(retrieval.get("hits"), list):
        raise PlannerValidationError("Retrieval must contain a hits list")
    seen = set()
    result = []
    digest = record["source_sha256"]
    for hit in retrieval["hits"]:
        if not isinstance(hit, dict):
            raise PlannerValidationError("Retrieval hits must be objects")
        if hit.get("role") != "document":
            continue
        eid, start, end, quote = (hit.get(k) for k in ("evidence_id", "char_start", "char_end", "quote"))
        if (hit.get("doc_id") != record["document_id"] or hit.get("source_sha256") != digest
                or hit.get("verified") is not True or not isinstance(eid, str) or not eid
                or eid in seen or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or text[start:end] != quote):
            raise PlannerValidationError("Document retrieval evidence ID/hash/span is invalid")
        seen.add(eid)
        result.append({"evidence_id": eid, "quote": quote, "char_start": start,
                       "char_end": end, "source_sha256": digest, "role": "document",
                       "verified": True, "needs_review": True,
                       "relevance_status": "algorithmic_candidate_not_human_verified"})
    return result


def build_review_context(record: dict, source_text: str, retrieval: dict | None = None) -> dict:
    """Expose source-literal review items and bounded lexical task candidates.

    ``verified`` means exact source provenance. Categories, links and completeness
    remain unreviewed heuristics. No acceptance result, assignment or scheduling
    relation is inferred as a fact.
    """
    from .planner import _source_evidence
    from planning import generate_plan
    _source_evidence(record, source_text)
    document_hits = _document_hits(record, source_text, retrieval)
    digest, document = record["source_sha256"], record["document_id"]
    evidence = []
    total = 0
    for start, end, section in _source_segments(source_text):
        quote = source_text[start:end]
        categories = _categories(quote, section)
        if not categories:
            continue
        total += 1
        if len(evidence) < MAX_EVIDENCE:
            evidence.append({"evidence_id": f"{document}:C{start:06d}_{end:06d}",
                             "quote": quote, "char_start": start, "char_end": end,
                             "source_sha256": digest, "section": section,
                             "categories": categories, "verified": True,
                             "category_basis": "lexical_and_section_heuristics",
                             "status": "source_excerpt", "needs_review": True})
    steps = generate_plan(record, source_text=source_text)["steps"]
    tasks = [{"task_id": f"T{index:03d}", "description": step["description"],
              "evidence_ids": step["evidence_ids"]} for index, step in enumerate(steps, 1)]
    inverted = defaultdict(set)
    task_terms = {}
    for task in tasks:
        task_terms[task["task_id"]] = _terms(task["description"])
        for term in task_terms[task["task_id"]]:
            inverted[term].add(task["task_id"])
    by_id = {task["task_id"]: task for task in tasks}
    links = []
    linked = set()
    omitted_links = 0
    for item in evidence:
        terms = _terms(item["quote"])
        candidates = set().union(*(inverted[term] for term in terms)) if terms else set()
        scored = []
        for tid in candidates:
            shared = sorted(terms & task_terms[tid])
            if len(shared) < 2:
                continue
            # A task's own source sentence adds no cross-sentence review value.
            if any(record_item.get("char_start") == item["char_start"]
                   and record_item.get("char_end") == item["char_end"]
                   for items in record["fields"].values() for record_item in items
                   if record_item.get("evidence_id") in by_id[tid]["evidence_ids"]):
                continue
            scored.append((len(shared), tid, shared))
        scored.sort(key=lambda row: (-row[0], row[1]))
        omitted_links += max(0, len(scored) - MAX_LINKS_PER_ITEM)
        for _, tid, shared in scored[:MAX_LINKS_PER_ITEM]:
            linked.add(item["evidence_id"])
            links.append({"task_id": tid, "task_evidence_ids": by_id[tid]["evidence_ids"],
                          "context_evidence_id": item["evidence_id"],
                          "categories": item["categories"], "shared_terms": shared,
                          "status": "lexical_candidate", "needs_review": True})
    questions = []
    for category, message in (
            ("personnel", "Confirm availability and individual ownership; a staffing mention is not an assignment."),
            ("measures", "Confirm measurement conditions, baseline and denominator; a target is not an achieved result."),
            ("dependency_constraints", "Map the exact constraint to task stages; before approval does not necessarily mean before starting."),
            ("open_questions", "Resolve the stated pending decision before making implementation commitments.")):
        ids = [item["evidence_id"] for item in evidence if category in item["categories"]]
        if ids:
            questions.append({"kind": category, "evidence_ids": ids, "message": message,
                              "status": "needs_review"})
    return {"schema_version": SCHEMA_VERSION, "document_id": document, "source_sha256": digest,
            "evidence": evidence, "task_catalog": tasks, "task_links": links,
            "unlinked_evidence_ids": [item["evidence_id"] for item in evidence if item["evidence_id"] not in linked],
            "review_questions": questions, "retrieval_context": document_hits,
            "omitted_counts": {"source_items": total - len(evidence), "lexical_task_links": omitted_links},
            "status": "unreviewed_source_context", "human_review_complete": False,
            "coverage_status": "not_assessed",
            "warnings": ["Categories and lexical links are review candidates, not confirmed requirement coverage or semantic entailment.",
                         "Source targets, proposed acceptance checks and staffing mentions do not prove achieved results, approvals or assignments."]}


def validate_review_context(context: dict, record: dict, source_text: str,
                            retrieval: dict | None = None) -> None:
    """Reject edits, stale source, invented spans or inflated review status."""
    from .planner import PlannerValidationError
    if context != build_review_context(record, source_text, retrieval):
        raise PlannerValidationError("Review context differs from the verified source and deterministic review policy")
