"""Source-grounded evidence extraction: DeBERTa sentence classifier + cue rules."""

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
    "resources": re.compile(
        r"(\$\s?\d|\b\d[\d,.]*\s?(?:usd|eur|hours?|days?|weeks?|months?|FTE|person|developers?|engineers?)\b"
        r"|\b(?:budget|funding|cost|salary|salaries|compensation)\b)", re.I),
}

# Nhãn của mô hình -> tên trường trong kết quả
LABEL_TO_FIELD = {
    "OBJECTIVE": "objectives",
    "REQUIREMENT": "requirements",
    "ACTIVITY": "activities",
    "EXPECTED_RESULT": "expected_results",
}
# Các trường mô hình không phân loại: vẫn dùng regex như cũ
RULE_ONLY_FIELDS = ("deliverables", "technologies", "resources")

# Ngưỡng điểm tin cậy riêng cho từng nhãn (chọn từ bảng P/R trên tập test)
MIN_SCORE_BY_LABEL = {
    "OBJECTIVE": 0.60,
    "ACTIVITY": 0.70,
    "EXPECTED_RESULT": 0.70,
    "REQUIREMENT": 0.60,
}
DEFAULT_MIN_SCORE = 0.5

MIN_CHARS = 15       # sau khi làm sạch, ngắn hơn thì bỏ
MIN_WORDS = 4

# Từ lửng: câu kết thúc bằng các từ này thường là mảnh bị cắt dở
DANGLING_WORDS = {
    "a", "an", "the", "any", "to", "of", "and", "or", "as", "for",
    "with", "in", "on", "by", "at", "from", "that", "which", "showing",
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


def _is_table_row(sentence: str) -> bool:
    return sentence.count("|") >= 2


def _clean_value(sentence: str) -> str:
    """Dòng bảng markdown -> ghép các ô thành một câu; câu thường giữ nguyên."""
    if not _is_table_row(sentence):
        return sentence
    cells = [c.strip() for c in sentence.split("|") if c.strip()]
    return " ".join(cells)


_NUMBERED_TITLE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+|[-*]\s*)?[A-Z][^.!?]{0,60}$")
_CITATION = re.compile(r"\[\[?\d+\]?\]|\(#page-")
_PAST = re.compile(
    r"^\s*(?:[-*]\s*)?(?:"
    r"(?:We|I)\s+(?:now\s+)?(?:discuss(?:ed)?|dug|learned|conducted|found|observed|showed|ran|tested|created|did)\b"
    r"|(?:We|I)\s+(?:have|has)\s+(?:also\s+)?(?:managed|already|successfully|completed|built|implemented)\b"
    r"|(?:Created|Compared|Provided|Tested|Observed|Found)\b"
    r"|.{0,60}\b(?:were|was)\s+(?:provided|compared|analy[sz]ed|conducted|used|obtained)\b"
    r")",
    re.I)

_REF_SECTION = re.compile(r"reference|bibliograph|acknowledg|appendix", re.I)


def _is_noise(value: str) -> bool:
    """Mảnh quá ngắn, bị cắt dở, tiêu đề hoặc câu trích dẫn: không nhận từ mô hình."""
    words = value.split()
    if len(value) < MIN_CHARS or len(words) < MIN_WORDS:
        return True
    if value.rstrip().endswith(","):
        return True
    if _CITATION.search(value):          # câu trích dẫn tài liệu khác
        return True
    if _NUMBERED_TITLE.match(value) and not value.rstrip().endswith("."):
        return True                      # tiêu đề ngắn không có dấu chấm
    if _PAST.match(value):
        return True
    return words[-1].strip(".,;:").lower() in DANGLING_WORDS


_PROTECT = re.compile(r"\b(e\.g|i\.e|etc|vs|Fig|No|approx|Dr|Mr|Ms)\.", re.I)
_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP = re.compile(r"^[|\s:\-]+$")


def _split_sentences(line: str) -> list[str]:
    protected = _PROTECT.sub(lambda m: m.group(0).replace(".", "§"), line)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9\"'(])", protected)
    return [p.replace("§", ".").strip() for p in parts if p.strip()]


def _sentences_with_context(text: str) -> Iterator[tuple[str, str, str]]:
    """Backward-compat wrapper used by older code paths."""
    for item in sentences_with_context(text):
        yield item["text"], item["section_last"], item["prev"]


def sentences_with_context(text: str) -> Iterator[dict]:
    """Mỗi câu kèm: đường dẫn heading, heading cuối, câu trước, vị trí ký tự."""
    path: list[str] = []
    prev = ""
    in_fence = False
    offset = 0
    total_lines = max(len(text.split("\n")) - 1, 1)
    pending = None

    for line_no, raw in enumerate(text.split("\n")):
        line_start = offset
        offset += len(raw) + 1
        s = raw.strip()

        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s or s.startswith(">"):
            continue
        if _TABLE_SEP.match(s):
            continue

        m = _HEAD.match(s)
        if m:
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            prev = s
            continue

        for sent in _split_sentences(s):
            pos = text.find(sent, line_start)
            item = {
                "text": sent,
                "section": " > ".join(path),
                "section_last": path[-1] if path else "",
                "prev": prev,
                "pos": line_no / total_lines,
                "start": pos,
                "end": pos + len(sent) if pos >= 0 else -1,
            }
            if pending is not None:
                pending["next_text"] = sent
                yield pending
            pending = item
            prev = sent

    if pending is not None:
        pending["next_text"] = ""
        yield pending


def extract_document(document_id: str, text: str, use_model: bool = True) -> dict[str, Any]:
    fields: dict[str, list[dict[str, Any]]] = {name: [] for name in FIELD_CUES}
    items = list(sentences_with_context(text))
    for k, it in enumerate(items):
        it["next"] = items[k + 1]["text"] if k + 1 < len(items) else ""

    predictions: list[dict[str, Any] | None] = [None] * len(items)
    if use_model and items:
        import ensemble_classifier as sentence_classifier
        predictions = sentence_classifier.classify([
            {"text": i["text"], "section": i["section"], "prev_text": i["prev"],
             "pos": i["pos"], "next_text": i["next"]}
            for i in items
        ])

    for idx, (it, pred) in enumerate(zip(items, predictions)):
        sentence = it["text"]
        value = _clean_value(sentence)
        evidence_id = f"{document_id}:S{idx:05d}"
        verified = it["start"] >= 0 and text[it["start"]:it["end"]] == sentence

        if pred is None:
            targets = [n for n, cue in FIELD_CUES.items() if cue.search(sentence)]
        else:
            targets = []
            field = LABEL_TO_FIELD.get(pred["label"])
            thr = MIN_SCORE_BY_LABEL.get(pred["label"], DEFAULT_MIN_SCORE)
            if (field and pred["score"] >= thr and not _is_noise(value)
                    and not _REF_SECTION.search(it["section"])):
                targets.append(field)
            if (pred["label"] in ("ACTIVITY", "EXPECTED_RESULT")
                    and FIELD_CUES["deliverables"].search(sentence)):
                targets.append("deliverables")
            targets += [n for n in ("technologies", "resources")
                        if FIELD_CUES[n].search(sentence)]

        for field in dict.fromkeys(targets):
            entry = {
                "value": value,
                "quote": sentence,
                "evidence_id": evidence_id,
                "section": it["section"],
                "char_start": it["start"],
                "char_end": it["end"],
                "verified": verified,
                "verification": "offset_match" if verified else "failed",
            }
            if pred is not None and field in LABEL_TO_FIELD.values():
                entry["label_score"] = pred["score"]
            fields[field].append(entry)

    return {
        "document_id": document_id,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "extraction_method": "deberta_classifier_plus_cue_rules" if use_model else "source_cue_rules",
        "fields": fields,
    }