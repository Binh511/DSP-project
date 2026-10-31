"""Conservative patterns with exact Unicode source offsets and explicit ambiguity."""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import re


METHOD = "deterministic_source_patterns_v1"
FIELD_NAMES = ("dates", "identifiers", "milestones", "durations", "budgets", "technologies", "personnel")
_ISO_DATE = re.compile(r"(?<![\w/])(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?![\w/])")
_SLASH_DATE = re.compile(r"(?<![\w/])(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})(?![\w/])")
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
_BUDGET = re.compile(r"(?<![\w.,])(?:(?P<prefix>USD|EUR|VND|\$)\s*(?P<amount1>" + _NUMBER + r")(?!\d|[.,]\d)|"
                     r"(?P<amount2>" + _NUMBER + r")\s*(?P<suffix>USD|EUR|VND))(?!\w)", re.I)
_DURATION = re.compile(r"(?<![\w.,])(?P<number>\d{1,4}(?:[.,]\d{1,2})?)\s*(?P<unit>"
                       r"hours?|days?|weeks?|months?|years?|minutes?|giờ|ngày|tuần|tháng|năm|phút)(?!\w)", re.I)
_PERSONNEL = re.compile(r"(?<![\w.,])(?P<number>\d{1,3})\s*(?P<unit>"
                        r"developers?|engineers?|staff|people|persons?|team\s+members?|"
                        r"lập\s+trình\s+viên|kỹ\s+sư|nhân\s+sự|người)(?!\w)", re.I)
_IDENTIFIER = re.compile(r"(?<!\w)(?:(?:project|proposal|document|ticket|issue|grant|contract)\s+"
                         r"(?:id|identifier|no\.?|number)|ID|mã\s+(?:dự\s+án|tài\s+liệu|hồ\s+sơ))"
                         r"[ \t]*[:#=]?[ \t]+(?P<identifier>[A-Za-z0-9][A-Za-z0-9._/-]{0,63})(?!\w)", re.I)
_MILESTONE = re.compile(r"(?<!\w)(?:milestone|cột\s+mốc|mốc\s+tiến\s+độ)[ \t]*"
                        r"(?:#?\d{1,3}|[A-Z])?[ \t]*[:\-][ \t]*[^\n.!?]{3,160}", re.I)
_TECHNOLOGIES = ("Python", "JavaScript", "TypeScript", "Java", "Rust", "C++", "C#", "Docker",
                 "Kubernetes", "PostgreSQL", "MySQL", "SQLite", "Redis", "GitHub", "GitLab",
                 "PyTorch", "TensorFlow", "scikit-learn", "ChromaDB", "FAISS", "Prometheus",
                 "Grafana", "Filecoin", "Lotus", "React", "Next.js", "Node.js", "FastAPI", "Django",
                 "Flask", "BGE-M3", "mDeBERTa", "DeBERTa", "Qwen")
_TECH_PATTERN = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(value) for value in sorted(_TECHNOLOGIES, key=len, reverse=True))
                           + r")(?![\w+#])", re.I)
_CANONICAL_TECH = {value.casefold(): value for value in _TECHNOLOGIES}
_UNITS = {"hour": "hour", "hours": "hour", "giờ": "hour", "day": "day", "days": "day", "ngày": "day",
          "week": "week", "weeks": "week", "tuần": "week", "month": "month", "months": "month", "tháng": "month",
          "year": "year", "years": "year", "năm": "year", "minute": "minute", "minutes": "minute", "phút": "minute"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _amount(number: str) -> tuple[str | None, list[str]]:
    """Normalize only unambiguous separators; preserve alternatives otherwise."""
    if "," in number and "." in number:
        decimal_separator = "," if number.rfind(",") > number.rfind(".") else "."
        thousands = "." if decimal_separator == "," else ","
        normalized = number.replace(thousands, "").replace(decimal_separator, ".")
    elif number.count(",") > 1 or number.count(".") > 1:
        normalized = number.replace(",", "").replace(".", "")
    elif "," in number or "." in number:
        separator = "," if "," in number else "."
        left, right = number.split(separator)
        if len(right) == 3:
            return None, [left + right, left + "." + right]
        normalized = number.replace(",", ".")
    else:
        normalized = number
    try:
        result = format(Decimal(normalized), "f")
    except InvalidOperation:
        return None, []
    return result, []


def extract_facts(document_id: str, source_text: str) -> dict:
    """Extract literal mentions; all offsets use Python Unicode character units.

    A matched pattern is not a semantic guarantee. Amounts and date formats with
    multiple plausible interpretations retain ambiguity instead of guessing.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a nonempty string")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a nonempty string")
    digest = _hash(source_text)
    fields = {name: [] for name in FIELD_NAMES}
    seen = set()

    def add(field, match, value, policy, *, needs_review=False, start=None, end=None):
        left = match.start() if start is None else start
        right = match.end() if end is None else end
        key = (field, left, right)
        if key in seen:
            return
        seen.add(key)
        quote = source_text[left:right]
        fact_id = document_id + ":R" + _hash(f"{digest}|{field}|{left}|{right}|{quote}")[:20]
        fields[field].append({"evidence_id": fact_id, "quote": quote, "char_start": left, "char_end": right,
                              "source_sha256": digest, "value": value,
                              "normalization_policy": policy, "match_type": field,
                              "confidence_kind": "rule_pattern_not_semantic_validation",
                              "verified": True, "needs_review": needs_review})

    for match in _ISO_DATE.finditer(source_text):
        try:
            parsed = date(int(match["year"]), int(match["month"]), int(match["day"]))
        except ValueError:
            continue
        add("dates", match, {"iso_date": parsed.isoformat(), "ambiguous": False, "timezone": None},
            "calendar_valid_ISO_yyyy_mm_dd_no_timezone_inference")
    for match in _SLASH_DATE.finditer(source_text):
        day, month, year = (int(match[key]) for key in ("day", "month", "year"))
        try:
            parsed = date(year, month, day)
        except ValueError:
            continue
        ambiguous = day <= 12 and month <= 12 and day != month
        alternatives = [parsed.isoformat()]
        if ambiguous:
            alternatives.append(date(year, day, month).isoformat())
        add("dates", match, {"iso_date": None if ambiguous else parsed.isoformat(), "ambiguous": ambiguous,
                              "candidate_iso_dates": alternatives, "timezone": None},
            "slash_date_day_month_order_unconfirmed_when_both_le_12", needs_review=ambiguous)
    for match in _BUDGET.finditer(source_text):
        if (re.match(r"\s*(?:million|billion|thousand|triệu|tỷ|nghìn|ngàn|[kmb])\b", source_text[match.end():], re.I)
                or re.match(r"\s*[-–]\s*\d", source_text[match.end():])
                or re.search(r"[-−]\s*$", source_text[max(0, match.start()-3):match.start()])):
            # Unsupported scales, negative amounts and ranges must not be
            # presented as a partially parsed positive exact budget.
            continue
        raw_currency = match["prefix"] or match["suffix"]
        raw_number = match["amount1"] or match["amount2"]
        amount, alternatives = _amount(raw_number)
        currency = raw_currency.upper() if raw_currency != "$" else "UNSPECIFIED_DOLLAR"
        ambiguous = bool(alternatives) or raw_currency == "$"
        add("budgets", match, {"amount": amount, "currency": currency,
                                "numeric_ambiguous": bool(alternatives), "candidate_amounts": alternatives,
                                "currency_ambiguous": raw_currency == "$"},
            "explicit_currency_only_separator_ambiguity_preserved_no_dollar_country_inference",
            needs_review=ambiguous)
    for match in _DURATION.finditer(source_text):
        number = format(Decimal(match["number"].replace(",", ".")), "f")
        add("durations", match, {"quantity": number, "unit": _UNITS[match["unit"].casefold()]},
            "literal_numeric_duration_no_calendar_conversion_or_start_end_inference")
    for match in _PERSONNEL.finditer(source_text):
        add("personnel", match, {"quantity": int(match["number"]), "role_mention": match["unit"]},
            "literal_numeric_staff_mention_not_assignment_or_availability", needs_review=True)
    for match in _IDENTIFIER.finditer(source_text):
        identifier = match["identifier"].rstrip(".,/")
        if not identifier or (not any(char.isdigit() for char in identifier)
                              and not any(char in "-_" for char in identifier)
                              and not identifier.isupper()):
            continue
        end = match.end("identifier") - (len(match["identifier"]) - len(identifier))
        add("identifiers", match, identifier, "literal_explicit_identifier_label_case_preserved", end=end)
    for match in _MILESTONE.finditer(source_text):
        quote = match.group().rstrip()
        add("milestones", match, quote, "literal_milestone_labeled_text_not_confirmed_schedule", needs_review=True,
            end=match.start() + len(quote))
    for match in _TECH_PATTERN.finditer(source_text):
        add("technologies", match, _CANONICAL_TECH[match.group().casefold()],
            "case_insensitive_fixed_technology_lexicon_not_requirement_or_compatibility", needs_review=True)
    for items in fields.values():
        items.sort(key=lambda item: (item["char_start"], item["char_end"], item["evidence_id"]))
    return {"document_id": document_id, "source_sha256": digest, "method": METHOD,
            "offset_unit": "unicode_codepoint", "fields": fields,
            "human_review_complete": False,
            "warnings": [{"code": "pattern_coverage_limited", "status": "needs_review", "severity": "review",
                          "message": "Rules detect literal patterns only; undetected facts may still occur in the source.",
                          "evidence_ids": []}]}


def resource_warnings(record: dict, plan: dict, facts: dict) -> list[dict]:
    """Report detection coverage and unconfirmed plan associations without invention."""
    if not all(isinstance(value, dict) for value in (record, plan, facts)):
        raise ValueError("record, plan and facts must be mappings")
    document_id = facts.get("document_id")
    digest = facts.get("source_sha256")
    if (not isinstance(document_id, str) or not document_id or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or record.get("document_id") != document_id or plan.get("document_id") != document_id
            or record.get("source_sha256") != digest or plan.get("source_sha256") != digest
            or not isinstance(facts.get("fields"), dict)):
        raise ValueError("Warnings need source-consistent record, plan and rule facts")
    fields = facts["fields"]
    warnings = []
    for field, label in (("budgets", "budget/currency"), ("dates", "calendar date"),
                         ("personnel", "numeric personnel"), ("technologies", "known technology")):
        items = fields.get(field, [])
        if not isinstance(items, list):
            raise ValueError("Rule fact fields must be lists")
        if not items:
            warnings.append({"code": field + "_not_detected", "severity": "review", "status": "not_detected",
                             "message": "No explicit " + label + " mention was detected by the configured rules; "
                             "this is not proof the document lacks it. Request human confirmation before planning.",
                             "evidence_ids": []})
        else:
            warnings.append({"code": field + "_mentions_need_review", "severity": "review", "status": "needs_review",
                             "message": "Literal " + label + " mentions were detected; allocation, applicability and "
                             "completeness have not been confirmed by a person.",
                             "evidence_ids": [item["evidence_id"] for item in items]})
    steps = plan.get("steps", [])
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError("Plan steps must be objects in a list")
    associated = [eid for step in steps for entry in step.get("required_resources", [])
                  for eid in entry.get("evidence_ids", [])]
    if associated:
        warnings.append({"code": "resource_associations_need_review", "severity": "review", "status": "needs_review",
                         "message": "Task/resource associations are LLM suggestions; source mentions do not establish "
                         "that the resource is necessary, available, assigned or sufficient.",
                         "evidence_ids": sorted(set(associated))})
    if plan.get("dependency_candidates"):
        warnings.append({"code": "dependency_candidates_need_review", "severity": "review", "status": "needs_review",
                         "message": "Dependency candidates are suggestions; do not publish them as a confirmed schedule.",
                         "evidence_ids": []})
    return warnings
