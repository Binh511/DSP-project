from pathlib import Path
from collections import Counter
import csv
import hashlib
import re
import shutil


# ============================================================
# PSAI CLEANING PIPELINE v6
#
# Input:
#   D:\fall26\DSP_PROJECT\data_processed\PSAI001\PSAI001.md
#
# Output:
#   D:\fall26\DSP_PROJECT\data_cleaned\PSAI001\PSAI001.md
#   D:\fall26\DSP_PROJECT\data_cleaned\PSAI001\images\...
#
# v6 changes vs v5:
#   1. NEW: _has_diagram_content() — loose diagram continuation check.
#   2. NEW: merge_adjacent_diagram_blocks() — merges diagram blocks
#      that are separated by blank lines (fixes Architecture diagram
#      being split into 2 blocks).
#   3. wrap_diagram_block() drops blank lines inside the block.
#   4. process_blocks() merges diagram blocks before handling.
#
# IMPORTANT:
#   - This script DOES NOT do BIO labeling.
#   - This script DOES NOT lowercase text.
#   - Technical terms are preserved.
#   - Images are copied but not deleted.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR = BASE_DIR / "data_cleaned"

REPORT_FILE = BASE_DIR / "cleaning_report.csv"

RESET_OUTPUT = True
COPY_IMAGES = True


# ============================================================
# 1. MARKDOWN / OCR NORMALIZATION
# ============================================================

def normalize_basic(text: str) -> str:
    """Normalize OCR/Markdown artifacts without changing case."""

    replacements = {
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",

        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\\([_*[\](){}#+.!|`~>-])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def strip_markdown_decoration(text: str) -> str:
    """
    Remove formatting markers while preserving their text.
    NOTE: This REMOVES headings. Do not use on heading lines.
    """

    text = text.strip()

    text = re.sub(r"^\s*#{1,6}\s+", "", text)

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)

    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = re.sub(r"\\([_*[\](){}#+.!|`~>-])", r"\1", text)

    return text.strip()


def strip_markdown_links(text: str) -> str:
    text = re.sub(
        r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)",
        r"\1",
        text
    )
    return text


ICON_RE = re.compile(
    r":(?:page_facing_up|memo|pencil2?|link|pushpin):"
    r"|(?<!\w)(?:page_facing_up|pushpin|pencil2)(?!\w)",
    re.IGNORECASE,
)


def clean_inline_text(text: str) -> str:
    text = normalize_basic(text)
    text = strip_markdown_links(text)
    text = strip_markdown_decoration(text)
    text = ICON_RE.sub("", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return re.sub(r"\s+", " ", text).strip()


def normalize_heading(line: str) -> str:
    """
    Keep leading # markers, strip bold/italic/links INSIDE heading text.
    '# **Project Description**' -> '# Project Description'
    '###   Security'            -> '### Security'
    """
    s = normalize_basic(line).rstrip()

    m = re.match(r"^(#{1,6})\s*(.*)$", s)
    if not m:
        return s

    hashes, rest = m.group(1), m.group(2)
    rest = strip_markdown_links(rest)
    rest = strip_markdown_decoration(rest)
    rest = re.sub(r"\s+", " ", rest).strip()

    return f"{hashes} {rest}".strip()


# ============================================================
# 2. METADATA DETECTION
# ============================================================

FIELD_LABELS = [
    "student name", "student id", "student number",
    "full name", "author", "authors", "writer",
    "prepared by", "submitted by", "proposer",
    "team name", "team", "organization", "organisation",

    "university", "school", "faculty", "department",
    "instructor", "lecturer", "teacher", "advisor",
    "supervisor", "course", "course name", "course code",
    "class", "section",

    "date", "submission date", "submitted date",
    "created date", "modified date", "updated date",
    "version", "document version",

    "report no", "report number", "report id",
    "document no", "document number", "document id",
    "file id", "file name", "file number",
    "project id", "project number", "project code",

    "proposal category", "project category", "grant category",
    "project status", "proposal status", "status",
    "level", "grant status",

    "payment address", "wallet address", "payment",
    "bank account", "account number",
    "transaction id", "transaction hash", "txid", "tx hash",
]

PROJECT_NAME_LABELS = [
    "name of project", "project name", "project title",
    "proposal title", "document title", "paper title",
]


def is_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s+", line))


def is_list_item(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line))


def normalize_for_metadata(line: str) -> str:
    s = normalize_basic(line)

    s = re.sub(r"^\s*[-*+]\s+", "", s)
    s = re.sub(r"^\s*#{1,6}\s*", "", s)

    s = re.sub(r"^\*{1,3}", "", s)
    s = re.sub(r"\*{1,3}$", "", s)

    s = re.sub(r"^_{1,3}", "", s)
    s = re.sub(r"_{1,3}$", "", s)

    s = strip_markdown_links(s)
    s = strip_markdown_decoration(s)

    return re.sub(r"\s+", " ", s).strip()


def is_document_id(line: str) -> bool:
    s = normalize_for_metadata(line)
    patterns = [
        r"PSAI[_\-]?\d+",
        r"FILE[_\-]?\d+",
        r"FIL[_\-]?\d+",
        r"DOC[_\-]?\d+",
    ]
    return any(re.fullmatch(p, s, re.IGNORECASE) for p in patterns)


def is_page_marker(line: str) -> bool:
    s = normalize_for_metadata(line)
    patterns = [
        r"page\s+\d+",
        r"page\s+\d+\s+of\s+\d+",
        r"\d+\s*/\s*\d+",
    ]
    return any(re.fullmatch(p, s, re.IGNORECASE) for p in patterns)


def is_standalone_date(line: str) -> bool:
    s = normalize_for_metadata(line)
    patterns = [
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        r"[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}",
    ]
    return any(re.fullmatch(p, s, re.IGNORECASE) for p in patterns)


def is_field_metadata(line: str) -> bool:
    if is_heading(line):
        return False

    s = normalize_for_metadata(line)
    s = re.sub(r"^\d+\s+", "", s).strip()
    s = strip_markdown_links(s)

    for label in FIELD_LABELS:
        pattern = r"^" + re.escape(label) + r"\s*(?:[:\-]\s*.*)?$"
        if re.match(pattern, s, re.IGNORECASE):
            return True

    if re.search(r"\bstatus\s*:", s, re.IGNORECASE):
        return True

    if (re.search(r"\blevel\b", s, re.IGNORECASE)
            and re.search(r"\bstatus\b", s, re.IGNORECASE)):
        return True

    return False


def is_project_name_field(line: str) -> bool:
    if is_heading(line):
        return False

    s = normalize_for_metadata(line)
    for label in PROJECT_NAME_LABELS:
        if re.match(rf"^{re.escape(label)}\s*[:\-]", s, re.IGNORECASE):
            return True
    return False


def is_blockchain_identifier(line: str) -> bool:
    s = normalize_for_metadata(line)
    if re.fullmatch(r"0x[a-fA-F0-9]{40,66}", s):
        return True
    if re.fullmatch(r"[A-Za-z0-9]{40,100}", s):
        return True
    return False


def is_admin_link_line(line: str) -> bool:
    s = normalize_for_metadata(line)
    if (re.search(r"\bstatus\b", s, re.IGNORECASE)
            and "github.com" in s.lower()):
        return True
    if (re.search(r"\blevel\b", s, re.IGNORECASE)
            and "github.com" in s.lower()):
        return True
    return False


def is_metadata(line: str) -> bool:
    if not line.strip():
        return False
    if is_document_id(line):
        return True
    if is_page_marker(line):
        return True
    if is_standalone_date(line):
        return True
    if is_field_metadata(line):
        return True
    if is_project_name_field(line):
        return True
    if is_blockchain_identifier(line):
        return True
    if is_admin_link_line(line):
        return True
    return False


# ============================================================
# 3. METADATA BLOCK HANDLING
# ============================================================

def is_metadata_label_only(line: str) -> bool:
    s = normalize_for_metadata(line)

    labels = [
        "payment address", "wallet address", "team name",
        "author", "proposer", "student name", "student id",
        "status", "level", "version",
        "project id", "document id", "file id",
    ]

    for label in labels:
        if re.fullmatch(rf"{re.escape(label)}\s*[:\-]?", s, re.IGNORECASE):
            return True
    return False


def is_metadata_continuation(line: str, previous_metadata_label: str | None) -> bool:
    if not previous_metadata_label:
        return False

    s = normalize_for_metadata(line)
    if not s:
        return False

    if previous_metadata_label in {
        "payment address", "wallet address",
        "transaction id", "transaction hash", "txid", "tx hash",
    }:
        if re.fullmatch(r"[A-Za-z0-9:_\-]{20,120}", s):
            return True

    if previous_metadata_label in {
        "team name", "author", "proposer",
        "student name", "student id",
    }:
        if (re.match(r"^#{1,6}\s+", line)
                or re.match(r"^\s*[-*+]\s+", line)):
            return False
        if len(s) <= 150:
            return True

    return False


def get_metadata_label(line: str) -> str | None:
    s = normalize_for_metadata(line)
    for label in FIELD_LABELS:
        if re.fullmatch(rf"{re.escape(label)}\s*[:\-]?", s, re.IGNORECASE):
            return label.lower()
    return None


# ============================================================
# 4. TEMPLATE / FORM INSTRUCTIONS
# ============================================================

def is_form_instruction(line: str) -> bool:
    s = normalize_for_metadata(line)
    patterns = [
        r"^to submit a proposal\b",
        r"^please title your\b",
        r"^choose one of\b",
        r"^do you agree to\b",
        r"^please provide\b",
        r"^please enter\b",
        r"^select one of\b",
    ]
    return any(re.match(p, s, re.IGNORECASE) for p in patterns)


# ============================================================
# 5. DUPLICATES
# ============================================================

def comparison_key(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def remove_consecutive_duplicates(lines):
    output = []
    removed = 0
    previous_key = None

    for line in lines:
        s = line.strip()

        if not s:
            output.append("")
            previous_key = None
            continue

        if is_image_line(line):
            output.append(line)
            previous_key = None
            continue

        key = comparison_key(s)

        if key and key == previous_key and len(key) > 8:
            removed += 1
            continue

        output.append(line)
        previous_key = key

    return output, removed


def remove_duplicate_paragraphs(lines):
    paragraphs = []
    current = []

    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                paragraphs.append(current)
                current = []

    if current:
        paragraphs.append(current)

    seen = set()
    output_paragraphs = []
    removed = 0

    for paragraph in paragraphs:
        text = " ".join(paragraph)
        key = comparison_key(text)

        if len(key) < 80:
            output_paragraphs.append(paragraph)
            continue

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        output_paragraphs.append(paragraph)

    output = []
    for paragraph in output_paragraphs:
        output.extend(paragraph)
        output.append("")

    return output, removed


# ============================================================
# 6. REPEATED HEADER / FOOTER
# ============================================================

def find_repeated_headers(lines):
    counter = Counter()

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if is_image_line(line):
            continue
        if "|" in s:
            continue
        if is_list_item(line):
            continue
        if len(s) < 12 or len(s) > 150:
            continue
        if s.startswith("#"):
            continue

        key = comparison_key(s)
        if key:
            counter[key] += 1

    return {key for key, count in counter.items() if count >= 5}


def remove_repeated_headers(lines):
    repeated = find_repeated_headers(lines)
    output = []
    removed = 0

    for line in lines:
        s = line.strip()
        if not s:
            output.append(line)
            continue

        if is_image_line(line) or "|" in s or is_list_item(line):
            output.append(line)
            continue

        key = comparison_key(s)
        if key in repeated:
            removed += 1
            continue

        output.append(line)

    return output, removed


# ============================================================
# 7. IMAGE REFERENCES
# ============================================================

def is_image_line(line: str) -> bool:
    return bool(re.match(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", line.strip()))


# ============================================================
# 8. TABLE HANDLING
# ============================================================

def table_column_count(line: str) -> int:
    s = line.strip()
    if "|" not in s:
        return 0
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return len(s.split("|"))


def is_markdown_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    if not cells:
        return False
    return all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


def table_quality(block):
    table_lines = [line for line in block if "|" in line]

    if len(table_lines) < 2:
        return "NONE"

    has_separator = any(is_markdown_separator(line) for line in table_lines)

    counts = [
        table_column_count(line)
        for line in table_lines
        if table_column_count(line) > 0
    ]

    if has_separator and len(counts) >= 2:
        return "PASS"

    if counts and len(set(counts)) == 1:
        return "PASS"

    return "REVIEW"


def clean_valid_table(block):
    output = []

    for line in block:
        line = normalize_basic(line)
        line = re.sub(r"\s*\|\s*", " | ", line)

        if line.startswith(" | "):
            line = "|" + line[3:]

        output.append(line.strip())

    return output


def reconstruct_table(block):
    """
    Rebuild a broken table into a valid markdown table.
    Returns list of markdown lines, or None if not possible.
    """
    rows = []
    for line in block:
        s = line.strip()
        if "|" not in s:
            continue
        if is_markdown_separator(s):
            continue
        s = s.strip("|")
        cells = [clean_inline_text(c) for c in s.split("|")]
        cells = [c for c in cells if c != ""]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return None

    ncol = max(len(r) for r in rows)
    if ncol < 2:
        return None

    for r in rows:
        while len(r) < ncol:
            r.append("")

    output = []
    header = rows[0]
    output.append("| " + " | ".join(header) + " |")
    output.append("|" + "|".join(["---"] * ncol) + "|")

    for r in rows[1:]:
        output.append("| " + " | ".join(r) + " |")

    return output


def flatten_broken_table(block):
    """Fallback: flatten broken table into one line of text."""
    parts = []

    for line in block:
        s = normalize_basic(line).strip()
        if not s:
            continue
        if is_markdown_separator(s):
            continue

        s = s.strip("|").strip()
        if not s:
            continue

        cells = s.split("|")
        for cell in cells:
            cell = clean_inline_text(cell)
            if not cell:
                continue
            if re.fullmatch(r":?-{3,}:?", cell):
                continue
            parts.append(cell)

    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def handle_broken_table(block):
    """
    Try to reconstruct. If column count is inconsistent
    across rows (> 2 mismatches), fall back to flatten.
    """
    rebuilt = reconstruct_table(block)

    if rebuilt:
        header_cols = rebuilt[0].count("|") - 1
        bad_rows = sum(
            1 for line in rebuilt[2:]
            if line.count("|") - 1 != header_cols
        )
        if bad_rows == 0:
            return rebuilt

    flattened = flatten_broken_table(block)
    return [flattened] if flattened else []


# ============================================================
# 9. DIAGRAM HANDLING
# ============================================================

DIAGRAM_CHARS = re.compile(
    r"[─│┌┐└┘├┤┬┴┼╔╗╚╝║═╠╣╦╩╬→←↑↓↔⟶⟵]{2,}"
)


def is_diagram_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    if len(DIAGRAM_CHARS.findall(s)) >= 2:
        return True

    if re.fullmatch(r"[-=_|+ ]{5,}", s):
        return True

    if re.search(r"[-=]{3,}\s+[-=]{3,}", s):
        return True

    return False


def _has_diagram_content(block) -> bool:
    """
    Loose check for a diagram continuation block.
    Fires if any line looks like part of an ASCII diagram.
    """
    if not block:
        return False

    for line in block:
        s = line.strip()
        if not s:
            continue

        if len(DIAGRAM_CHARS.findall(s)) >= 2:
            return True

        if re.fullmatch(r"[-=_|+ ]{5,}", s):
            return True

        # Pipe-heavy + long dash sequence -> ASCII art
        if "|" in s and re.search(r"[-=]{5,}", s):
            return True

    return False


def block_is_diagram(block) -> bool:
    if len(block) < 1:
        return False

    diagram_lines = sum(is_diagram_line(l) for l in block)
    if diagram_lines == 0:
        return False

    has_separator = any(is_markdown_separator(l) for l in block)
    if has_separator and diagram_lines / len(block) < 0.5:
        return False

    return diagram_lines / len(block) >= 0.3


def merge_adjacent_diagram_blocks(blocks):
    """
    If a block is a diagram and the next block is diagram-like,
    merge them so the whole diagram is wrapped into one code block.
    """
    merged = []
    i = 0

    while i < len(blocks):
        current = list(blocks[i])
        is_diagram = block_is_diagram(current) or _has_diagram_content(current)

        j = i
        while is_diagram and j + 1 < len(blocks):
            nxt = blocks[j + 1]
            if _has_diagram_content(nxt):
                current = current + [""] + list(nxt)
                j += 1
            else:
                break

        merged.append(current)
        i = j + 1

    return merged


def wrap_diagram_block(block):
    """Wrap a diagram block in ```text```. Blank lines inside are dropped."""
    output = ["```text"]
    for line in block:
        if line.strip() == "":
            continue
        output.append(line.rstrip())
    output.append("```")
    return output


# ============================================================
# 10. DATE-MILESTONE SPLITTING (Roadmap)
# ============================================================

DATE_MILESTONE_PATTERN = re.compile(
    r"(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{4}:)",
    re.IGNORECASE,
)


def split_date_milestones(text: str) -> list[str]:
    """
    '01 sep 2021: A 07 sep 2021: B'
    -> ['- 01 sep 2021: A', '- 07 sep 2021: B']
    """
    matches = list(DATE_MILESTONE_PATTERN.finditer(text))
    if len(matches) < 2:
        return [text]

    prefix = text[: matches[0].start()].strip()
    output = []
    if prefix:
        output.append(prefix)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = text[start:end].strip()
        if part:
            output.append("- " + part)

    return output


# ============================================================
# 11. PARAGRAPH / BULLET REPAIR
# ============================================================

def expand_inline_bullets(text: str) -> list[str]:
    """
    ': * a, * b, * c'  ->  [':', '- a', '- b', '- c']
    Guarded against diagram-like text.
    """
    if len(DIAGRAM_CHARS.findall(text)) >= 2:
        return [text]
    if re.search(r"[-=_]{4,}", text):
        return [text]

    if text.count("* ") + text.count("- ") < 2:
        return [text]

    m = re.match(r"^(.*?)(?=\s*[*\-]\s)(.*)$", text)
    if not m:
        return [text]

    prefix = m.group(1).rstrip(" :,-")
    items_str = m.group(2)

    items = re.split(r"\s*[*\-]\s+", items_str)
    items = [clean_inline_text(i) for i in items if i.strip()]

    if len(items) < 2:
        return [text]

    output = []
    if prefix:
        output.append(prefix + ":")
    for item in items:
        output.append("- " + item)

    return output


def repair_paragraph_block(block):
    if not block:
        return []

    output = []
    current = ""

    def flush_current():
        nonlocal current
        if current:
            cleaned = clean_inline_text(current)
            for chunk in expand_inline_bullets(cleaned):
                output.extend(split_date_milestones(chunk))
            current = ""

    if block_is_diagram(block):
        return wrap_diagram_block(block)

    for raw_line in block:
        line = raw_line.strip()

        if not line:
            continue

        if is_diagram_line(line) and not is_list_item(line):
            flush_current()
            output.append("```text")
            output.append(line)
            output.append("```")
            continue

        if is_image_line(line):
            flush_current()
            output.append(line)
            continue

        if is_heading(line):
            flush_current()
            output.append(normalize_heading(line))
            continue

        if is_list_item(line):
            flush_current()
            output.append(clean_inline_text(line))
            continue

        line = clean_inline_text(line)
        if not line:
            continue

        if not current:
            current = line
        else:
            if current.endswith("-") and not current.endswith("--"):
                current = current[:-1] + line
            else:
                current += " " + line

    flush_current()

    return output


# ============================================================
# 12. BLOCK PROCESSOR
# ============================================================

def process_blocks(lines):
    blocks = []
    current = []

    for line in lines:
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)

    if current:
        blocks.append(current)

    # NEW: merge diagram blocks separated by blank lines.
    blocks = merge_adjacent_diagram_blocks(blocks)

    output = []
    table_count = 0
    table_pass = 0
    table_review = 0

    for block in blocks:
        # 1. Diagram?
        if block_is_diagram(block):
            output.extend(wrap_diagram_block(block))
            output.append("")
            continue

        # 2. Table?
        table_lines = sum("|" in line for line in block)

        if table_lines >= 2:
            table_count += 1
            quality = table_quality(block)

            if quality == "PASS":
                table_pass += 1
                output.extend(clean_valid_table(block))
            else:
                table_review += 1
                output.extend(handle_broken_table(block))
        else:
            output.extend(repair_paragraph_block(block))

        output.append("")

    return output, table_count, table_pass, table_review


# ============================================================
# 13. MAIN CLEANING FUNCTION
# ============================================================

def clean_document(raw_text: str):

    stats = {
        "metadata_removed": 0,
        "template_removed": 0,
        "duplicate_lines_removed": 0,
        "duplicate_paragraphs_removed": 0,
        "repeated_headers_removed": 0,
    }

    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw_text.split("\n")

    # LAYER 1 — NORMALIZE
    lines = [normalize_basic(line) for line in lines]

    # LAYER 2 — METADATA
    filtered = []
    pending_metadata_label = None

    for line in lines:
        s = line.strip()

        if not s:
            filtered.append("")
            pending_metadata_label = None
            continue

        if is_metadata_continuation(line, pending_metadata_label):
            stats["metadata_removed"] += 1
            pending_metadata_label = None
            continue

        if is_metadata(line):
            stats["metadata_removed"] += 1
            pending_metadata_label = get_metadata_label(line)
            continue

        if is_form_instruction(line):
            stats["template_removed"] += 1
            pending_metadata_label = None
            continue

        if is_metadata_label_only(line):
            stats["metadata_removed"] += 1
            pending_metadata_label = get_metadata_label(line)
            continue

        pending_metadata_label = None
        filtered.append(line)

    lines = filtered

    # LAYER 3 — REPEATED HEADER / FOOTER
    lines, removed = remove_repeated_headers(lines)
    stats["repeated_headers_removed"] = removed

    # LAYER 4 — CONSECUTIVE DUPLICATES
    lines, removed = remove_consecutive_duplicates(lines)
    stats["duplicate_lines_removed"] = removed

    # LAYER 5 — TABLE / DIAGRAM / PARAGRAPH
    (lines, table_count, table_pass, table_review) = process_blocks(lines)

    # LAYER 6 — DUPLICATE PARAGRAPHS
    lines, removed = remove_duplicate_paragraphs(lines)
    stats["duplicate_paragraphs_removed"] = removed

    # LAYER 7 — FINAL NORMALIZATION (keeps headings, code blocks)
    final_lines = []
    previous_blank = False
    inside_code_block = False

    for line in lines:
        line = line.rstrip()

        if not line.strip():
            if not previous_blank:
                final_lines.append("")
            previous_blank = True
            continue

        if line.strip().startswith("```"):
            final_lines.append(line.strip())
            inside_code_block = not inside_code_block
            previous_blank = False
            continue

        if inside_code_block:
            final_lines.append(line)
            previous_blank = False
            continue

        if is_image_line(line):
            final_lines.append(line.strip())
            previous_blank = False
            continue

        if is_heading(line):
            final_lines.append(normalize_heading(line))
            previous_blank = False
            continue

        if "|" in line:
            final_lines.append(line.strip())
            previous_blank = False
            continue

        line = clean_inline_text(line)
        if line:
            final_lines.append(line)
            previous_blank = False

    while final_lines and not final_lines[-1].strip():
        final_lines.pop()

    clean_text = "\n".join(final_lines)

    return clean_text, stats, table_count, table_pass, table_review


# ============================================================
# 14. COPY IMAGES
# ============================================================

def copy_images(source_folder: Path, target_folder: Path) -> int:
    source_images = source_folder / "images"
    if not source_images.exists():
        return 0

    target_images = target_folder / "images"

    if target_images.exists():
        shutil.rmtree(target_images)

    shutil.copytree(source_images, target_images)

    return sum(1 for p in target_images.rglob("*") if p.is_file())


# ============================================================
# 15. PIPELINE VALIDATION + QUALITY CHECK
# ============================================================

def validate_for_pipeline(clean_text: str) -> dict:
    issues = []

    headings = re.findall(r"^#{1,6}\s+\S", clean_text, re.MULTILINE)
    if len(headings) < 2:
        issues.append("no_headings")

    has_valid_table = bool(re.search(
        r"^\|.+\|\s*\n\|[\s\-:|]+\|\s*\n",
        clean_text, re.MULTILINE
    ))

    # Unwrapped diagram?
    for block in re.split(r"\n\s*\n", clean_text):
        block_lines = block.split("\n")
        if block_is_diagram(block_lines) and "```" not in block:
            issues.append("unwrapped_diagram")
            break

    if re.search(r"\* .+?, \* ", clean_text):
        issues.append("inline_bullets")

    if re.search(r"^#{1,6}\s*\*\*", clean_text, re.MULTILINE):
        issues.append("bold_in_heading")

    # Fence balance
    if clean_text.count("```") % 2 != 0:
        issues.append("unbalanced_code_fence")

    return {
        "headings": len(headings),
        "has_valid_table": has_valid_table,
        "issues": issues,
    }


def quality_check(raw: str, clean: str, stats: dict) -> str:
    if not clean.strip():
        return "FAIL"

    if len(clean) < 100:
        return "REVIEW"

    if len(raw) > 0:
        reduction = 1 - len(clean) / len(raw)
        if reduction > 0.60:
            return "REVIEW"

    if (stats["metadata_removed"] > 50
            and len(clean) < len(raw) * 0.5):
        return "REVIEW"

    return "PASS"


# ============================================================
# 16. PROCESS ONE DOCUMENT
# ============================================================

def process_document(md_file: Path):
    doc_id = md_file.parent.name

    raw = md_file.read_text(encoding="utf-8", errors="replace")

    (clean, stats, table_count, table_pass, table_review) = clean_document(raw)

    output_folder = OUTPUT_DIR / doc_id
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / md_file.name
    output_file.write_text(clean, encoding="utf-8")

    image_count = 0
    if COPY_IMAGES:
        image_count = copy_images(md_file.parent, output_folder)

    raw_chars = len(raw)
    clean_chars = len(clean)
    removed_chars = raw_chars - clean_chars
    removed_percent = (removed_chars / raw_chars * 100) if raw_chars else 0

    quality = quality_check(raw, clean, stats)
    pipeline_check = validate_for_pipeline(clean)

    if table_review > 0 and quality == "PASS":
        quality = "REVIEW"

    if pipeline_check["issues"] and quality == "PASS":
        quality = "REVIEW"

    result = {
        "ID": doc_id,
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "removed_chars": removed_chars,
        "removed_percent": round(removed_percent, 2),

        "metadata_removed": stats["metadata_removed"],
        "template_removed": stats["template_removed"],
        "duplicate_lines_removed": stats["duplicate_lines_removed"],
        "duplicate_paragraphs_removed": stats["duplicate_paragraphs_removed"],
        "repeated_headers_removed": stats["repeated_headers_removed"],

        "table_count": table_count,
        "table_pass": table_pass,
        "table_review": table_review,

        "headings": pipeline_check["headings"],
        "has_valid_table": int(pipeline_check["has_valid_table"]),
        "pipeline_issues": ",".join(pipeline_check["issues"]),

        "images": image_count,
        "quality": quality,
        "output_file": str(output_file),
        "status": "OK",
    }

    print(
        f"[{doc_id}] "
        f"RAW={raw_chars:,} "
        f"CLEAN={clean_chars:,} "
        f"META={stats['metadata_removed']} "
        f"DUP={stats['duplicate_lines_removed'] + stats['duplicate_paragraphs_removed']} "
        f"TABLE={table_count}/{table_pass}/{table_review} "
        f"HEAD={pipeline_check['headings']} "
        f"ISSUES={pipeline_check['issues']} "
        f"QUALITY={quality}"
    )

    return result


# ============================================================
# 17. DUPLICATE DOCUMENTS
# ============================================================

def find_duplicate_documents(results):
    hashes = {}
    duplicates = []

    for result in results:
        if result.get("status") != "OK":
            continue

        path = Path(result["output_file"])
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        normalized = comparison_key(text)

        file_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if file_hash in hashes:
            duplicates.append({
                "ID": result["ID"],
                "same_as": hashes[file_hash],
            })
        else:
            hashes[file_hash] = result["ID"]

    return duplicates


# ============================================================
# 18. REPORT
# ============================================================

def save_report(results, duplicates):
    fields = [
        "ID",
        "raw_chars", "clean_chars", "removed_chars", "removed_percent",
        "metadata_removed", "template_removed",
        "duplicate_lines_removed", "duplicate_paragraphs_removed",
        "repeated_headers_removed",
        "table_count", "table_pass", "table_review",
        "headings", "has_valid_table", "pipeline_issues",
        "images", "quality", "output_file", "status",
    ]

    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})

        for duplicate in duplicates:
            writer.writerow({
                "ID": duplicate["ID"],
                "status": "DUPLICATE",
                "output_file": f"Same as {duplicate['same_as']}",
            })


# ============================================================
# 19. MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PSAI CLEANING PIPELINE v6")
    print("=" * 75)

    print(f"INPUT : {INPUT_DIR}")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print()

    if not INPUT_DIR.exists():
        print("ERROR: input folder does not exist.")
        return

    if RESET_OUTPUT and OUTPUT_DIR.exists():
        print("Removing previous cleaned output...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.rglob("*.md"))

    if not files:
        print("ERROR: no .md files found.")
        return

    print(f"Found {len(files)} Markdown documents.")
    print()

    results = []

    for md_file in files:
        try:
            result = process_document(md_file)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {md_file}: {e}")
            results.append({
                "ID": md_file.parent.name,
                "status": "ERROR",
                "quality": "FAIL",
                "output_file": "",
            })

    duplicates = find_duplicate_documents(results)
    save_report(results, duplicates)

    print()
    print("=" * 75)
    print("CLEANING FINISHED")
    print("=" * 75)

    print(f"Documents : {len(results)}")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Report    : {REPORT_FILE}")

    print()
    print("IMPORTANT: Review rows with QUALITY=REVIEW before chunking/tokenization.")


if __name__ == "__main__":
    main()