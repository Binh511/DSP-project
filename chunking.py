from pathlib import Path
import json
import re
import hashlib
from typing import Iterator

import tiktoken


# ============================================================
# PSAI CHUNKING PIPELINE v2
#
# Input:
#   <thư mục chứa file>/data_cleaned/<doc_id>/<doc_id>.md
#
# Output:
#   <thư mục chứa file>/data_chunked/<doc_id>/<doc_id>.jsonl
#   <thư mục chứa file>/data_chunked/<doc_id>/<doc_id>.stats.json
#
# Strategy:
#   - Split by markdown heading (#, ##, ###, ####)
#   - Keep heading_path in each chunk (prefix cho MỌI chunk)
#   - Max 512 tokens, overlap 64
#   - Tables: gom nhiều dòng vào một chunk (~350 token)
#   - Diagrams kept whole
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data_cleaned"
OUTPUT_DIR = BASE_DIR / "data_chunked"

RESET_OUTPUT = True

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
TABLE_GROUP_TOKENS = 350
MIN_CHUNK_TOKENS = 20          # bỏ chunk rác
TOKENIZER_NAME = "cl100k_base"


# ============================================================
# TOKENIZER
# ============================================================

_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))


# ============================================================
# MARKDOWN PARSING
# ============================================================

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

CODE_FENCE_RE = re.compile(r"^```")


def parse_blocks(text: str):
    """
    Split markdown into blocks with heading context.
    Returns list of dicts:
      {level, heading, heading_path, content, block_type}
    block_type in {"section", "table", "diagram"}
    """
    lines = text.split("\n")

    blocks = []
    heading_stack = []      # list of (level, heading)
    current_heading = None
    current_lines = []
    inside_fence = False
    fence_lines = []

    def flush_section():
        nonlocal current_lines
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if content:
            path = [h for _, h in heading_stack]
            blocks.append({
                "level": current_heading[0] if current_heading else 0,
                "heading": current_heading[1] if current_heading else "",
                "heading_path": path,
                "content": content,
                "block_type": "section",
            })
        current_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # --- code fence ---
        if CODE_FENCE_RE.match(s):
            if not inside_fence:
                flush_section()
                inside_fence = True
                fence_lines = [line]
            else:
                fence_lines.append(line)
                inside_fence = False
                path = [h for _, h in heading_stack]
                blocks.append({
                    "level": current_heading[0] if current_heading else 0,
                    "heading": current_heading[1] if current_heading else "",
                    "heading_path": path,
                    "content": "\n".join(fence_lines).strip(),
                    "block_type": "diagram",
                })
                fence_lines = []
            i += 1
            continue

        if inside_fence:
            fence_lines.append(line)
            i += 1
            continue

        # --- heading ---
        m = HEADING_RE.match(s)
        if m:
            flush_section()
            level = len(m.group(1))
            heading = m.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            current_heading = (level, heading)
            i += 1
            continue

        # --- table ---
        if s.startswith("|") and s.endswith("|"):
            flush_section()
            table_lines = []
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("|") and t.endswith("|"):
                    table_lines.append(t)
                    i += 1
                else:
                    break

            path = [h for _, h in heading_stack]
            blocks.append({
                "level": current_heading[0] if current_heading else 0,
                "heading": current_heading[1] if current_heading else "",
                "heading_path": path,
                "content": "\n".join(table_lines),
                "block_type": "table",
            })
            continue

        # --- normal line ---
        current_lines.append(line)
        i += 1

    flush_section()
    return blocks


# ============================================================
# TABLE SERIALIZATION
# ============================================================

def parse_markdown_table(table_text: str):
    """
    Return (header, rows) or (None, None) if invalid.
    """
    lines = [l for l in table_text.split("\n") if l.strip()]
    if len(lines) < 2:
        return None, None

    def split_row(line):
        s = line.strip().strip("|")
        return [c.strip() for c in s.split("|")]

    header = split_row(lines[0])
    separator_ok = all(
        re.fullmatch(r":?-{3,}:?", c) for c in split_row(lines[1])
    )
    if not separator_ok:
        return None, None

    rows = []
    for line in lines[2:]:
        rows.append(split_row(line))

    return header, rows


def serialize_table_rows(table_text: str):
    """
    Yield each table row as a text line: 'col1: a | col2: b'
    """
    header, rows = parse_markdown_table(table_text)
    if not header:
        yield table_text
        return

    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        parts = []
        for h, c in zip(header, padded):
            if c:
                parts.append(f"{h}: {c}")
        if parts:
            yield " | ".join(parts)


# ============================================================
# TOKEN-BASED SPLIT (fallback for long sections)
# ============================================================

def split_text_by_tokens(text: str,
                         max_tokens: int,
                         overlap_tokens: int,
                         prefix: str = ""):
    """
    Cắt text dài thành các mảnh <= max_tokens (đã tính cả prefix heading),
    có overlap. Mảnh nào cũng được gắn lại prefix.
    """
    enc = get_encoder()
    budget = max(max_tokens - count_tokens(prefix), overlap_tokens * 2)
    ids = enc.encode(text)

    if len(ids) <= budget:
        yield prefix + text
        return

    start = 0
    while start < len(ids):
        end = min(start + budget, len(ids))
        piece = enc.decode(ids[start:end])

        # cố gắng cắt ở ranh giới đoạn / câu gần cuối
        if end < len(ids):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                idx = piece.rfind(sep)
                if idx > len(piece) * 0.5:
                    piece = piece[: idx + len(sep)]
                    break
            # số token THỰC SỰ đã dùng sau khi cắt
            end = start + max(1, len(enc.encode(piece)))

        piece = piece.strip()
        if piece:
            yield prefix + piece

        if end >= len(ids):
            break
        start = max(end - overlap_tokens, start + 1)


# ============================================================
# CHUNK BUILDER
# ============================================================

def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-C{index:04d}"


def _chunk_record(doc_id, idx, heading_path, heading, level, block_type, text):
    return {
        "chunk_id": make_chunk_id(doc_id, idx),
        "doc_id": doc_id,
        "chunk_index": idx,
        "heading_path": heading_path,
        "section": heading,
        "level": level,
        "block_type": block_type,
        "text": text,
        "token_count": count_tokens(text),
        "char_count": len(text),
        "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    }


def build_chunks(doc_id: str, blocks) -> list[dict]:
    chunks = []
    idx = 0

    for block in blocks:
        block_type = block["block_type"]
        heading_path = block["heading_path"]
        heading = block["heading"]
        level = block["level"]
        content = block["content"]

        if not content.strip():
            continue

        prefix = ("# " + " > ".join(heading_path) + "\n\n") if heading_path else ""

        # --- diagram: một chunk, giữ nguyên ---
        if block_type == "diagram":
            chunks.append(_chunk_record(
                doc_id, idx, heading_path, heading, level,
                "diagram", prefix + content))
            idx += 1
            continue

        # --- table: gom nhiều dòng vào một chunk, có heading prefix ---
        if block_type == "table":
            rows = [r for r in serialize_table_rows(content) if r.strip()]
            buf, buf_tok = [], 0
            for row_text in rows + [None]:
                rt = count_tokens(row_text) if row_text else 0
                if buf and (row_text is None or buf_tok + rt > TABLE_GROUP_TOKENS):
                    chunks.append(_chunk_record(
                        doc_id, idx, heading_path, heading, level,
                        "table_rows", prefix + "\n".join(buf)))
                    idx += 1
                    buf, buf_tok = [], 0
                if row_text:
                    if rt > TABLE_GROUP_TOKENS:      # dòng quá dài: cắt riêng
                        for piece in split_text_by_tokens(
                                row_text, MAX_TOKENS, OVERLAP_TOKENS, prefix=prefix):
                            chunks.append(_chunk_record(
                                doc_id, idx, heading_path, heading, level,
                                "table_rows", piece))
                            idx += 1
                        continue
                    buf.append(row_text)
                    buf_tok += rt
            continue

        # --- section ---
        full_text = prefix + content
        full_tokens = count_tokens(full_text)
        if full_tokens <= MAX_TOKENS:
            if full_tokens < MIN_CHUNK_TOKENS:
                continue
            chunks.append(_chunk_record(
                doc_id, idx, heading_path, heading, level,
                "section", full_text))
            idx += 1
        else:
            for piece in split_text_by_tokens(
                content, MAX_TOKENS, OVERLAP_TOKENS, prefix=prefix
            ):
                if count_tokens(piece) < MIN_CHUNK_TOKENS:
                    continue
                chunks.append(_chunk_record(
                    doc_id, idx, heading_path, heading, level,
                    "section_split", piece))
                idx += 1

    return chunks


# ============================================================
# STATS
# ============================================================

def compute_stats(doc_id: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "doc_id": doc_id,
            "num_chunks": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "chunks_by_type": {},
        }

    token_counts = [c["token_count"] for c in chunks]

    by_type = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1

    sections = sorted({tuple(c["heading_path"]) for c in chunks})

    return {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 2),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "chunks_by_type": by_type,
        "num_sections": len(sections),
        "sections": [" > ".join(s) for s in sections],
    }


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(md_file: Path) -> dict:
    doc_id = md_file.parent.name

    text = md_file.read_text(encoding="utf-8", errors="replace")

    blocks = parse_blocks(text)
    chunks = build_chunks(doc_id, blocks)

    out_folder = OUTPUT_DIR / doc_id
    out_folder.mkdir(parents=True, exist_ok=True)

    jsonl_file = out_folder / f"{doc_id}.jsonl"
    with jsonl_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = compute_stats(doc_id, chunks)
    stats_file = out_folder / f"{doc_id}.stats.json"
    stats_file.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{doc_id}] chunks={stats['num_chunks']} "
        f"tokens={stats['total_tokens']} "
        f"avg={stats['avg_tokens']} "
        f"min={stats['min_tokens']} max={stats['max_tokens']} "
        f"types={stats['chunks_by_type']}"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PSAI CHUNKING PIPELINE v2")
    print("=" * 75)
    print(f"INPUT : {INPUT_DIR}")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print(f"MAX_TOKENS={MAX_TOKENS}  OVERLAP={OVERLAP_TOKENS}  TABLE_GROUP={TABLE_GROUP_TOKENS}")
    print()

    if not INPUT_DIR.exists():
        print("ERROR: input folder does not exist.")
        return

    if RESET_OUTPUT and OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.rglob("*.md"))
    if not files:
        print("ERROR: no .md files found.")
        return

    print(f"Found {len(files)} cleaned documents.")
    print()

    all_stats = []
    for md_file in files:
        try:
            stats = process_document(md_file)
            all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] {md_file}: {e}")

    total_chunks = sum(s["num_chunks"] for s in all_stats)
    total_tokens = sum(s.get("total_tokens", 0) for s in all_stats)

    summary = {
        "num_documents": len(all_stats),
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "per_document": all_stats,
    }

    summary_file = OUTPUT_DIR / "_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("CHUNKING FINISHED")
    print("=" * 75)
    print(f"Documents : {len(all_stats)}")
    print(f"Chunks    : {total_chunks}")
    print(f"Tokens    : {total_tokens:,}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Summary   : {summary_file}")


if __name__ == "__main__":
    main()