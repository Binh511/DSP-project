from pathlib import Path
import argparse
import json
import re
import hashlib


# ============================================================
# PSAI CHUNKING PIPELINE v3
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
#   - Diagrams retained, split when needed to respect the token budget
#   - Source line ranges identify the enclosing Markdown block, not exact chunk spans
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data_cleaned"
OUTPUT_DIR = BASE_DIR / "data_chunked"

RESET_OUTPUT = False  # Existing outputs require explicit --overwrite; never delete the tree.

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
TABLE_GROUP_TOKENS = 350
MIN_CHUNK_TOKENS = 1          # Short requirements/results must not silently disappear.
TOKENIZER_NAME = "cl100k_base"


# ============================================================
# TOKENIZER
# ============================================================

_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))


# ============================================================
# MARKDOWN PARSING
# ============================================================

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")


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
    fence_marker = ""
    fence_start = 0
    section_start = 0

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
                "source_start_line": section_start + 1,
                "source_end_line": i,
            })
        current_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # --- code fence ---
        fence_match = CODE_FENCE_RE.match(s)
        if fence_match and (not inside_fence or (
                fence_match.group(1)[0] == fence_marker[0]
                and len(fence_match.group(1)) >= len(fence_marker)
                and not fence_match.group(2).strip())):
            if not inside_fence:
                flush_section()
                inside_fence = True
                fence_lines = [line]
                fence_marker = fence_match.group(1)
                fence_start = i
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
                    "source_start_line": fence_start + 1,
                    "source_end_line": i + 1,
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
            table_start = i
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
                "source_start_line": table_start + 1,
                "source_end_line": i,
            })
            continue

        # --- normal line ---
        if not current_lines:
            section_start = i
        current_lines.append(line)
        i += 1

    if inside_fence:
        blocks.append({
            "level": current_heading[0] if current_heading else 0,
            "heading": current_heading[1] if current_heading else "",
            "heading_path": [h for _, h in heading_stack],
            "content": "\n".join(fence_lines).strip(), "block_type": "diagram",
            "source_start_line": fence_start + 1, "source_end_line": len(lines),
            "unterminated_fence": True,
        })
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
    if not rows:
        yield table_text
        return

    for row in rows:
        if len(row) > len(header):
            # Malformed tables must remain visible instead of dropping trailing cells.
            yield " | ".join(row)
            continue
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
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens > overlap_tokens >= 0.")
    budget = max_tokens - count_tokens(prefix)
    if budget <= 0:
        raise ValueError("Heading prefix consumes the complete token budget; shorten it explicitly.")
    effective_overlap = min(overlap_tokens, budget - 1)
    if count_tokens(prefix + text) <= max_tokens:
        yield prefix + text
        return

    # Slice the original Unicode string. Decoding arbitrary token windows can split
    # a UTF-8 character and introduce replacement characters absent from the source.
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        end = start
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(prefix + text[start:middle]) <= max_tokens:
                end, low = middle, middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError("Token budget cannot fit one source character with the heading prefix.")
        piece = text[start:end]

        # cố gắng cắt ở ranh giới đoạn / câu gần cuối
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                idx = piece.rfind(sep)
                if idx > len(piece) * 0.5:
                    piece = piece[: idx + len(sep)]
                    break
            # số token THỰC SỰ đã dùng sau khi cắt
            end = start + len(piece)

        piece = piece.strip()
        if piece:
            yield prefix + piece

        if end >= len(text):
            break
        next_start = end
        while next_start > start + 1 and count_tokens(text[next_start - 1:end]) <= effective_overlap:
            next_start -= 1
        start = max(next_start, start + 1)


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

        block_chunk_start = len(chunks)
        # --- diagram ---
        if block_type == "diagram":
            for piece in split_text_by_tokens(content, MAX_TOKENS, OVERLAP_TOKENS, prefix=prefix):
                chunks.append(_chunk_record(doc_id, idx, heading_path, heading, level, "diagram", piece))
                idx += 1
            for record in chunks[block_chunk_start:]:
                record.update({key: block[key] for key in ("source_start_line", "source_end_line", "unterminated_fence") if key in block})
            continue

        # --- table: gom nhiều dòng vào một chunk, có heading prefix ---
        if block_type == "table":
            rows = [r for r in serialize_table_rows(content) if r.strip()]
            buf, buf_tok = [], 0
            for row_text in rows + [None]:
                rt = count_tokens(row_text) if row_text else 0
                if buf and (row_text is None or buf_tok + rt > TABLE_GROUP_TOKENS):
                    for piece in split_text_by_tokens("\n".join(buf), MAX_TOKENS, OVERLAP_TOKENS, prefix=prefix):
                        chunks.append(_chunk_record(doc_id, idx, heading_path, heading, level, "table_rows", piece))
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
            for record in chunks[block_chunk_start:]:
                record.update({key: block[key] for key in ("source_start_line", "source_end_line") if key in block})
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

        for record in chunks[block_chunk_start:]:
            record.update({key: block[key] for key in ("source_start_line", "source_end_line") if key in block})

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

def process_document(md_file: Path, *, output_dir=None, overwrite=False) -> dict:
    md_file = Path(md_file)
    doc_id = md_file.parent.name
    out_folder = Path(output_dir or OUTPUT_DIR) / doc_id
    if out_folder.resolve() == md_file.parent.resolve():
        raise ValueError("Chunk output must be separate from the source document.")
    if out_folder.exists() and any(out_folder.iterdir()) and not overwrite:
        raise FileExistsError(f"Output already exists: {out_folder}. Use --overwrite explicitly.")
    source_bytes = md_file.read_bytes()
    text = source_bytes.decode("utf-8")

    blocks = parse_blocks(text)
    chunks = build_chunks(doc_id, blocks)
    for chunk in chunks:
        chunk.update({"source_file": str(md_file.resolve()),
                      "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                      "chunking_version": "psai-chunking-v3"})

    out_folder.mkdir(parents=True, exist_ok=True)

    jsonl_file = out_folder / f"{doc_id}.jsonl"
    with jsonl_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = compute_stats(doc_id, chunks)
    stats.update({"source_file": str(md_file.resolve()),
                  "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "chunking_version": "psai-chunking-v3", "tokenizer_name": TOKENIZER_NAME,
                  "max_tokens": MAX_TOKENS, "overlap_tokens": OVERLAP_TOKENS,
                  "minimum_chunk_tokens": MIN_CHUNK_TOKENS})
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="Chunk Markdown; retain short content and source provenance.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print("=" * 75)
    print("PSAI CHUNKING PIPELINE v3")
    print("=" * 75)
    print(f"INPUT : {args.input_dir}")
    print(f"OUTPUT: {args.output_dir}")
    print(f"MAX_TOKENS={MAX_TOKENS}  OVERLAP={OVERLAP_TOKENS}  TABLE_GROUP={TABLE_GROUP_TOKENS}")
    print()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_dir}")
    if args.output_dir.resolve() == args.input_dir.resolve() or args.output_dir.resolve() in args.input_dir.resolve().parents or args.input_dir.resolve() in args.output_dir.resolve().parents:
        raise ValueError("Input and output trees must be separate and non-overlapping.")
    files = sorted(args.input_dir.rglob("*.md"))
    if not files:
        raise ValueError("No .md files found; no output was changed.")
    doc_ids = [path.parent.name for path in files]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Multiple Markdown files map to the same document ID.")
    if (args.output_dir / "_summary.json").exists() and not args.overwrite:
        raise FileExistsError("Output summary already exists; choose a new output directory or --overwrite.")

    print(f"Found {len(files)} cleaned documents.")
    print()

    all_stats = []
    failures = []
    for md_file in files:
        try:
            stats = process_document(md_file, output_dir=args.output_dir, overwrite=args.overwrite)
            all_stats.append(stats)
        except Exception as e:
            print(f"[ERROR] {md_file}: {e}")
            failures.append({"source": str(md_file), "error": str(e)})

    total_chunks = sum(s["num_chunks"] for s in all_stats)
    total_tokens = sum(s.get("total_tokens", 0) for s in all_stats)

    summary = {
        "num_documents": len(all_stats),
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "per_document": all_stats,
        "failures": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = args.output_dir / "_summary.json"
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
    print(f"Output    : {args.output_dir}")
    print(f"Summary   : {summary_file}")
    if failures:
        raise RuntimeError("Chunking completed with errors; inspect the summary before embedding.")


if __name__ == "__main__":
    main()
