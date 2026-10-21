from pathlib import Path
import json
import os
import re
import time
import hashlib
from typing import Optional
from tqdm import tqdm

import google.generativeai as genai
from transformers import AutoTokenizer


# ============================================================
# PSAI AUTO LABELING v1  — Google Gemini
#
# Input:
#   data_chunked/<doc_id>/<doc_id>.jsonl
#
# Output:
#   data_labeled/<doc_id>_labeled.jsonl    (spans + BIO)
#   data_labeled/_labeling_summary.json    (cost, stats)
#
# Labels:
#   OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT, OTHER
#
# Skip:
#   - block_type = "diagram"
#   - chunks < MIN_CHUNK_TOKENS
# ============================================================


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\fall26\DSP_PROJECT")

CHUNK_DIR = BASE_DIR / "data_chunked"
OUTPUT_DIR = BASE_DIR / "data_labeled"
CHECKPOINT_DIR = BASE_DIR / "data_labeled" / "_checkpoints"

# API
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"       # nhanh, rẻ, ổn định cho NER
                                        # có thể đổi: gemini-2.5-flash, gemini-2.5-pro
TEMPERATURE = 0.0                       # deterministic
MAX_OUTPUT_TOKENS = 2048

# Retry
MAX_RETRIES = 5
RETRY_DELAY = 5                         # giây

# Skip rules
SKIP_BLOCK_TYPES = {"diagram"}
MIN_CHUNK_TOKENS = 20                   # bỏ chunk rác

# Tokenizer cho BIO
TOKENIZER_NAME = "bert-base-multilingual-cased"

# Labels
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]

LABEL2ID = {"O": 0}
for i, label in enumerate(LABELS, start=1):
    LABEL2ID[f"B-{label}"] = i * 2 - 1
    LABEL2ID[f"I-{label}"] = i * 2
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Pricing (USD per 1M tokens) — cập nhật theo Google
# Gemini 2.0 Flash (2025-01): input $0.10, output $0.40
PRICE_INPUT_PER_M = 0.10
PRICE_OUTPUT_PER_M = 0.40


# ============================================================
# GEMINI SETUP
# ============================================================

def init_gemini():
    if not API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY chưa được set. "
            "Chạy: $env:GOOGLE_API_KEY = '<your API key>'"
        )
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        },
    )
    return model


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """You are an information extraction assistant for project proposals.

Label every relevant text span in the input with ONE of these 5 labels:

- OBJECTIVE: project goal, purpose, what the project aims to achieve
- REQUIREMENT: constraint, must-have, condition, technical requirement
- ACTIVITY: task, action, work item, development step
- EXPECTED_RESULT: deliverable, output, expected outcome
- OTHER: information not belonging to the above (team, admin, background)

STRICT RULES:
1. Return ONLY a JSON object, no markdown, no explanation.
2. Each span must have: start (int), end (int), label (string), text (string).
3. `start` and `end` are CHARACTER OFFSETS in the input text.
4. `text` must EXACTLY equal input[start:end].
5. DO NOT label headings (lines starting with #).
6. DO NOT label short filler words.
7. Prefer short meaningful spans over long sentences.
8. If a span is ambiguous, use OTHER.
9. If nothing matches, return {"spans": []}.

INPUT TEXT:
\"\"\"
{text}
\"\"\"

OUTPUT JSON:
"""


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


# ============================================================
# GEMINI CALL WITH RETRY
# ============================================================

def call_gemini(model, prompt: str) -> tuple[Optional[dict], dict]:
    """
    Trả về (parsed_json, usage_info).
    parsed_json = {"spans": [...]} hoặc None nếu fail.
    """
    last_err = None
    usage = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)

            # usage
            try:
                um = response.usage_metadata
                usage["input_tokens"] = getattr(um, "prompt_token_count", 0)
                usage["output_tokens"] = getattr(um, "candidates_token_count", 0)
            except Exception:
                pass

            raw = response.text.strip()

            # strip code fences if any
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            if not isinstance(data, dict) or "spans" not in data:
                raise ValueError("Missing 'spans' key")

            return data, usage

        except Exception as e:
            last_err = e
            wait = RETRY_DELAY * (attempt + 1)
            print(f"   [retry {attempt + 1}/{MAX_RETRIES}] {e} — wait {wait}s")
            time.sleep(wait)

    print(f"   [FAIL] {last_err}")
    return None, usage


# ============================================================
# VALIDATE SPANS
# ============================================================

def validate_spans(text: str, spans: list) -> list:
    """
    Lọc span lỗi:
      - start/end ngoài text
      - text không khớp input[start:end]
      - label không thuộc LABELS
      - span quá ngắn
      - span nằm trên heading
    """
    valid = []
    heading_ranges = find_heading_ranges(text)

    for s in spans:
        try:
            start = int(s["start"])
            end = int(s["end"])
            label = str(s["label"]).upper().strip()
            span_text = s.get("text", "")
        except Exception:
            continue

        if label not in LABELS:
            continue
        if start < 0 or end > len(text) or start >= end:
            continue
        if text[start:end] != span_text:
            # thử tự sửa nếu text khớp sau khi strip
            if text[start:end].strip() != span_text.strip():
                continue
        if end - start < 3:
            continue
        # bỏ span nằm trong heading
        if any(h_start <= start and end <= h_end
               for h_start, h_end in heading_ranges):
            continue

        valid.append({
            "start": start,
            "end": end,
            "label": label,
            "text": text[start:end],
        })

    # sort theo start
    valid.sort(key=lambda x: x["start"])

    # loại overlap: giữ span dài hơn
    no_overlap = []
    for s in valid:
        if no_overlap and s["start"] < no_overlap[-1]["end"]:
            prev = no_overlap[-1]
            if (s["end"] - s["start"]) > (prev["end"] - prev["start"]):
                no_overlap[-1] = s
            continue
        no_overlap.append(s)

    return no_overlap


def find_heading_ranges(text: str):
    ranges = []
    for m in re.finditer(r"^#{1,6}\s+.+$", text, re.MULTILINE):
        ranges.append((m.start(), m.end()))
    return ranges


# ============================================================
# CHAR SPANS -> BIO TAGS
# ============================================================

def spans_to_bio(text: str, spans: list, tokenizer) -> dict:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    offsets = encoding["offset_mapping"]
    tags = ["O"] * len(offsets)

    for span in spans:
        start, end, label = span["start"], span["end"], span["label"]

        tok_idx = []
        for i, (ts, te) in enumerate(offsets):
            if ts == te:
                continue
            if te <= start:
                continue
            if ts >= end:
                break
            tok_idx.append(i)

        if not tok_idx:
            continue

        tags[tok_idx[0]] = f"B-{label}"
        for i in tok_idx[1:]:
            tags[i] = f"I-{label}"

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "tags": tags,
        "tag_ids": [LABEL2ID[t] for t in tags],
    }


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(doc_id: str) -> set:
    cp = CHECKPOINT_DIR / f"{doc_id}.done"
    if not cp.exists():
        return set()
    return set(cp.read_text(encoding="utf-8").splitlines())


def append_checkpoint(doc_id: str, chunk_id: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp = CHECKPOINT_DIR / f"{doc_id}.done"
    with cp.open("a", encoding="utf-8") as f:
        f.write(chunk_id + "\n")


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(jsonl_path: Path, model, tokenizer) -> dict:
    doc_id = jsonl_path.parent.name
    chunks = load_chunks(jsonl_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{doc_id}_labeled.jsonl"

    done = load_checkpoint(doc_id)

    stats = {
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "labeled": 0,
        "skipped_type": 0,
        "skipped_short": 0,
        "failed": 0,
        "already_done": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    with out_file.open("a", encoding="utf-8") as fout:
        for chunk in tqdm(chunks, desc=f"[{doc_id}]", unit="chunk"):
            cid = chunk["chunk_id"]

            if cid in done:
                stats["already_done"] += 1
                continue

            if chunk.get("block_type") in SKIP_BLOCK_TYPES:
                stats["skipped_type"] += 1
                append_checkpoint(doc_id, cid)
                continue

            if chunk.get("token_count", 0) < MIN_CHUNK_TOKENS:
                stats["skipped_short"] += 1
                append_checkpoint(doc_id, cid)
                continue

            text = chunk["text"]
            prompt = build_prompt(text)

            data, usage = call_gemini(model, prompt)
            stats["input_tokens"] += usage["input_tokens"]
            stats["output_tokens"] += usage["output_tokens"]

            if data is None:
                stats["failed"] += 1
                continue

            raw_spans = data.get("spans", [])
            spans = validate_spans(text, raw_spans)

            bio = spans_to_bio(text, spans, tokenizer)

            record = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "section": chunk.get("section", ""),
                "heading_path": chunk.get("heading_path", []),
                "block_type": chunk.get("block_type", "section"),
                "text": text,
                "spans": spans,
                "input_ids": bio["input_ids"],
                "attention_mask": bio["attention_mask"],
                "tags": bio["tags"],
                "tag_ids": bio["tag_ids"],
                "num_spans": len(spans),
                "labeled_by": MODEL_NAME,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            stats["labeled"] += 1
            append_checkpoint(doc_id, cid)

    # cost
    cost = (
        stats["input_tokens"] / 1_000_000 * PRICE_INPUT_PER_M
        + stats["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
    )
    stats["cost_usd"] = round(cost, 6)

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("PSAI AUTO LABELING v1 — Google Gemini")
    print("=" * 75)
    print(f"CHUNK_DIR : {CHUNK_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"MODEL     : {MODEL_NAME}")
    print()

    if not CHUNK_DIR.exists():
        print("ERROR: chunk folder does not exist.")
        return

    print("Initializing Gemini...")
    model = init_gemini()

    print(f"Loading tokenizer {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    jsonl_files = sorted(CHUNK_DIR.rglob("*.jsonl"))
    if not jsonl_files:
        print("ERROR: no chunk files found.")
        return

    print(f"Found {len(jsonl_files)} documents.")
    print()

    all_stats = []
    for jf in jsonl_files:
        try:
            stats = process_document(jf, model, tokenizer)
            all_stats.append(stats)
            print(
                f"[{stats['doc_id']}] "
                f"labeled={stats['labeled']} "
                f"skip_type={stats['skipped_type']} "
                f"skip_short={stats['skipped_short']} "
                f"failed={stats['failed']} "
                f"cost=${stats['cost_usd']}"
            )
        except Exception as e:
            print(f"[ERROR] {jf}: {e}")

    total_cost = sum(s.get("cost_usd", 0) for s in all_stats)
    total_labeled = sum(s["labeled"] for s in all_stats)
    total_input = sum(s["input_tokens"] for s in all_stats)
    total_output = sum(s["output_tokens"] for s in all_stats)

    summary = {
        "model": MODEL_NAME,
        "num_documents": len(all_stats),
        "total_labeled_chunks": total_labeled,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 6),
        "per_document": all_stats,
    }

    summary_file = OUTPUT_DIR / "_labeling_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 75)
    print("LABELING FINISHED")
    print("=" * 75)
    print(f"Documents        : {len(all_stats)}")
    print(f"Labeled chunks   : {total_labeled}")
    print(f"Input tokens     : {total_input:,}")
    print(f"Output tokens    : {total_output:,}")
    print(f"Total cost       : ${total_cost:.4f}")
    print(f"Output           : {OUTPUT_DIR}")
    print(f"Summary          : {summary_file}")


if __name__ == "__main__":
    main()