import hashlib
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import extraction

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
CSV_PATH = ROOT / "data_segment_cls" / "segment_classification.csv"
CLEANED = ROOT / "data_cleaned"
MODEL_DIRS = [ROOT / "models" / f"kfold_f{i}" for i in range(5)]
OUTPUT_PATH = ROOT / "pseudo_candidates.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

BULLET_RE = re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
MODAL_RE = re.compile(r"\b(will|must|should|shall|need to|needs to)\b", re.I)
IMPER_RE = re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b",
    re.I,
)
TIME_RE = re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", re.I)


def context(section, prev, pos, text, nxt):
    bullet = 1 if BULLET_RE.match(text) else 0
    cue = (
        f"{int(bool(MODAL_RE.search(text)))}"
        f"{int(bool(IMPER_RE.match(text)))}"
        f"{int(bool(TIME_RE.search(text)))}"
    )
    return (
        f"Section: {str(section).strip()[-120:]} | Pos: {int(pos * 10)} | "
        f"Bullet: {bullet} | Cue: {cue} | Previous: {str(prev).strip()[-150:]} | "
        f"Next: {str(nxt).strip()[:150]}"
    )


# Find unlabeled documents and discard duplicate document contents.
labeled = set(
    pd.read_csv(CSV_PATH, encoding="utf-8-sig", usecols=["document_id"])[
        "document_id"
    ]
)
docs, seen_hash = {}, set()
for path in sorted(list(CLEANED.rglob("*.md")) + list(CLEANED.rglob("*.txt"))):
    document_id = path.parent.name
    if document_id in labeled or document_id in docs:
        continue
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if content_hash in seen_hash:
        continue
    seen_hash.add(content_hash)
    docs[document_id] = text
print("Tài liệu chưa gán nhãn (đã bỏ trùng):", len(docs))

# Split documents into candidate sentences.
rows = []
for document_id, text in docs.items():
    items = list(extraction.sentences_with_context(text))
    for item in items:
        value = extraction._clean_value(item["text"])
        if extraction._is_noise(value) or extraction._REF_SECTION.search(item["section"]):
            continue
        rows.append(
            {
                "document_id": document_id,
                "text": item["text"],
                "section": item["section"],
                "prev_text": item["prev"],
                "pos": item["pos"],
                "next_text": item["next_text"],
            }
        )

candidate_columns = [
    "document_id",
    "text",
    "section",
    "prev_text",
    "pos",
    "next_text",
]
cand = pd.DataFrame(rows, columns=candidate_columns)
print("Số câu ứng viên:", len(cand))


def softmax(logits):
    exponentials = np.exp(logits - logits.max(1, keepdims=True))
    return exponentials / exponentials.sum(1, keepdims=True)


if len(cand):
    missing_models = [str(path) for path in MODEL_DIRS if not path.is_dir()]
    if missing_models:
        raise FileNotFoundError(
            "Không tìm thấy thư mục mô hình fold: " + ", ".join(missing_models)
        )

    acc = np.zeros((len(cand), len(LABELS)))
    contexts = [
        context(row.section, row.prev_text, row.pos, row.text, row.next_text)
        for row in cand.itertuples()
    ]
    texts = [text[:600] for text in cand["text"]]
    for model_dir in MODEL_DIRS:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir)
            .to(device)
            .eval()
        )
        outputs = []
        for start in range(0, len(cand), 32):
            encoded = tokenizer(
                texts[start : start + 32],
                contexts[start : start + 32],
                truncation="longest_first",
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits.float().cpu().numpy()
            outputs.append(softmax(logits))
        acc += np.vstack(outputs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("xong", model_dir)

    probabilities = acc / len(MODEL_DIRS)
    cand["pred"] = [LABELS[index] for index in probabilities.argmax(1)]
    cand["score"] = probabilities.max(1)
else:
    cand["pred"] = pd.Series(dtype="object")
    cand["score"] = pd.Series(dtype="float64")
    print("Không có câu ứng viên để dự đoán; sẽ ghi CSV rỗng.")

# Summarize predicted labels by confidence threshold.
print("\nSố câu theo nhãn dự đoán và ngưỡng:")
counts_by_threshold = {
    threshold: cand[cand.score >= threshold].pred.value_counts()
    for threshold in (0.6, 0.7, 0.8, 0.9)
}
print(pd.DataFrame(counts_by_threshold).reindex(LABELS).fillna(0).astype(int))

# Show random high-confidence examples for manual review.
random.seed(0)
for label in LABELS[:4]:
    subset = cand[(cand.pred == label) & (cand.score >= 0.8)]
    print(f"\n===== {label}: {len(subset)} câu >=0.8, xem 10 câu ngẫu nhiên =====")
    for row in subset.sample(min(10, len(subset)), random_state=0).itertuples():
        print(f"  [{row.score:.2f}] {row.document_id}: {row.text[:110]}")

cand.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nĐã ghi {OUTPUT_PATH.name} (cần cho bước sau)")
