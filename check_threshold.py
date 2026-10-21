import re
from pathlib import Path

import numpy as np
import pandas as pd

import sentence_classifier

# SỬA nếu file CSV nằm chỗ khác
CSV_PATH = Path("data_segment_cls/segment_classification.csv")

LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

def clean_label(x):
    x = str(x).replace("\\", "").replace("`", "").strip().upper()
    return re.sub(r"^[BI]-", "", x)

df["label"] = df["label"].map(clean_label)
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)

# ngữ cảnh (giống lúc train)
df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)
paths, prevs, positions = [], [], []
for _, group in df.groupby("document_id", sort=False):
    stack, prev, total = [], "", len(group)
    for index, text in enumerate(group["text"]):
        stripped = text.strip()
        match = re.match(r"^(#+)\s*(.*)$", stripped)
        if match:
            level = len(match.group(1))
            stack = stack[:level - 1] + [match.group(2).strip()]
        paths.append(" > ".join(stack))
        prevs.append(prev)
        positions.append(index / max(total - 1, 1))
        prev = stripped
df["section"], df["prev_text"], df["pos"] = paths, prevs, positions
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")

# chia tập y hệt lúc train
rng = np.random.RandomState(42)
doc_ids = df["document_id"].unique()
rng.shuffle(doc_ids)
n = len(doc_ids)
test_docs = set(doc_ids[int(n * 0.8) + int(n * 0.1):])
test_df = df[df["document_id"].isin(test_docs)].reset_index(drop=True)
print("Số câu test:", len(test_df), "(mong đợi 1073)")

preds = sentence_classifier.classify([
    {"text": r.text, "section": r.section, "prev_text": r.prev_text,
     "pos": r.pos, "next_text": r.next_text}
    for r in test_df.itertuples()
])
test_df["pred"] = [p["label"] for p in preds]
test_df["score"] = [p["score"] for p in preds]

print("\nNgưỡng | nhãn             | số đoán |   P    |   R")
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    for lab in LABELS[:4]:
        kept = test_df[(test_df["pred"] == lab) & (test_df["score"] >= t)]
        true_total = (test_df["label"] == lab).sum()
        hit = (kept["label"] == lab).sum()
        p = hit / len(kept) if len(kept) else 0
        r = hit / true_total if true_total else 0
        print(f" {t:.1f}   | {lab:16s} | {len(kept):7d} | {p:.2f} | {r:.2f}")
    print("-" * 52)