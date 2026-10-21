import hashlib
import re
import sys

import numpy as np
import pandas as pd

import ensemble_classifier as sentence_classifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
df = pd.read_csv("data_segment_cls/segment_classification.csv", encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper()))
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)

paths, prevs, poss = [], [], []
for _, g in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(g)
    for k, t in enumerate(g["text"]):
        s = t.strip()
        m = re.match(r"^(#+)\s*(.*)$", s)
        if m:
            stack = stack[:len(m.group(1)) - 1] + [m.group(2).strip()]
        paths.append(" > ".join(stack))
        prevs.append(prev)
        poss.append(k / max(n - 1, 1))
        prev = s
df["section"], df["prev_text"], df["pos"] = paths, prevs, poss
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")


def doc_key(g):
    return hashlib.md5(
        " ".join(sorted(g["text"].str.lower().str.strip())).encode("utf-8")
    ).hexdigest()


d2k = {d: doc_key(g) for d, g in df.groupby("document_id")}
k2d = {}
for d, k in d2k.items():
    k2d.setdefault(k, []).append(d)
keys = sorted(k2d)
np.random.RandomState(42).shuffle(keys)
n = len(keys)
test_docs = {d for k in keys[int(n * 0.9):] for d in k2d[k]}
t = df[df.document_id.isin(test_docs)].reset_index(drop=True)
print("test:", len(t), "câu,", len(test_docs), "tài liệu (phải là ~800 câu, 12 tài liệu)")

preds = sentence_classifier.classify([
    {"text": r.text, "section": r.section, "prev_text": r.prev_text,
     "pos": r.pos, "next_text": r.next_text} for r in t.itertuples()])
t["pred"] = [p["label"] for p in preds]
t["score"] = [p["score"] for p in preds]

print("\nNgưỡng | nhãn             | số đoán |   P    |   R")
for thr in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
    for lab in LABELS[:4]:
        kept = t[(t.pred == lab) & (t.score >= thr)]
        gold = (t.label == lab).sum()
        hit = (kept.label == lab).sum()
        print(f" {thr:.2f}  | {lab:16s} | {len(kept):7d} | "
              f"{hit/len(kept) if len(kept) else 0:.2f} | {hit/gold if gold else 0:.2f}")
    print("-" * 52)
