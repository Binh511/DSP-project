import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import extraction
import sentence_classifier

DOC_ID = sys.argv[1] if len(sys.argv) > 1 else "PSAI150"
CSV_PATH = Path("data_segment_cls/segment_classification.csv")
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())
)
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)
d = df[df["document_id"] == DOC_ID].sort_values("source_line").reset_index(drop=True)

paths, prevs, positions = [], [], []
stack, prev, total = [], "", len(d)
for index, text in enumerate(d["text"]):
    stripped = text.strip()
    match = re.match(r"^(#+)\s*(.*)$", stripped)
    if match:
        level = len(match.group(1))
        stack = stack[:level - 1] + [match.group(2).strip()]
    paths.append(" > ".join(stack))
    prevs.append(prev)
    positions.append(index / max(total - 1, 1))
    prev = stripped
d["section"], d["prev_text"], d["pos"] = paths, prevs, positions
d = d[~d["text"].str.strip().str.startswith("#")].reset_index(drop=True)
d["next_text"] = d["text"].shift(-1).fillna("")

preds = sentence_classifier.classify([
    {"text": r.text, "section": r.section, "prev_text": r.prev_text,
     "pos": r.pos, "next_text": r.next_text}
    for r in d.itertuples()
])
d["pred"] = [p["label"] for p in preds]
d["score"] = [p["score"] for p in preds]

# áp dụng ngưỡng riêng như trong extraction.py
def kept(row):
    thr = extraction.MIN_SCORE_BY_LABEL.get(row["pred"], extraction.DEFAULT_MIN_SCORE)
    return row["pred"] != "OTHER" and row["score"] >= thr

d["kept"] = d.apply(kept, axis=1)

print(f"=== {DOC_ID}: {len(d)} câu có nhãn thật ===")
print(f"{'nhãn':16s} {'nhãn thật':>10s} {'mô hình giữ':>12s} {'đúng':>6s}")
for lab in LABELS[:4]:
    gold = (d["label"] == lab).sum()
    k = d[(d["pred"] == lab) & d["kept"]]
    hit = (k["label"] == lab).sum()
    print(f"{lab:16s} {gold:10d} {len(k):12d} {hit:6d}")

wrong = d[(d["pred"] == "ACTIVITY") & d["kept"] & (d["label"] != "ACTIVITY")]
print(f"\nCâu mô hình giữ là ACTIVITY nhưng nhãn thật khác ({len(wrong)} câu), xem 12 câu đầu:")
for r in wrong.head(12).itertuples():
    print(f"  [{r.score:.2f}] thật={r.label:15s} {r.text[:90]}")