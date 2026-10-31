import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "pseudo_candidates.csv"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT"]

cand = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("Tổng ứng viên:", len(cand))

is_table = cand["text"].str.count(r"\|") >= 2
print("Mảnh bảng (>=2 dấu |):", int(is_table.sum()))
print("\nTỉ lệ mảnh bảng trong từng nhãn dự đoán:")
print(
    cand.groupby("pred")["text"].apply(
        lambda sentences: round((sentences.str.count(r"\|") >= 2).mean(), 3)
    )
)

clean = cand[~is_table]

print("\nTài liệu chiếm nhiều nhất trong mỗi nhãn dự đoán (score >= 0.8, sau khi bỏ bảng):")
for label in LABELS:
    subset = clean[(clean.pred == label) & (clean.score >= 0.8)]
    top = subset["document_id"].value_counts().head(5)
    top_documents = ", ".join(f"{document_id}={count}" for document_id, count in top.items())
    print(f"  {label:16s} tổng {len(subset):4d} | top: {top_documents}")

print("\nSố câu sau khi bỏ bảng:")
counts_by_threshold = {
    threshold: clean[clean.score >= threshold].pred.value_counts()
    for threshold in (0.6, 0.7, 0.8)
}
print(pd.DataFrame(counts_by_threshold).reindex(LABELS).fillna(0).astype(int))

for label in ("OBJECTIVE", "ACTIVITY"):
    subset = clean[
        (clean.pred == label)
        & (clean.score >= 0.8)
        & (clean.document_id != "PSAI162")
    ]
    print(f"\n===== {label}: 15 câu ngẫu nhiên (đã bỏ bảng và PSAI162) =====")
    for row in subset.sample(min(15, len(subset)), random_state=1).itertuples():
        print(f"  [{row.score:.2f}] {row.document_id}: {row.text[:110]}")
