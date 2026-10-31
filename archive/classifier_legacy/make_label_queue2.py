import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig")
done = pd.read_csv("label_queue.csv", encoding="utf-8-sig")
done_keys = set(zip(done["document_id"], done["text"].str.strip()))

cand["text"] = cand["text"].str.strip()
cand = cand[(cand["text"].str.count(r"\|") < 2) & (cand["document_id"] != "PSAI162")]
cand = cand[cand["text"].str.split().str.len() >= 6]
cand = cand[[(d, t) not in done_keys for d, t in zip(cand["document_id"], cand["text"])]]

PLAN = [   # (nhãn mô hình đoán, điểm từ, điểm đến, số câu)
    ("ACTIVITY", 0.70, 0.85, 80),
    ("OBJECTIVE", 0.50, 0.85, 40),
    ("REQUIREMENT", 0.75, 0.95, 40),
    ("EXPECTED_RESULT", 0.75, 0.95, 40),
]
parts = []
for lab, lo, hi, n in PLAN:
    sub = cand[(cand.pred == lab) & cand.score.between(lo, hi)]
    sub = sub.sample(frac=1, random_state=0).groupby("document_id").head(4)
    pick = sub.sample(min(n, len(sub)), random_state=1)
    print(f"{lab:16s} dải {lo}-{hi}: có {len(sub)} câu, lấy {len(pick)}")
    parts.append(pick)

q = pd.concat(parts).sort_values(["document_id", "pos"])
q = q.rename(columns={"pred": "mo_hinh_doan", "score": "diem"})
q["nhan_moi"] = ""
q["prev_text"] = q["prev_text"].fillna("").str.slice(-150)
q[["document_id", "mo_hinh_doan", "diem", "text", "prev_text", "nhan_moi"]].to_csv(
    "label_queue2.csv", index=False, encoding="utf-8-sig")
print("Đã ghi label_queue2.csv:", len(q), "câu,", q["document_id"].nunique(), "tài liệu")
