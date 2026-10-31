import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig")
cand = cand[cand["text"].str.count(r"\|") < 2]

for lab in ("REQUIREMENT", "EXPECTED_RESULT"):
    sub = cand[(cand.pred == lab) & (cand.score.between(0.5, 0.75)) & (cand.document_id != "PSAI162")]
    print(f"\n===== {lab}: {len(sub)} câu điểm 0.5-0.75, xem 20 câu =====")
    for i, r in enumerate(sub.sample(min(20, len(sub)), random_state=3).itertuples(), 1):
        print(f"{i:2d}. [{r.score:.2f}] {r.text[:140]}")
