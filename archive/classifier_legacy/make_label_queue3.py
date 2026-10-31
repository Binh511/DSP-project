import pandas as pd

cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig")
cand["text"] = cand["text"].str.strip()

done = set()
for f in ("label_queue.csv", "label_queue2.csv"):
    d = pd.read_csv(f, encoding="utf-8-sig")
    done |= set(zip(d["document_id"], d["text"].str.strip()))

cand = cand[cand["text"].str.count(r"\|") < 2]
cand = cand[cand["text"].str.split().str.len() >= 6]
cand = cand[[(d, t) not in done for d, t in zip(cand["document_id"], cand["text"])]]

pool = cand.sample(frac=1, random_state=7).groupby("document_id").head(2)
q = pool.sample(min(150, len(pool)), random_state=8).sort_values(["document_id", "pos"])
q["prev_text"] = q["prev_text"].fillna("").str.slice(-150)
q["nhan_moi"] = ""
q[["document_id", "text", "prev_text", "nhan_moi"]].to_csv(
    "label_queue3.csv", index=False, encoding="utf-8-sig")
print("Đã ghi label_queue3.csv:", len(q), "câu,", q["document_id"].nunique(), "tài liệu")
