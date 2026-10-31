import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig")
cand = cand[(cand["text"].str.count(r"\|") < 2) & (cand["document_id"] != "PSAI162")].copy()
cand["text"] = cand["text"].str.strip()
cand = cand[cand["text"].str.split().str.len() >= 6]

# chỉ lấy câu mô hình do dự: điểm dự đoán thấp, và đoán là một trong hai nhãn yếu
unsure = cand[cand["pred"].isin(["REQUIREMENT", "EXPECTED_RESULT"]) & cand["score"].between(0.40, 0.75)]

# tối đa 6 câu mỗi tài liệu để dữ liệu không dồn vào vài tài liệu
unsure = unsure.sample(frac=1, random_state=0).groupby("document_id").head(6)
queue = unsure.sample(min(120, len(unsure)), random_state=1).sort_values(["document_id", "pos"])

queue = queue.rename(columns={"pred": "mo_hinh_doan", "score": "diem"})
queue["nhan_moi"] = ""
queue["prev_text"] = queue["prev_text"].fillna("").str.slice(-150)
queue[["document_id", "mo_hinh_doan", "diem", "text", "prev_text", "nhan_moi"]].to_csv(
    "label_queue.csv", index=False, encoding="utf-8-sig")
print("Có", len(unsure), "câu do dự; đã ghi label_queue.csv:", len(queue), "câu,",
      queue["document_id"].nunique(), "tài liệu")
