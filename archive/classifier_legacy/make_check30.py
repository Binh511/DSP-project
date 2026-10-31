import pandas as pd

q = pd.read_csv("label_queue3.csv", encoding="utf-8-sig")
s = q.sample(30, random_state=5).sort_index()
s["prev_text"] = s["prev_text"].fillna("").str.slice(-150)
s["nhan_ban"] = ""
s[["document_id", "text", "prev_text", "nhan_ban"]].to_csv(
    "check30.csv", index=False, encoding="utf-8-sig")
print("Đã ghi check30.csv:", len(s), "dòng")
