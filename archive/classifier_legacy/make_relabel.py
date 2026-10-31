import pandas as pd

a = pd.read_csv("label_queue.csv", encoding="utf-8-sig")
b = pd.read_csv("label_queue2.csv", encoding="utf-8-sig")
q = pd.concat([a, b]).drop_duplicates(["document_id", "text"])
q = q.sample(frac=1, random_state=11).reset_index(drop=True)
q["prev_text"] = q["prev_text"].fillna("").str.slice(-150)
q["nhan_ban"] = ""
q[["document_id", "text", "prev_text", "nhan_ban"]].to_csv(
    "label_relabel.csv", index=False, encoding="utf-8-sig")
print("Đã ghi label_relabel.csv:", len(q), "câu,", q["document_id"].nunique(), "tài liệu")