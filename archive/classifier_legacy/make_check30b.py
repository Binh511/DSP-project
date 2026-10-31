import pandas as pd

e = pd.read_csv("extra_labeled2.csv", encoding="utf-8-sig")
parts = [g.sample(min(6, len(g)), random_state=3) for _, g in e.groupby("label")]
s = pd.concat(parts).sample(frac=1, random_state=4)
s["prev_text"] = s["prev_text"].fillna("").str.slice(-150)
s["nhan_ban"] = ""
s[["document_id", "text", "prev_text", "nhan_ban"]].to_csv(
    "check30b.csv", index=False, encoding="utf-8-sig")
print("Đã ghi check30b.csv:", len(s), "dòng")
