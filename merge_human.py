import re, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]

def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())

h = pd.read_csv("label_relabel.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
h["nhan_ban"] = h["nhan_ban"].map(clean)
print(f"Tổng dòng: {len(h)} | đã gán: {(h['nhan_ban'] != '').sum()}")

h = h[h["nhan_ban"] != ""].copy()
bad = h[~h["nhan_ban"].isin(LABELS)]
if len(bad):
    print("Có nhãn không hợp lệ:", len(bad))
    print(bad[["document_id", "nhan_ban", "text"]].head(10).to_string())
    raise SystemExit("Dừng, chưa ghi gì.")
h["text"] = h["text"].str.strip()

agent = pd.concat([
    pd.read_csv("extra_labeled.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False),
    pd.read_csv("extra_labeled2.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False),
])
agent["text"] = agent["text"].str.strip()
agent["nhan_agent"] = agent["label"].map(clean)
agent = agent.drop_duplicates(["document_id", "text"])

cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
cand["text"] = cand["text"].str.strip()
ctx = cand.drop_duplicates(["document_id", "text"])[
    ["document_id", "text", "section", "prev_text", "pos", "next_text"]]

m = h.drop(columns=["prev_text"]).merge(ctx, on=["document_id", "text"], how="left") \
     .merge(agent[["document_id", "text", "nhan_agent"]], on=["document_id", "text"], how="left")
miss = m[m["pos"].isna()]
if len(miss):
    print("Không tìm thấy ngữ cảnh cho", len(miss), "câu:")
    print(miss[["document_id", "text"]].head(10).to_string())
    raise SystemExit("Dừng, chưa ghi gì.")

agree = (m["nhan_ban"] == m["nhan_agent"]).mean()
kappa = cohen_kappa_score(m["nhan_ban"], m["nhan_agent"])
print(f"\nĐộ khớp bạn vs agent trên {len(m)} câu: {agree:.1%} | kappa = {kappa:.2f}")

print("\nPhân bố nhãn của bạn:")
print(m["nhan_ban"].value_counts())
print("\nPhân bố nhãn của agent trên cùng các câu:")
print(m["nhan_agent"].value_counts())

print("\nDòng = nhãn của bạn, cột = nhãn của agent:")
print(pd.crosstab(m["nhan_ban"], m["nhan_agent"]).reindex(index=LABELS, columns=LABELS).fillna(0).astype(int))

out = m.rename(columns={"nhan_ban": "label"})[
    ["document_id", "text", "label", "section", "prev_text", "pos", "next_text"]]
out.to_csv("extra_human.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi extra_human.csv:", len(out), "câu,", out["document_id"].nunique(), "tài liệu")