import re, sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
LABELS = {"OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"}


def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


q = pd.read_csv("label_queue2.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
q["nhan_moi"] = q["nhan_moi"].map(clean)
bad = q[~q["nhan_moi"].isin(LABELS)]
if len(bad):
    print("Có nhãn trống hoặc không hợp lệ:", len(bad), "/", len(q))
    print(bad[["document_id", "nhan_moi", "text"]].head(10).to_string())
    raise SystemExit("Dừng, chưa ghi gì.")

cand = pd.read_csv("pseudo_candidates.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
cand["text"] = cand["text"].str.strip()
q["text"] = q["text"].str.strip()
ctx = cand.drop_duplicates(["document_id", "text"])[
    ["document_id", "text", "section", "prev_text", "pos", "next_text"]]

m = q.drop(columns=["prev_text"]).merge(ctx, on=["document_id", "text"], how="left")
miss = m[m["pos"].isna()]
if len(miss):
    print("Không tìm thấy ngữ cảnh cho", len(miss), "câu:")
    print(miss[["document_id", "text"]].head(10).to_string())
    raise SystemExit("Dừng, chưa ghi gì.")

m["pos"] = m["pos"].astype(float)
m["diem"] = m["diem"].astype(float)
m["dung"] = m["mo_hinh_doan"] == m["nhan_moi"]

print(f"Số câu: {len(m)} | mô hình đúng: {m['dung'].mean():.1%}")
print("\nĐộ chính xác thật theo nhãn mô hình đoán:")
print(m.groupby("mo_hinh_doan")["dung"].agg(so_cau="size", chinh_xac="mean").round(2))

print("\nTheo dải điểm (gộp cả 4 nhãn):")
m["dai"] = pd.cut(m["diem"], [0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
print(m.groupby("dai", observed=True)["dung"].agg(so_cau="size", chinh_xac="mean").round(2))

print("\nDòng = mô hình đoán, cột = nhãn của bạn:")
print(pd.crosstab(m["mo_hinh_doan"], m["nhan_moi"]))

out = m.rename(columns={"nhan_moi": "label"})[
    ["document_id", "text", "label", "section", "prev_text", "pos", "next_text"]]
out.to_csv("extra_labeled2.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi extra_labeled2.csv:", len(out), "câu,", out["document_id"].nunique(), "tài liệu")
