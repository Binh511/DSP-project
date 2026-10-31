import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CSV = Path("data_segment_cls/segment_classification.csv")
REVIEW = Path("review_queue.csv")
BACKUP = CSV.with_name("segment_classification.before_review.csv")
LABELS = {"OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"}


def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


rv = pd.read_csv(REVIEW, encoding="utf-8-sig", dtype=str, keep_default_na=False)
rv["nhan_moi"] = rv["nhan_moi"].map(clean)
filled = rv[rv["nhan_moi"] != ""].copy()
print(f"Tổng dòng review: {len(rv)} | dòng đã điền nhan_moi: {len(filled)}")

invalid = filled[~filled["nhan_moi"].isin(LABELS)]
if len(invalid):
    print("\nCÓ NHÃN KHÔNG HỢP LỆ, sửa trong Excel rồi chạy lại:")
    print(invalid[["document_id", "sentence_id", "nhan_moi", "text"]].to_string(index=False))
    raise SystemExit(1)

same = filled[filled["nhan_moi"] == filled["nhan_tay"].map(clean)]
if len(same):
    print(f"Bỏ qua {len(same)} dòng nhãn mới trùng nhãn cũ.")
    filled = filled.drop(same.index)

df = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
df["_key"] = df["document_id"].str.strip() + "||" + df["sentence_id"].str.strip()
duplicate_keys = df["_key"].duplicated(keep=False)
if duplicate_keys.any():
    print("Cảnh báo: có", int(df["_key"].duplicated().sum()), "khoá (document_id, sentence_id) bị trùng")

filled["_key"] = filled["document_id"].str.strip() + "||" + filled["sentence_id"].str.strip()
if filled["_key"].duplicated().any():
    raise SystemExit("Dừng: khóa (document_id, sentence_id) bị trùng trong review; chưa ghi gì.")

missing = filled[~filled["_key"].isin(df["_key"])]
if len(missing):
    print(f"\nKhông tìm thấy {len(missing)} dòng trong CSV gốc (có thể Excel đã đổi sentence_id):")
    print(missing[["document_id", "sentence_id", "text"]].head(10).to_string(index=False))
    raise SystemExit("Dừng, chưa ghi gì.")

if duplicate_keys.any() and filled["_key"].isin(df.loc[duplicate_keys, "_key"]).any():
    raise SystemExit("Dừng: có khóa cần sửa bị trùng trong CSV gốc; chưa sao lưu hoặc ghi gì.")

if not BACKUP.exists():
    shutil.copy2(CSV, BACKUP)
    print("Đã sao lưu:", BACKUP)

index_by_key = dict(zip(df["_key"], df.index))
changes = []
for _, r in filled.iterrows():
    i = index_by_key[r["_key"]]
    old = clean(df.at[i, "label"])
    changes.append((old, r["nhan_moi"], r["document_id"], df.at[i, "text"][:90]))
    df.at[i, "label"] = r["nhan_moi"]

df = df.drop(columns=["_key"])
df.to_csv(CSV, index=False, encoding="utf-8-sig")

ch = pd.DataFrame(changes, columns=["cu", "moi", "document_id", "text"])
print(f"\nĐã đổi nhãn {len(ch)} câu. Bảng thay đổi (dòng = nhãn cũ, cột = nhãn mới):")
print(pd.crosstab(ch["cu"], ch["moi"]))
print("\nPhân bố nhãn mới của toàn CSV:")
print(df["label"].map(clean).value_counts())
print("\nTài liệu bị sửa nhiều nhất:")
print(ch["document_id"].value_counts().head(5))
