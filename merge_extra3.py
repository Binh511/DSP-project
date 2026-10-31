import re, sys
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.stdout.reconfigure(encoding="utf-8")
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
THR = {"OBJECTIVE": 0.70, "REQUIREMENT": 0.60, "ACTIVITY": 0.70, "EXPECTED_RESULT": 0.70}


def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


q = pd.read_csv("label_queue3.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
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
    ["document_id", "text", "pred", "score", "section", "prev_text", "pos", "next_text"]]
m = q.drop(columns=["prev_text"]).merge(ctx, on=["document_id", "text"], how="left")
miss = m[m["pos"].isna()]
if len(miss):
    print("Không tìm thấy ngữ cảnh cho", len(miss), "câu:")
    print(miss[["document_id", "text"]].head(10).to_string())
    raise SystemExit("Dừng, chưa ghi gì.")

m["pos"] = m["pos"].astype(float)
m["score"] = m["score"].astype(float)
m["gold"] = m["nhan_moi"]


def apply_thr(row, shift=0.0, thr=THR):
    if row["pred"] == "OTHER":
        return "OTHER"
    return row["pred"] if row["score"] >= thr[row["pred"]] + shift else "OTHER"


print(f"Số câu: {len(m)} | {m['document_id'].nunique()} tài liệu")
print("\nPhân bố nhãn bạn gán (mẫu ngẫu nhiên, phản ánh tỉ lệ thật trên tài liệu mới):")
print(m["gold"].value_counts())

m["final"] = m.apply(apply_thr, axis=1)
print("\n===== ĐẦU RA THẬT (argmax + ngưỡng đang dùng) =====")
print(f"accuracy {(m.final == m.gold).mean():.3f} | macro-F1 {f1_score(m.gold, m.final, labels=LABELS, average='macro', zero_division=0):.3f}")
print(classification_report(m.gold, m.final, labels=LABELS, digits=2, zero_division=0))

print("===== THỬ DỊCH NGƯỠNG (cộng thêm vào cả 4 ngưỡng hiện tại) =====")
print("dịch   accuracy  macro-F1  P_ACT  R_ACT  P_REQ  R_REQ  P_ER   R_ER   P_OBJ  R_OBJ")
for sh in (-0.1, 0.0, 0.1, 0.15, 0.2):
    f = m.apply(lambda r: apply_thr(r, sh), axis=1)
    row = [f"{sh:+.2f}", f"{(f == m.gold).mean():.3f}",
           f"{f1_score(m.gold, f, labels=LABELS, average='macro', zero_division=0):.3f}"]
    for lab in ("ACTIVITY", "REQUIREMENT", "EXPECTED_RESULT", "OBJECTIVE"):
        kept = (f == lab).sum(); hit = ((f == lab) & (m.gold == lab)).sum(); g = (m.gold == lab).sum()
        row += [f"{hit / kept if kept else 0:.2f}", f"{hit / g if g else 0:.2f}"]
    print("  ".join(row))

out = m.rename(columns={"gold": "label"})[
    ["document_id", "text", "label", "section", "prev_text", "pos", "next_text"]]
out.to_csv("extra_labeled3.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi extra_labeled3.csv:", len(out), "câu")
