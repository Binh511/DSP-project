import re, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]


def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


mine = pd.read_csv("check30.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
mine["nhan_ban"] = mine["nhan_ban"].map(clean)
bad = mine[~mine["nhan_ban"].isin(LABELS)]
if len(bad):
    print("Có nhãn trống hoặc không hợp lệ:", len(bad), "/", len(mine))
    raise SystemExit("Dừng, hãy điền nốt cột nhan_ban rồi chạy lại.")

agent = pd.read_csv("label_queue3.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
agent["nhan_agent"] = agent["nhan_moi"].map(clean)
m = mine.merge(agent[["document_id", "text", "nhan_agent"]], on=["document_id", "text"], how="left")
if m["nhan_agent"].isna().any():
    raise SystemExit("Có dòng không khớp với label_queue3.csv (text bị sửa?), dừng.")

agree = (m["nhan_ban"] == m["nhan_agent"]).mean()
kappa = cohen_kappa_score(m["nhan_ban"], m["nhan_agent"])
print(f"Số dòng: {len(m)} | khớp: {int(agree * len(m))}/{len(m)} = {agree:.1%} | Cohen's kappa = {kappa:.2f}")

print("\nDòng = nhãn của bạn, cột = nhãn của agent:")
print(pd.crosstab(m["nhan_ban"], m["nhan_agent"]).reindex(index=LABELS, columns=LABELS).fillna(0).astype(int))

diff = m[m["nhan_ban"] != m["nhan_agent"]]
print(f"\n===== {len(diff)} dòng bất đồng =====")
for r in diff.itertuples():
    print(f"- bạn={r.nhan_ban} | agent={r.nhan_agent} | {r.document_id}: {r.text[:120]}")

print("\nKết luận gợi ý:",
      "nhãn agent đủ tin để dùng." if agree >= 0.85 else
      "khớp thấp hơn 85%, chưa nên dùng nhãn agent để đo hoặc train.")
