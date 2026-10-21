import re
import shutil
from pathlib import Path

import pandas as pd

CSV = Path("data_segment_cls/segment_classification.csv")
BACKUP = CSV.with_name("segment_classification.backup.csv")
LABELS = {"OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"}

if not BACKUP.exists():
    shutil.copy(CSV, BACKUP)
    print("Đã sao lưu:", BACKUP)

df = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def clean(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


df["label"] = df["label"].map(clean)
df = df[df["label"].isin(LABELS) & (df["text"].str.strip() != "")].reset_index(drop=True)

# ---------- quy tắc ----------
SQL = re.compile(r"SELECT\s|FROM public|\(\)\s+function", re.I)
DOCKER = re.compile(r"^[-\s]*\d+\.\s*(Base Image|File Copy|Packages|Ports|ENTRYPOINT)", re.I)
TITLE = re.compile(
    r"^(Maintenance|Plan for \d{4}|Move suggestions|Code: .+|Table \d+\.\d+:.*|"
    r"\d+W: Release|MVP \d+ Release|Mid Term Evaluation\.?)$", re.I)
NARR = re.compile(
    r"^[\s\-●*]*(?:I|We)\s+(?:have\s+|had\s+)?(?:also\s+|already\s+|successfully\s+)?"
    r"(?:started|read|dug|explored|studied|built|tested|learned|worked|managed|made|wrote)\b", re.I)
PLAN = re.compile(
    r"^[\s\-●*]*(?:\d+[.)]+\s*)?(?:Implement|Add|Create|Fix|Design|Build|Develop|"
    r"Development of|Refactor|Write|Update|Provide|Establish|Finalize|Finalise|Integrate)\b")

changes = []


def setlabel(i, new, rule):
    old = df.at[i, "label"]
    if old != new:
        changes.append({
            "document_id": df.at[i, "document_id"], "sentence_id": df.at[i, "sentence_id"],
            "old": old, "new": new, "rule": rule, "text": df.at[i, "text"][:200]})
        df.at[i, "label"] = new


for i in df.index:
    text = df.at[i, "text"].strip()
    lab = df.at[i, "label"]
    doc = df.at[i, "document_id"]
    sid = str(df.at[i, "sentence_id"])

    if lab == "OTHER":
        if PLAN.match(text) and not re.search(r"\b(I|We)\b.*\b(have|had)\b", text):
            setlabel(i, "ACTIVITY", "R5_plan_item")
        continue
    if SQL.search(text):
        setlabel(i, "OTHER", "R1_sql_doc")
    elif DOCKER.match(text):
        setlabel(i, "OTHER", "R1_docker_doc")
    elif TITLE.match(text) and lab in ("ACTIVITY", "EXPECTED_RESULT"):
        setlabel(i, "OTHER", "R2_title")
    elif NARR.match(text) and not re.search(r"\bwill\b", text):
        setlabel(i, "OTHER", "R3_narration")
    elif doc == "PSAI064" and sid.startswith("L") and not re.search(r"\bwill\b", text):
        setlabel(i, "OTHER", "R4_paper_content")
    elif lab == "EXPECTED_RESULT" and PLAN.match(text):
        setlabel(i, "ACTIVITY", "R5_plan_item")

# ---------- R6: cùng một câu nhưng nhãn khác nhau giữa các tài liệu ----------
df["_norm"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
ties = []
for norm, g in df[df["_norm"].str.len() >= 25].groupby("_norm"):
    if g["label"].nunique() > 1:
        counts = g["label"].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            ties.append(g[["document_id", "sentence_id", "label", "text"]])
        else:
            for i in g.index:
                setlabel(i, counts.index[0], "R6_duplicate_majority")

df = df.drop(columns=["_norm"])
df.to_csv(CSV, index=False, encoding="utf-8-sig")
pd.DataFrame(changes).to_csv("label_changes.csv", index=False, encoding="utf-8-sig")
if ties:
    pd.concat(ties).to_csv("label_ties.csv", index=False, encoding="utf-8-sig")

print("Tổng số thay đổi:", len(changes))
print(pd.DataFrame(changes).groupby(["rule", "old", "new"]).size() if changes else "không có")
print("Số nhóm câu trùng bị hòa (cần xem tay):", len(ties))
print("Phân bố nhãn mới:")
print(df["label"].value_counts())