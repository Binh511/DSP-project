import re
import pandas as pd
import sentence_classifier

LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
df = pd.read_csv("data_segment_cls/segment_classification.csv", encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper()))
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)

preds = sentence_classifier.classify([{"text": t} for t in df["text"]])
df["pred"] = [p["label"] for p in preds]
df["score"] = [p["score"] for p in preds]

bad = df[(df["pred"] != df["label"]) & (df["score"] >= 0.9)]
print("bất đồng chắc chắn:", len(bad), "/", len(df))
print(pd.crosstab(bad["label"], bad["pred"]))
bad.sort_values("score", ascending=False).head(200).to_csv(
    "label_conflicts.csv", index=False, encoding="utf-8-sig")
print("Đã ghi label_conflicts.csv")