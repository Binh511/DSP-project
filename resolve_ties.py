import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sentence_classifier as sc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV = Path("data_segment_cls/segment_classification.csv")
BACKUP = CSV.with_name("segment_classification.before_ties.csv")
if not CSV.is_file():
    raise FileNotFoundError(f"Classification dataset not found: {CSV}")
if not BACKUP.exists():
    shutil.copy2(CSV, BACKUP)
    print("Đã sao lưu:", BACKUP)

df = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
orig_cols = list(df.columns)
required_cols = {"document_id", "sentence_id", "text", "label", "source_line"}
missing_cols = required_cols.difference(orig_cols)
if missing_cols:
    raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing_cols))}")

df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)

paths, prevs, poss = [], [], []
for _, group in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(group)
    for k, text in enumerate(group["text"]):
        stripped = text.strip()
        match = re.match(r"^(#+)\s*(.*)$", stripped)
        if match:
            stack = stack[:len(match.group(1)) - 1] + [match.group(2).strip()]
        paths.append(" > ".join(stack))
        prevs.append(prev)
        poss.append(k / max(n - 1, 1))
        prev = stripped

df["_section"], df["_prev"], df["_pos"] = paths, prevs, poss
df["_next"] = df.groupby("document_id")["text"].shift(-1).fillna("")
df["_norm"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

candidates = df[
    (df["_norm"].str.len() >= 25)
    & ~df["text"].str.strip().str.startswith("#")
]
groups = [group for _, group in candidates.groupby("_norm") if group["label"].nunique() > 1]
print("Số nhóm trùng khác nhãn:", len(groups))

# Resolve against the v5 model, independent of the classifier's default model.
sc.MODEL_DIR = Path(__file__).parent / "models" / "psai_deberta_ctx_v5"
sc._load()
labels = sc._config["labels"]


def probs(rows):
    results = []
    for start in range(0, len(rows), 32):
        batch = rows.iloc[start:start + 32]
        records = batch.to_dict("records")
        contexts = [
            sc._context(row["_section"], row["_prev"], row["_pos"], row["text"], row["_next"])
            for row in records
        ]
        enc = sc._tokenizer(
            batch["text"].tolist(),
            contexts,
            truncation=True,
            max_length=sc._config.get("max_len", 192),
            padding=True,
            return_tensors="pt",
        ).to(sc._device)
        with torch.no_grad():
            logits = sc._model(**enc).logits.float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        results.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    return np.vstack(results)


PLAN = re.compile(
    r"^[\s\-●*]*(?:\d+[.)]+\s*)?(?:Implement|Add|Create|Fix|Design|Build|Develop|"
    r"Development of|Refactor|Write|Update|Provide|Establish|Finalize|Finalise|Integrate)\b")

tied_indices = [index for group in groups for index in group.index]
tied_probabilities = probs(df.loc[tied_indices]) if tied_indices else np.empty((0, len(labels)))
prob_by_index = dict(zip(tied_indices, tied_probabilities))

report = []
for group in groups:
    candidates_for_group = sorted(group["label"].unique())
    text = group["text"].iloc[0].strip()
    if "ACTIVITY" in candidates_for_group and PLAN.match(text):
        new_label, method, margin = "ACTIVITY", "rule_plan", 1.0
    else:
        mean_probability = np.array(
            [prob_by_index[index] for index in group.index]
        ).mean(axis=0)
        candidate_probabilities = {
            label: mean_probability[labels.index(label)]
            for label in candidates_for_group
        }
        ranked = sorted(candidate_probabilities.items(), key=lambda item: -item[1])
        new_label = ranked[0][0]
        margin = ranked[0][1] - ranked[1][1]
        method = "model" if margin >= 0.15 else "review"

    for index, row in group.iterrows():
        report.append({
            "document_id": row["document_id"],
            "sentence_id": row["sentence_id"],
            "old": row["label"],
            "new": new_label,
            "method": method,
            "margin": round(float(margin), 3),
            "text": row["text"][:200],
        })
        df.at[index, "label"] = new_label

df[orig_cols].to_csv(CSV, index=False, encoding="utf-8-sig")
report_df = pd.DataFrame(
    report, columns=["document_id", "sentence_id", "old", "new", "method", "margin", "text"]
)
report_df.to_csv("ties_resolved.csv", index=False, encoding="utf-8-sig")
print(report_df.groupby("method").size())
print("Số dòng đổi nhãn:", int((report_df["old"] != report_df["new"]).sum()))
print(df["label"].value_counts())
