import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score

import sentence_classifier as sc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
MODELS = ["psai_deberta_ctx_v5", "psai_deberta_ctx_v6", "psai_deberta_ctx_v7"]

df = pd.read_csv(ROOT / "data_segment_cls" / "segment_classification.csv", encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper()))
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)

paths, prevs, poss = [], [], []
for _, g in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(g)
    for k, text in enumerate(g["text"]):
        sentence = text.strip()
        match = re.match(r"^(#+)\s*(.*)$", sentence)
        if match:
            stack = stack[:len(match.group(1)) - 1] + [match.group(2).strip()]
        paths.append(" > ".join(stack))
        prevs.append(prev)
        poss.append(k / max(n - 1, 1))
        prev = sentence
df["section"], df["prev_text"], df["pos"] = paths, prevs, poss
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")


def doc_key(group):
    content = " ".join(sorted(group["text"].str.lower().str.strip()))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


d2k = {doc: doc_key(group) for doc, group in df.groupby("document_id")}
k2d = {}
for doc, key in d2k.items():
    k2d.setdefault(key, []).append(doc)
keys = sorted(k2d)
np.random.RandomState(42).shuffle(keys)
n = len(keys)
test_docs = {doc for key in keys[int(n * 0.9):] for doc in k2d[key]}
t = df[df.document_id.isin(test_docs)].reset_index(drop=True)
y = t["label"].map(LABELS.index).values
print("test:", len(t), "câu,", len(test_docs), "tài liệu")


def probs(model_name):
    sc.MODEL_DIR = ROOT / "models" / model_name
    sc._model = None
    sc._load()
    if sc._config["labels"] != LABELS:
        raise ValueError(f"Unexpected label order in {model_name}: {sc._config['labels']}")

    batches = []
    for start in range(0, len(t), 32):
        batch = t.iloc[start:start + 32]
        contexts = [
            sc._context(row.section, row.prev_text, row.pos, row.text, row.next_text)
            for row in batch.itertuples()
        ]
        enc = sc._tokenizer(
            list(batch["text"]),
            contexts,
            truncation=True,
            max_length=sc._config.get("max_len", 192),
            padding=True,
            return_tensors="pt",
        ).to(sc._device)
        with torch.no_grad():
            logits = sc._model(**enc).logits.float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        batches.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    return np.vstack(batches)


allp = {}
for model_name in MODELS:
    allp[model_name] = probs(model_name)
    print(model_name, "macro-F1:", round(f1_score(y, allp[model_name].argmax(1), average="macro"), 4))

ensemble = np.mean(list(allp.values()), axis=0)
print("ENSEMBLE macro-F1:", round(f1_score(y, ensemble.argmax(1), average="macro"), 4))
print(classification_report(y, ensemble.argmax(1), target_names=LABELS, digits=3))
