import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

CSV_PATH = Path("data_segment_cls/segment_classification.csv")
OUT_DIR = Path("models/psai_deberta_ctx_v7")
BASE_MODEL = "microsoft/mdeberta-v3-base"   # EN + VI. Chỉ tiếng Anh: microsoft/deberta-v3-base
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 256
SEED = 42

assert torch.cuda.is_available(), "Không thấy GPU: cài lại torch bản CUDA"
use_bf16 = torch.cuda.is_bf16_supported()

# ---------- đọc và làm sạch dữ liệu ----------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper()))
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)

import re as _re
BULLET_RE = _re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")

df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)
_paths, _prevs, _pos = [], [], []
for _doc, _g in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(_g)
    for k, t in enumerate(_g["text"]):
        s = t.strip()
        m = _re.match(r"^(#+)\s*(.*)$", s)
        if m:
            lvl = len(m.group(1))
            stack = stack[:lvl - 1] + [m.group(2).strip()]
        _paths.append(" > ".join(stack))
        _prevs.append(prev)
        _pos.append(k / max(n - 1, 1))
        prev = s
df["section"], df["prev_text"], df["pos"] = _paths, _prevs, _pos

# bỏ dòng tiêu đề markdown khỏi dữ liệu train (không phải câu thật)
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")

# ---------- chia train/val/test, gom tài liệu trùng vào cùng một tập ----------
def doc_key(g):
    joined = " ".join(sorted(g["text"].str.lower().str.strip()))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()

doc_to_key = {d: doc_key(g) for d, g in df.groupby("document_id")}
key_to_docs = {}
for d, k in doc_to_key.items():
    key_to_docs.setdefault(k, []).append(d)

keys = sorted(key_to_docs)
rng = np.random.RandomState(SEED)
rng.shuffle(keys)
n = len(keys)

def take(ks):
    return {d for k in ks for d in key_to_docs[k]}

train_docs = take(keys[:int(n * 0.8)])
val_docs = take(keys[int(n * 0.8):int(n * 0.9)])
test_docs = take(keys[int(n * 0.9):])
print(f"tài liệu: train={len(train_docs)} val={len(val_docs)} test={len(test_docs)}")

MODAL_RE = _re.compile(r"\b(will|must|should|shall|need to|needs to)\b", _re.I)
IMPER_RE = _re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b",
    _re.I)
TIME_RE = _re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", _re.I)


def context(section, prev, pos, text, nxt):   # PHẢI giống sentence_classifier._context
    bullet = 1 if BULLET_RE.match(text) else 0
    cue = f"{int(bool(MODAL_RE.search(text)))}{int(bool(IMPER_RE.match(text)))}{int(bool(TIME_RE.search(text)))}"
    return (f"Section: {str(section).strip()[-120:]} | Pos: {int(pos * 10)} | "
            f"Bullet: {bullet} | Cue: {cue} | Previous: {str(prev).strip()[-150:]} | "
            f"Next: {str(nxt).strip()[:150]}")

tok = AutoTokenizer.from_pretrained(BASE_MODEL)

def make_ds(subset):
    d = Dataset.from_dict({
        "text": subset["text"].str.slice(0, 600).tolist(),
        "ctx": [context(s, p, q, t, n) for s, p, q, t, n in
                zip(subset["section"], subset["prev_text"], subset["pos"],
                    subset["text"], subset["next_text"])],
        "label": [L2I[l] for l in subset["label"]],
    })
    return d.map(lambda b: tok(b["text"], b["ctx"], truncation="longest_first",
                               max_length=MAX_LEN),
                 batched=True, remove_columns=["text", "ctx"])

train_ds = make_ds(df[df.document_id.isin(train_docs)])
val_ds = make_ds(df[df.document_id.isin(val_docs)])
test_ds = make_ds(df[df.document_id.isin(test_docs)])
print(f"câu: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

# ---------- class weight (xử lý lệch nhãn) ----------
y = np.array(train_ds["label"])
counts = np.maximum(np.bincount(y, minlength=len(LABELS)), 1)
weights = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=weights.to(logits.device, dtype=logits.dtype),
            label_smoothing=0.1)
        return (loss, out) if return_outputs else loss

def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)), label2id=L2I,
    dtype=torch.float32)
model.to("cuda")
print("model device:", next(model.parameters()).device)
args = TrainingArguments(
    output_dir="runs/psai_cls",
    num_train_epochs=6,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=32,
    warmup_steps=150,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True, fp16=False, tf32=True,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="macro_f1",
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to="none", seed=SEED,
)
trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=val_ds, processing_class=tok,
                          data_collator=DataCollatorWithPadding(tok),
                          compute_metrics=metrics)
trainer.train()

# ---------- chọn other_bias trên VAL, báo cáo TEST một lần ----------
def logits_of(ds):
    return trainer.predict(ds).predictions, np.array(ds["label"])

vl, vy = logits_of(val_ds)
oid = L2I["OTHER"]
best_bias, best_f1 = 0.0, -1.0
for b in np.arange(-1.0, 1.01, 0.25):
    l = vl.copy()
    l[:, oid] += b
    f = f1_score(vy, l.argmax(-1), average="macro")
    if f > best_f1:
        best_bias, best_f1 = float(b), f
print("other_bias (chọn trên val):", best_bias, "| macro_f1 val:", round(best_f1, 4))

tl, ty = logits_of(test_ds)
tl[:, oid] += best_bias
print(classification_report(ty, tl.argmax(-1), target_names=LABELS, digits=3))

OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
(OUT_DIR / "inference_config.json").write_text(json.dumps(
    {"labels": LABELS, "other_bias": best_bias, "max_len": MAX_LEN, "ctx_version": 3}, indent=2),
    encoding="utf-8")
print("Đã lưu:", OUT_DIR)