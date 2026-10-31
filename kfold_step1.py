import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


CSV_PATH = Path(__file__).resolve().parent / "data_segment_cls" / "segment_classification.csv"

LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
L2I = {label: index for index, label in enumerate(LABELS)}
SEED = 42

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(
        r"^[BI]-",
        "",
        str(x).replace("\\", "").replace("`", "").strip().upper(),
    )
)
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)

paths, prevs, poss = [], [], []
for _, group in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(group)
    for k, text in enumerate(group["text"]):
        stripped = text.strip()
        match = re.match(r"^(#+)\s*(.*)$", stripped)
        if match:
            stack = stack[: len(match.group(1)) - 1] + [match.group(2).strip()]
        paths.append(" > ".join(stack))
        prevs.append(prev)
        poss.append(k / max(n - 1, 1))
        prev = stripped

df["section"], df["prev_text"], df["pos"] = paths, prevs, poss
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")


def doc_key(group):
    content = " ".join(sorted(group["text"].str.lower().str.strip()))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


doc2key = {document_id: doc_key(group) for document_id, group in df.groupby("document_id")}
df["group"] = df["document_id"].map(doc2key)

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
df["fold"] = -1
for fold, (_, test_idx) in enumerate(
    splitter.split(df, df["label"], groups=df["group"])
):
    df.loc[test_idx, "fold"] = fold

print(
    "Tổng:",
    len(df),
    "câu |",
    df["document_id"].nunique(),
    "tài liệu |",
    df["group"].nunique(),
    "nhóm",
)
print("\nSố câu / số tài liệu mỗi fold:")
print(df.groupby("fold").agg(cau=("text", "size"), tai_lieu=("document_id", "nunique")))
print("\nPhân bố nhãn mỗi fold:")
label_counts = pd.crosstab(df["fold"], df["label"]).reindex(columns=LABELS, fill_value=0)
print(label_counts)

assert (df.groupby("group")["fold"].nunique() == 1).all(), (
    "Rò rỉ: có nhóm nằm ở nhiều fold"
)
print("\nOK: không có rò rỉ giữa các fold")


# ======================= BƯỚC 3: TRAIN 5 FOLD =======================
import gc
import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

assert torch.cuda.is_available(), "Không thấy GPU"

BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN = 256

BULLET_RE = re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
MODAL_RE = re.compile(r"\b(will|must|should|shall|need to|needs to)\b", re.I)
IMPER_RE = re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b",
    re.I,
)
TIME_RE = re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", re.I)


def context(section, prev, pos, text, nxt):
    bullet = 1 if BULLET_RE.match(text) else 0
    cue = (
        f"{int(bool(MODAL_RE.search(text)))}"
        f"{int(bool(IMPER_RE.match(text)))}"
        f"{int(bool(TIME_RE.search(text)))}"
    )
    return (
        f"Section: {str(section).strip()[-120:]} | Pos: {int(pos * 10)} | "
        f"Bullet: {bullet} | Cue: {cue} | Previous: {str(prev).strip()[-150:]} | "
        f"Next: {str(nxt).strip()[:150]}"
    )


tok = AutoTokenizer.from_pretrained(BASE_MODEL)


def make_ds(sub):
    dataset = Dataset.from_dict(
        {
            "text": sub["text"].str.slice(0, 600).tolist(),
            "ctx": [
                context(section, prev, pos, text, nxt)
                for section, prev, pos, text, nxt in zip(
                    sub["section"],
                    sub["prev_text"],
                    sub["pos"],
                    sub["text"],
                    sub["next_text"],
                )
            ],
            "label": [L2I[label] for label in sub["label"]],
        }
    )
    return dataset.map(
        lambda batch: tok(
            batch["text"],
            batch["ctx"],
            truncation="longest_first",
            max_length=MAX_LEN,
        ),
        batched=True,
        remove_columns=["text", "ctx"],
    )


class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=weights.to(logits.device),
            label_smoothing=0.1,
        )
        return (loss, outputs) if return_outputs else loss


use_bf16 = torch.cuda.is_bf16_supported()

def softmax(logits):
    exponentials = np.exp(logits - logits.max(1, keepdims=True))
    return exponentials / exponentials.sum(1, keepdims=True)


val_P, val_Y, test_P, test_Y = [], [], [], []
fold_f1 = []
project_dir = Path(__file__).resolve().parent

for test_fold in range(5):
    val_fold = (test_fold + 1) % 5
    print(f"\n########## FOLD {test_fold} (val = fold {val_fold}) ##########")

    train_ds = make_ds(df[~df.fold.isin([test_fold, val_fold])])
    val_ds = make_ds(df[df.fold == val_fold])
    test_ds = make_ds(df[df.fold == test_fold])
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    y = np.array(train_ds["label"])
    counts = np.maximum(np.bincount(y, minlength=len(LABELS)), 1)
    weights = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=L2I,
        dtype=torch.float32,
    )
    model.to("cuda")

    args = TrainingArguments(
        output_dir=str(project_dir / "runs" / "kfold_tmp"),
        num_train_epochs=5,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=32,
        warmup_steps=150,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        eval_strategy="no",
        save_strategy="no",
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()

    vp = softmax(trainer.predict(val_ds).predictions)
    vy = np.array(val_ds["label"])
    tp = softmax(trainer.predict(test_ds).predictions)
    ty = np.array(test_ds["label"])
    val_P.append(vp)
    val_Y.append(vy)
    test_P.append(tp)
    test_Y.append(ty)

    f1 = f1_score(ty, tp.argmax(1), average="macro")
    fold_f1.append(f1)
    print(f">>> fold {test_fold}: macro-F1 test = {f1:.4f}")

    output_dir = project_dir / "models" / f"kfold_f{test_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    (output_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "other_bias": 0.0,
                "max_len": MAX_LEN,
                "ctx_version": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

print("\n================ TỔNG KẾT 5 FOLD ================")
print("macro-F1 từng fold:", [round(score, 4) for score in fold_f1])
print(f"macro-F1 trung bình = {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

print("\nF1 từng nhãn (trung bình ± độ lệch chuẩn qua 5 fold):")
for label in LABELS:
    label_id = L2I[label]
    scores = [
        f1_score(targets == label_id, probabilities.argmax(1) == label_id)
        for probabilities, targets in zip(test_P, test_Y)
    ]
    print(f"  {label:16s} {np.mean(scores):.3f} ± {np.std(scores):.3f}")

VP, VY = np.vstack(val_P), np.concatenate(val_Y)
TP, TY = np.vstack(test_P), np.concatenate(test_Y)
print("\nP/R gộp cả 5 fold: VAL (dùng để chọn ngưỡng) | TEST (kiểm tra)")
for label in LABELS[:4]:
    label_id = L2I[label]
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        row = []
        for probabilities, targets in ((VP, VY), (TP, TY)):
            kept = (probabilities.argmax(1) == label_id) & (
                probabilities.max(1) >= threshold
            )
            hit = (kept & (targets == label_id)).sum()
            precision = hit / kept.sum() if kept.sum() else 0
            recall = hit / max((targets == label_id).sum(), 1)
            row.append(f"{precision:.2f}/{recall:.2f}")
        print(
            f"{label:16s} thr={threshold:.1f} | "
            f"val {row[0]} | test {row[1]}"
        )
