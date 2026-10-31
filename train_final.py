import re, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

assert torch.cuda.is_available(), "Không thấy GPU"

CSV_PATH = "data_segment_cls/segment_classification.csv"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN = 256
SEEDS = [11, 22, 33]

# ---- dữ liệu: giống hệt các bước trước ----
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["label"] = df["label"].map(
    lambda x: re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper()))
df = df[df["label"].isin(LABELS)].dropna(subset=["text"])
df = df[df["text"].str.strip() != ""].reset_index(drop=True)
df["source_line"] = df["source_line"].astype(int)
df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)

paths, prevs, poss = [], [], []
for _, g in df.groupby("document_id", sort=False):
    stack, prev, n = [], "", len(g)
    for k, t in enumerate(g["text"]):
        s = t.strip()
        m = re.match(r"^(#+)\s*(.*)$", s)
        if m:
            stack = stack[:len(m.group(1)) - 1] + [m.group(2).strip()]
        paths.append(" > ".join(stack)); prevs.append(prev); poss.append(k / max(n - 1, 1))
        prev = s
df["section"], df["prev_text"], df["pos"] = paths, prevs, poss
df = df[~df["text"].str.strip().str.startswith("#")].reset_index(drop=True)
df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")
print("Số câu train:", len(df), "| số tài liệu:", df["document_id"].nunique())

BULLET_RE = re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
MODAL_RE = re.compile(r"\b(will|must|should|shall|need to|needs to)\b", re.I)
IMPER_RE = re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b", re.I)
TIME_RE = re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", re.I)

def context(section, prev, pos, text, nxt):
    bullet = 1 if BULLET_RE.match(text) else 0
    cue = f"{int(bool(MODAL_RE.search(text)))}{int(bool(IMPER_RE.match(text)))}{int(bool(TIME_RE.search(text)))}"
    return (f"Section: {str(section).strip()[-120:]} | Pos: {int(pos * 10)} | "
            f"Bullet: {bullet} | Cue: {cue} | Previous: {str(prev).strip()[-150:]} | "
            f"Next: {str(nxt).strip()[:150]}")

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
ds = Dataset.from_dict({
    "text": df["text"].str.slice(0, 600).tolist(),
    "ctx": [context(s, p, q, t, n) for s, p, q, t, n in
            zip(df["section"], df["prev_text"], df["pos"], df["text"], df["next_text"])],
    "label": [L2I[l] for l in df["label"]],
}).map(lambda b: tok(b["text"], b["ctx"], truncation="longest_first", max_length=MAX_LEN),
       batched=True, remove_columns=["text", "ctx"])

y = np.array(ds["label"])
counts = np.maximum(np.bincount(y, minlength=len(LABELS)), 1)
weights = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=weights.to(logits.device), label_smoothing=0.1)
        return (loss, out) if return_outputs else loss

for seed in SEEDS:
    print(f"\n########## MÔ HÌNH CUỐI, seed {seed} ##########")
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32)
    model.to("cuda")

    args = TrainingArguments(
        output_dir="runs/final_tmp",
        num_train_epochs=5, learning_rate=3e-5,
        per_device_train_batch_size=8, gradient_accumulation_steps=2,
        warmup_steps=150, weight_decay=0.01, max_grad_norm=1.0,
        bf16=True, fp16=False, tf32=True,
        eval_strategy="no", save_strategy="no",
        dataloader_num_workers=0, report_to="none", seed=seed,
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=ds,
                              processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    out = Path(f"models/final_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    (out / "inference_config.json").write_text(json.dumps(
        {"labels": LABELS, "other_bias": 0.0, "max_len": MAX_LEN, "ctx_version": 3}, indent=2),
        encoding="utf-8")
    print("Đã lưu:", out)

    del model, trainer
    gc.collect(); torch.cuda.empty_cache()

print("\nXONG 3 mô hình cuối.")
