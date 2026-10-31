import re, gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

assert torch.cuda.is_available(), "Không thấy GPU"
CSV_PATH = "data_segment_cls/segment_classification.csv"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}
BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN, SEED, REPEAT = 256, 11, 3
COLS = ["document_id", "text", "label", "section", "prev_text", "pos", "next_text"]

# ---- dữ liệu gốc (đã gồm 39 nhãn sửa) ----
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
base = df[COLS]

# ---- nhãn mới, chia đôi theo tài liệu ----
ex = pd.concat([
    pd.read_csv("extra_labeled.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False),
    pd.read_csv("extra_labeled2.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False),
]).drop_duplicates(["document_id", "text"]).reset_index(drop=True)
ex["pos"] = ex["pos"].astype(float)
print("trùng tài liệu với dữ liệu gốc:", len(set(ex.document_id) & set(df.document_id)))
docs = sorted(ex.document_id.unique())
np.random.RandomState(SEED).shuffle(docs)
A_docs = set(docs[:len(docs) // 2])
exA = ex[ex.document_id.isin(A_docs)][COLS]
exB = ex[~ex.document_id.isin(A_docs)][COLS].reset_index(drop=True)
print(f"nửa A (train): {len(exA)} câu | nửa B (đánh giá): {len(exB)} câu")

train = pd.concat([base] + [exA] * REPEAT).reset_index(drop=True)

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


def softmax(l):
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def predict(model, tok, sub):
    model.eval()
    texts = [t[:600] for t in sub["text"]]
    ctxs = [context(r.section, r.prev_text, r.pos, r.text, r.next_text) for r in sub.itertuples()]
    out = []
    for s in range(0, len(sub), 32):
        enc = tok(texts[s:s+32], ctxs[s:s+32], truncation="longest_first",
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out.append(softmax(model(**enc).logits.float().cpu().numpy()))
    return np.vstack(out)


yB = exB["label"].map(L2I).values


def report(name, P):
    pred = P.argmax(1)
    print(f"\n===== {name} trên nửa B =====")
    print(f"accuracy {(pred == yB).mean():.3f} | macro-F1 {f1_score(yB, pred, average='macro'):.3f}")
    print(classification_report(yB, pred, labels=range(5), target_names=LABELS, digits=2, zero_division=0))


# ---- mốc so sánh: 3 mô hình cuối cũ (chưa thấy nhãn mới) ----
Ps = []
for name in ("final_s11", "final_s22", "final_s33"):
    t = AutoTokenizer.from_pretrained(f"models/{name}")
    m = AutoModelForSequenceClassification.from_pretrained(f"models/{name}").to("cuda")
    Ps.append(predict(m, t, exB))
    del m; torch.cuda.empty_cache()
report("MÔ HÌNH CŨ (ensemble 3)", np.mean(Ps, axis=0))

# ---- mô hình thử: dữ liệu cũ + nửa A ----
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
ds = Dataset.from_dict({
    "text": train["text"].str.slice(0, 600).tolist(),
    "ctx": [context(r.section, r.prev_text, r.pos, r.text, r.next_text) for r in train.itertuples()],
    "label": [L2I[l] for l in train["label"]],
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


set_seed(SEED)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32).to("cuda")
args = TrainingArguments(
    output_dir="runs/exp_extra_tmp", num_train_epochs=5, learning_rate=3e-5,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,
    warmup_steps=150, weight_decay=0.01, max_grad_norm=1.0,
    bf16=True, fp16=False, tf32=True, eval_strategy="no", save_strategy="no",
    dataloader_num_workers=0, report_to="none", seed=SEED)
trainer = WeightedTrainer(model=model, args=args, train_dataset=ds,
                          processing_class=tok, data_collator=DataCollatorWithPadding(tok))
trainer.train()
report("MÔ HÌNH MỚI (dữ liệu cũ + nửa A, 1 mô hình)", predict(model, tok, exB))
