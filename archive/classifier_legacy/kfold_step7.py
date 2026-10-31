import re, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

CSV_PATH = "data_segment_cls/segment_classification.csv"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- dữ liệu và fold: GIỐNG HỆT bước 1 ----
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

def doc_key(g):
    return hashlib.md5(" ".join(sorted(g["text"].str.lower().str.strip())).encode("utf-8")).hexdigest()
df["group"] = df["document_id"].map({d: doc_key(g) for d, g in df.groupby("document_id")})

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
df["fold"] = -1
for f, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
    df.loc[idx, "fold"] = f

# ---- ngữ cảnh: giống lúc train ----
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

# ---- chấm mỗi câu bằng mô hình CHƯA THẤY nó ----
df["pred"], df["score"] = "", 0.0
for f in range(5):
    md = Path(f"models/kfold_f{f}")
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md).to(device).eval()
    sub = df[df.fold == f]
    texts = [t[:600] for t in sub["text"]]
    ctxs = [context(r.section, r.prev_text, r.pos, r.text, r.next_text) for r in sub.itertuples()]
    out = []
    for s in range(0, len(sub), 32):
        enc = tok(texts[s:s+32], ctxs[s:s+32], truncation="longest_first",
                  max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(softmax(model(**enc).logits.float().cpu().numpy()))
    P = np.vstack(out)
    df.loc[sub.index, "pred"] = [LABELS[i] for i in P.argmax(1)]
    df.loc[sub.index, "score"] = P.max(1)
    del model; torch.cuda.empty_cache()
    print("xong fold", f)

# ---- chọn câu cần xem lại ----
q = df[df["text"].str.count(r"\|") < 2].copy()
N = 25
groups = [
    ("OTHER_do_nghi_la_EXPECTED_RESULT", (q.label == "OTHER") & (q.pred == "EXPECTED_RESULT") & (q.score >= 0.85)),
    ("OTHER_do_nghi_la_REQUIREMENT",     (q.label == "OTHER") & (q.pred == "REQUIREMENT") & (q.score >= 0.85)),
    ("OTHER_do_nghi_la_ACTIVITY",        (q.label == "OTHER") & (q.pred == "ACTIVITY") & (q.score >= 0.85)),
    ("OTHER_do_nghi_la_OBJECTIVE",       (q.label == "OTHER") & (q.pred == "OBJECTIVE") & (q.score >= 0.85)),
    ("EXPECTED_RESULT_do_nghi_khac",     (q.label == "EXPECTED_RESULT") & (q.pred != "EXPECTED_RESULT") & (q.score >= 0.6)),
    ("REQUIREMENT_do_nghi_khac",         (q.label == "REQUIREMENT") & (q.pred != "REQUIREMENT") & (q.score >= 0.6)),
]
parts = []
for name, mask in groups:
    sub = q[mask]
    pick = sub.sample(min(N, len(sub)), random_state=0).copy()
    pick["nhom"] = name
    parts.append(pick)
    print(f"{name:36s} có {len(sub):4d} câu, lấy {len(pick)}")

out = pd.concat(parts)
out = out.rename(columns={"label": "nhan_tay", "pred": "mo_hinh_doan", "score": "diem"})
out["nhan_moi"] = ""
cols = ["document_id", "sentence_id", "nhom", "nhan_tay", "mo_hinh_doan", "diem",
        "text", "prev_text", "nhan_moi"]
out["prev_text"] = out["prev_text"].str.slice(-150)
out[cols].to_csv("review_queue.csv", index=False, encoding="utf-8-sig")
print("\nĐã ghi review_queue.csv:", len(out), "câu")
