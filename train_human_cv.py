"""PSAI classifier: controlled experiments with manually labeled new proposals.

Commands (run from your DSP_PROJECT directory):
  python train_human_cv.py --mode audit
  python train_human_cv.py --mode cv --folds 0
  python train_human_cv.py --mode cv --folds all
  python train_human_cv.py --mode final

This does not modify existing training data, old checkpoints or production extraction.
Requires the same Hugging Face / PyTorch environment as kfold_step1.py.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
L2I = {name: i for i, name in enumerate(LABELS)}
ROOT = Path(__file__).resolve().parent
BASE_MODEL = "microsoft/mdeberta-v3-base"
MAX_LEN = 256
FOLD_SEED = 42
HUMAN_SPLIT_SEED = 2026
# Targeted repetition: extra review examples have more influence in the two weak categories.
# This is an experimental HYPOTHESIS, not a proven optimal weighting scheme.
REPETITIONS = {
    "OBJECTIVE": 2, "REQUIREMENT": 3, "ACTIVITY": 1,
    "EXPECTED_RESULT": 3, "OTHER": 1,
}
COLS = ["document_id", "text", "label", "section", "prev_text", "pos", "next_text"]

BULLET = re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
MODAL = re.compile(r"\b(will|must|should|shall|need to|needs to)\b", re.I)
IMPER = re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|"
    r"update|integrate|test|deploy|refactor|provide|release|document)\b", re.I)
TIME = re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", re.I)


def clean_label(x):
    return re.sub(r"^[BI]-", "", str(x).replace("\\", "").replace("`", "").strip().upper())


def text_key(x):
    return re.sub(r"\s+", " ", str(x)).casefold().strip()


def context(section, prev, pos, text, nxt):
    """EXACT same ctx_version=3 formatter as kfold_step1.py / train_final.py."""
    bullet = int(bool(BULLET.match(str(text))))
    cue = (f"{int(bool(MODAL.search(str(text))))}"
           f"{int(bool(IMPER.match(str(text))))}"
           f"{int(bool(TIME.search(str(text))))}")
    return (f"Section: {str(section).strip()[-120:]} | Pos: {int(float(pos) * 10)} | "
            f"Bullet: {bullet} | Cue: {cue} | Previous: {str(prev).strip()[-150:]} | "
            f"Next: {str(nxt).strip()[:150]}")


def read_original(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"document_id", "source_line", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing original columns: {sorted(required - set(df.columns))}")
    df["label"] = df["label"].map(clean_label)
    df = df[df.label.isin(LABELS)].dropna(subset=["text"]).copy()
    df = df[df.text.astype(str).str.strip().ne("")].copy()
    df["source_line"] = pd.to_numeric(df["source_line"], errors="raise").astype(int)
    df = df.sort_values(["document_id", "source_line"]).reset_index(drop=True)
    paths, prevs, positions = [], [], []
    for _, g in df.groupby("document_id", sort=False):
        stack, prev, n = [], "", len(g)
        for k, t in enumerate(g.text):
            s = t.strip()
            m = re.match(r"^(#+)\s*(.*)$", s)
            if m:
                stack = stack[:len(m.group(1)) - 1] + [m.group(2).strip()]
            paths.append(" > ".join(stack))
            prevs.append(prev)
            positions.append(k / max(n - 1, 1))
            prev = s
    df["section"], df["prev_text"], df["pos"] = paths, prevs, positions
    df = df[~df.text.str.strip().str.startswith("#")].reset_index(drop=True)
    df["next_text"] = df.groupby("document_id")["text"].shift(-1).fillna("")
    # EXACT same duplicate grouping & StratifiedGroupKFold as original kfold_step1.py
    def doc_key(g):
        return hashlib.md5(" ".join(sorted(g.text.str.lower().str.strip())).encode("utf-8")).hexdigest()
    df["group"] = df.document_id.map({d: doc_key(g) for d, g in df.groupby("document_id")})
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    df["fold"] = -1
    for fold, (_, ix) in enumerate(splitter.split(df, df.label, groups=df.group)):
        df.loc[ix, "fold"] = fold
    assert (df.groupby("group")["fold"].nunique() == 1).all()
    return df


def read_human(path, original):
    human = pd.read_csv(path, encoding="utf-8-sig", dtype={"document_id": str})
    if not set(COLS).issubset(human.columns):
        raise ValueError("extra_human.csv must contain: " + ", ".join(COLS))
    human = human[COLS].copy()
    human["label"] = human.label.map(clean_label)
    if human.label.isna().any() or (~human.label.isin(LABELS)).any():
        raise ValueError("Some human labels are invalid; resolve before training")
    if human[["document_id", "text"]].isna().any().any():
        raise ValueError("Missing document_id or text in extra_human.csv")
    human = human[human.text.str.strip().ne("")].copy()
    human["pos"] = pd.to_numeric(human.pos, errors="coerce")
    if human.pos.isna().any():
        # A single missing 'pos' can be reconstructed only approximately.
        # Use 0.0, and log this case rather than silently passing NaN to ctx_version=3.
        print("WARNING: missing human pos; filling with 0.0:", human.pos.isna().sum())
        human["pos"] = human.pos.fillna(0.0)
    for col in ["section", "prev_text", "next_text"]:
        human[col] = human[col].fillna("").astype(str)
    human["_norm"] = human.text.map(text_key)
    # Human duplicates are never counted multiple times as distinct samples.
    before = len(human)
    human = human.drop_duplicates(["document_id", "_norm"]).reset_index(drop=True)
    print("Human duplicate (doc, text) removed:", before - len(human))
    # New docs must never be in original fold test/val/train.
    overlap_docs = set(human.document_id) & set(original.document_id)
    if overlap_docs:
        print("WARNING: excluding human examples from original labeled docs:", len(overlap_docs))
        human = human[~human.document_id.isin(overlap_docs)].copy()
    orig_texts = set(original.text.map(text_key))
    duplicated = human._norm.isin(orig_texts)
    if duplicated.any():
        print("WARNING: excluding human sentences duplicating original corpus:", duplicated.sum())
        human = human[~duplicated].copy()
    # Prevent identical sentences from crossing the human train/holdout boundary.
    # Group by document first, then remove train duplicates of holdout text.
    human = human.reset_index(drop=True)
    rng = np.random.RandomState(HUMAN_SPLIT_SEED)
    doc_ids = np.array(sorted(human.document_id.unique()))
    rng.shuffle(doc_ids)
    n_holdout = max(1, round(len(doc_ids) * 0.25))
    held_docs = set(doc_ids[:n_holdout])
    human_test = human[human.document_id.isin(held_docs)].copy()
    human_train = human[~human.document_id.isin(held_docs)].copy()
    human_train = human_train[~human_train._norm.isin(set(human_test._norm))].copy()
    assert not (set(human_train.document_id) & set(human_test.document_id))
    assert not (set(human_train._norm) & set(human_test._norm))
    return human_train[COLS].reset_index(drop=True), human_test[COLS].reset_index(drop=True), sorted(held_docs)


def training_examples(human_train, repetitions):
    return pd.concat(
        [human_train[human_train.label.eq(lab)] for lab, n in repetitions.items() for _ in range(n)],
        ignore_index=True,
    )


def metrics(y, probabilities):
    pred = probabilities.argmax(axis=1)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "per_label_f1": {lab: float(f1_score(y == i, pred == i, zero_division=0)) for i, lab in enumerate(LABELS)},
    }


def print_metrics(title, m):
    print(f"{title}: N={m['n']} | Acc={m['accuracy']:.4f} | Macro-F1={m['macro_f1']:.4f}")
    print("  " + " | ".join(f"{lab}: {m['per_label_f1'][lab]:.3f}" for lab in LABELS))


def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def lazy_libs():
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires NVIDIA GPU/CUDA (same as existing kfold_step1.py).")
    return torch, Dataset, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


def make_ds(rows, tokenizer, Dataset):
    if len(rows) == 0:
        raise ValueError("Dataset is empty")
    ctxs = [context(r.section, r.prev_text, r.pos, r.text, r.next_text) for r in rows.itertuples()]
    ds = Dataset.from_dict({
        "text": rows.text.astype(str).str.slice(0, 600).tolist(),
        "ctx": ctxs,
        "label": [L2I[label] for label in rows.label],
    })
    return ds.map(lambda batch: tokenizer(batch["text"], batch["ctx"],
                                           truncation="longest_first", max_length=MAX_LEN),
                  batched=True, remove_columns=["text", "ctx"])


def predict_saved_model(model_dir, rows, torch, AutoModel, AutoTokenizer, batch=32):
    config_path = model_dir / "inference_config.json"
    if not config_path.exists():
        return None
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("ctx_version") != 3 or cfg["labels"] != LABELS:
        raise ValueError(f"Unsupported label order/ctx_version: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to("cuda").eval()
    outs = []
    for start in range(0, len(rows), batch):
        sub = rows.iloc[start:start + batch]
        enc = tokenizer(
            sub.text.str.slice(0, 600).tolist(),
            [context(r.section, r.prev_text, r.pos, r.text, r.next_text) for r in sub.itertuples()],
            truncation="longest_first", max_length=cfg.get("max_len", MAX_LEN),
            padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = model(**enc).logits.float().cpu().numpy()
        logits[:, L2I["OTHER"]] += cfg.get("other_bias", 0.0)
        outs.append(softmax(logits))
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return np.vstack(outs)


def train_model(rows, seed, model_dir, epochs, libs, validation=None):
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {model_dir}")
    from classifier_training import WeightedClassificationTrainer
    torch, Dataset, AutoModel, AutoTokenizer, DataCollator, Trainer, TrainingArguments, set_seed = libs
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = make_ds(rows, tok, Dataset)
    eval_ds = make_ds(validation, tok, Dataset) if validation is not None else None
    counts = np.maximum(np.bincount(np.array(train_ds["label"]), minlength=5), 1)
    weights = torch.tensor(len(train_ds) / (5 * counts), dtype=torch.float32)

    model = AutoModel.from_pretrained(BASE_MODEL, num_labels=5,
                                     id2label=dict(enumerate(LABELS)), label2id=L2I,
                                     dtype=torch.float32).to("cuda")
    bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(ROOT / "runs" / model_dir.name / "checkpoints"),
        num_train_epochs=epochs, learning_rate=3e-5,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        per_device_eval_batch_size=32, warmup_steps=150,
        weight_decay=0.01, max_grad_norm=1.0,
        bf16=bf16, fp16=not bf16, tf32=True,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if eval_ds is not None else "no",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_num_workers=0, report_to="none", seed=seed,
    )
    print("[VRAM-6GB-V2] confirmed train_human_cv.py: batch=1, accumulation=16, "
          "checkpointing=True, sequence_length=256, AdamW optimizer, "
          "explicit per-example class weights", flush=True)
    def compute_metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return {"macro_f1": metrics(prediction.label_ids, logits)["macro_f1"]}

    trainer = WeightedClassificationTrainer(model=model, args=args, train_dataset=train_ds,
                              class_weights=weights,
                              eval_dataset=eval_ds,
                              compute_metrics=compute_metrics if eval_ds is not None else None,
                              processing_class=tok, data_collator=DataCollator(tok))
    trainer.train()
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    trainer.state.save_to_json(str(model_dir / "training_state.json"))
    (model_dir / "inference_config.json").write_text(json.dumps(
        {"labels": LABELS, "other_bias": 0.0, "max_len": MAX_LEN, "ctx_version": 3,
         "text_max_chars": 600, "loss_normalization": "microbatch_mean_trainer_accumulation"},
        indent=2), encoding="utf-8")
    del model, trainer, tok, train_ds
    gc.collect(); torch.cuda.empty_cache()


def report_dataset(original, ht, hv, holdout_docs, out_dir):
    print("Original:", len(original), "sentences,", original.document_id.nunique(),
          "docs,", original.group.nunique(), "duplicate-content groups")
    print("Original class counts:", original.label.value_counts().to_dict())
    print("Manual new-labels available: TRAIN=", len(ht), "HOLDOUT=", len(hv),
          "holdout_docs=", len(holdout_docs))
    print("Human train labels:", ht.label.value_counts().to_dict())
    print("Human holdout labels:", hv.label.value_counts().to_dict())
    if len(original) != 7169 or original.document_id.nunique() != 130:
        print("IMPORTANT: original corpus differs from the reported 7,169 / 130;"
              " old 0.681 benchmark may not be directly comparable.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_manifest.json").write_text(json.dumps({
        "seed_original_fold": FOLD_SEED,
        "seed_human_holdout": HUMAN_SPLIT_SEED,
        "human_holdout_document_ids": holdout_docs,
        "original_count": len(original),
        "original_docs": original.document_id.nunique(),
        "human_train_count": len(ht), "human_holdout_count": len(hv),
        "human_train_class_counts": ht.label.value_counts().to_dict(),
        "human_holdout_class_counts": hv.label.value_counts().to_dict(),
        "original_fold_counts": original.fold.value_counts().sort_index().to_dict(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Add VERIFIED human labels to PSAI training")
    parser.add_argument("--mode", choices=["audit", "cv", "final"], default="audit")
    parser.add_argument("--folds", default="0", help='0 or 0,1,2,3,4 or all')
    parser.add_argument("--scheme", choices=["targeted", "once"], default="targeted")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--original", type=Path, default=ROOT / "data_segment_cls" / "segment_classification.csv")
    parser.add_argument("--human", type=Path, default=ROOT / "extra_human.csv")
    args = parser.parse_args()

    original = read_original(args.original)
    ht, hv, held_docs = read_human(args.human, original)
    out_dir = ROOT / "runs" / "human_experiment"
    report_dataset(original, ht, hv, held_docs, out_dir)
    if len(hv) < 10 or len(ht) < 50:
        raise RuntimeError("Too few usable human annotations after leakage filters")
    if args.mode == "audit":
        print("Audit only: no model loaded or trained; original data and models unchanged")
        return

    libs = lazy_libs()
    torch, Dataset, AutoModel, AutoTokenizer, *_ = libs
    repeats = REPETITIONS if args.scheme == "targeted" else {lab: 1 for lab in LABELS}
    extra = training_examples(ht, repeats)
    human_y = np.array([L2I[x] for x in hv.label])

    if args.mode == "cv":
        folds = list(range(5)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
        if not folds or any(f not in range(5) for f in folds) or len(folds) != len(set(folds)):
            raise ValueError("--folds must be 0..4 unique, or 'all'")
        stats, old_human, new_human = [], [], []
        for f in folds:
            val_f = (f + 1) % 5
            train = original[~original.fold.isin([f, val_f])][COLS]
            test = original[original.fold.eq(f)][COLS].reset_index(drop=True)
            train = pd.concat([train, extra], ignore_index=True)
            test_y = np.array([L2I[x] for x in test.label])
            print(f"\nFOLD {f}: TRAIN={len(train)} TEST={len(test)} VAL_fold={val_f} (unused)")
            assert not set(ht.document_id).intersection(set(original[original.fold.isin([f, val_f])].document_id))

            old = predict_saved_model(ROOT / "models" / f"kfold_f{f}", test, torch, AutoModel, AutoTokenizer)
            old_h = predict_saved_model(ROOT / "models" / f"kfold_f{f}", hv, torch, AutoModel, AutoTokenizer)
            if old is not None:
                print_metrics("OLD CV", metrics(test_y, old))
            else:
                print("WARNING: cannot find old kfold_f checkpoint for this fold")

            save_dir = ROOT / "models" / f"human_{args.scheme}_f{f}"
            train_model(train, FOLD_SEED, save_dir, args.epochs, libs)
            new = predict_saved_model(save_dir, test, torch, AutoModel, AutoTokenizer)
            new_h = predict_saved_model(save_dir, hv, torch, AutoModel, AutoTokenizer)
            nm = metrics(test_y, new)
            print_metrics("NEW CV", nm)
            if old is not None:
                print("Delta macro-F1:", f"{nm['macro_f1'] - metrics(test_y, old)['macro_f1']:+.4f}")
            stats.append({"fold": f, "old": metrics(test_y, old) if old is not None else None,
                          "new": nm})
            if old_h is not None:
                old_human.append(old_h)
            new_human.append(new_h)
            pd.DataFrame({"document_id": test.document_id, "label": test.label,
                          "old": [LABELS[k] for k in old.argmax(1)] if old is not None else [""]*len(test),
                          "new": [LABELS[k] for k in new.argmax(1)]}).to_csv(
                              out_dir / f"cv_fold{f}_{args.scheme}.csv", index=False, encoding="utf-8-sig")
        print("\n===== CV SUMMARY: matched test folds =====")
        for item in stats:
            print(item["fold"], "old=", round(item["old"]["macro_f1"], 4) if item["old"] else "NA",
                  "new=", round(item["new"]["macro_f1"], 4))
        print("mean new fold F1:", round(np.mean([s["new"]["macro_f1"] for s in stats]), 4))
        if all(x["old"] is not None for x in stats):
            print("mean old fold F1:", round(np.mean([s["old"]["macro_f1"] for s in stats]), 4))
        print("\n===== HUMAN HOLDOUT (never trained on these docs) =====")
        if len(old_human) == len(folds):
            print_metrics("OLD ensemble selected folds", metrics(human_y, np.mean(old_human, axis=0)))
        print_metrics("NEW ensemble selected folds", metrics(human_y, np.mean(new_human, axis=0)))
        (out_dir / f"cv_summary_{args.scheme}.json").write_text(json.dumps({
            "test_fold_results": stats, "human_holdout_docs": held_docs,
            "human_holdout_new": metrics(human_y, np.mean(new_human, axis=0)),
            "human_holdout_old": metrics(human_y, np.mean(old_human, axis=0))
            if len(old_human) == len(folds) else None,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    elif args.mode == "final":
        print("\nFinal experiment: all original labels + human TRAIN subset only;")
        print("human HOLDOUT remains unseen. Does not change the live ensemble.")
        train = pd.concat([original[COLS], extra], ignore_index=True)
        old_ps, new_ps = [], []
        for seed in (11, 22, 33):
            old = predict_saved_model(ROOT / "models" / f"final_s{seed}", hv,
                                      torch, AutoModel, AutoTokenizer)
            if old is not None:
                old_ps.append(old)
            save_dir = ROOT / "models" / f"human_{args.scheme}_s{seed}"
            train_model(train, seed, save_dir, args.epochs, libs)
            new_ps.append(predict_saved_model(save_dir, hv, torch, AutoModel, AutoTokenizer))
        print("\n===== FINAL ENSEMBLE ON NEW-DOC HUMAN HOLDOUT =====")
        if len(old_ps) == 3:
            print_metrics("OLD final_s11/22/33", metrics(human_y, np.mean(old_ps, axis=0)))
        else:
            print("Old final_s11/22/33 unavailable. Cannot compare on new docs")
        print_metrics("NEW human_s11/22/33", metrics(human_y, np.mean(new_ps, axis=0)))
        (out_dir / f"final_summary_{args.scheme}.json").write_text(json.dumps({
            "old": metrics(human_y, np.mean(old_ps, axis=0)) if len(old_ps) == 3 else None,
            "new": metrics(human_y, np.mean(new_ps, axis=0)),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOutputs:", out_dir)
    print("Old checkpoints and live extraction have NOT been modified.")


if __name__ == "__main__":
    main()
