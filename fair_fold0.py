"""PSAI: reproducible, leakage-controlled Fold-0 comparison.

Run in the project root (beside train_human_cv.py):
  python fair_fold0.py --mode audit
  python fair_fold0.py --mode baseline
  python fair_fold0.py --mode human
  python fair_fold0.py --mode compare

Never edits the original CSV or existing models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from train_human_cv import (
    ROOT, LABELS, L2I, COLS, read_original, read_human,
    text_key, lazy_libs, train_model, predict_saved_model,
)

OUT = ROOT / "runs" / "fair_fold0"
MANIFEST = OUT / "frozen_split.json"
BASELINE = ROOT / "models" / "fair_fold0_baseline"
HUMAN = ROOT / "models" / "fair_fold0_human_once"
ORIGINAL_PATH = ROOT / "data_segment_cls" / "segment_classification.csv"
HUMAN_PATH = ROOT / "extra_human.csv"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_rows(rows: pd.DataFrame) -> str:
    relevant = rows[["document_id", "source_line", "text", "label", "fold"]]
    return hashlib.sha256(relevant.to_csv(index=False).encode("utf-8")).hexdigest()


def human_digest(rows: pd.DataFrame) -> str:
    return hashlib.sha256(rows[COLS].to_csv(index=False).encode("utf-8")).hexdigest()


def partitions():
    orig = read_original(ORIGINAL_PATH)
    ht, hv, held_docs = read_human(HUMAN_PATH, orig)
    test = orig.loc[orig.fold.eq(0)].copy().reset_index(drop=True)
    val = orig.loc[orig.fold.eq(1)].copy().reset_index(drop=True)
    all_train = orig.loc[orig.fold.isin([2, 3, 4])].copy().reset_index(drop=True)

    # Remove train sentences that are text-identical to held-out fold0 or fold1.
    # Unlike just discarding repeated test sentences, this retains a fixed test set.
    held_texts = set(test.text.map(text_key)) | set(val.text.map(text_key))
    train = all_train.loc[~all_train.text.map(text_key).isin(held_texts)].copy().reset_index(drop=True)

    # 'read_human' already removes repeated source sentences and holdout texts,
    # but assert that leakage-control remains true if the helper is changed.
    train_keys = set(train.text.map(text_key))
    human_train_keys = set(ht.text.map(text_key))
    human_test_keys = set(hv.text.map(text_key))
    assert not train_keys.intersection(held_texts)
    assert not human_train_keys.intersection(held_texts)
    assert not human_train_keys.intersection(human_test_keys)
    assert not set(test.document_id).intersection(set(train.document_id))
    assert not set(val.document_id).intersection(set(train.document_id))
    assert not set(ht.document_id).intersection(set(orig.document_id))
    assert not set(hv.document_id).intersection(set(orig.document_id))
    assert not set(held_docs).intersection(set(ht.document_id))
    return orig, train, test, val, all_train, ht, hv, held_docs


def metadata(orig, train, test, val, original_train, ht, hv, held_docs):
    return {
        "description": "Fixed baseline vs human-once: test fold0, val fold1 unused, training folds2-4 minus exact normalized test/val overlap",
        "original_sha256": checksum(ORIGINAL_PATH),
        "human_sha256": checksum(HUMAN_PATH),
        "orig_row_digest": digest_rows(orig),
        "human_training_digest": human_digest(ht),
        "human_holdout_digest": human_digest(hv),
        "test_document_ids": sorted(test.document_id.unique()),
        "validation_document_ids": sorted(val.document_id.unique()),
        "human_holdout_document_ids": held_docs,
        "original_rows": len(orig), "original_documents": int(orig.document_id.nunique()),
        "nominal_train_rows": len(original_train),
        "purged_train_rows": len(train),
        "removed_from_train": len(original_train) - len(train),
        "test_rows": len(test), "val_rows": len(val),
        "human_train_rows": len(ht), "human_holdout_rows": len(hv),
    }


def freeze_or_verify(meta, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        prior = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if prior != meta:
            raise SystemExit("STOP: source data or split changed after audit.\n"
                             "Do not train or compare. Inspect frozen_split.json and source changes first.")
        print("Frozen split verified: unchanged since first audit.")
    elif mode == "audit":
        MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Frozen manifest created:", MANIFEST)
    else:
        raise SystemExit("Run 'python fair_fold0.py --mode audit' first to freeze the split.")


def show_metrics(title: str, rows: pd.DataFrame, probabilities):
    gold = np.array([L2I[x] for x in rows.label])
    pred = probabilities.argmax(axis=1)
    print(f"\n{title} | N={len(rows)} | Acc={accuracy_score(gold, pred):.4f} | "
          f"Macro-F1={f1_score(gold, pred, labels=range(5), average='macro', zero_division=0):.4f}")
    print(classification_report(gold, pred, labels=range(5), target_names=LABELS,
                                zero_division=0, digits=3))
    return pred


def do_train(name, orig_train, ht, seed=42):
    if name == "baseline":
        save_dir = BASELINE
        train = orig_train[COLS].copy()
    else:
        save_dir = HUMAN
        # Additional annotations are included ONLY ONCE (no per-class repetition).
        train = pd.concat([orig_train[COLS], ht[COLS]], ignore_index=True)
    if save_dir.exists():
        raise SystemExit(f"STOP: {save_dir} already exists. Not overwriting checkpoints.")
    print(f"\n{name.upper()}: train={len(train)} | orig={len(orig_train)} | manual_added={len(train)-len(orig_train)}")
    print("Classes:", train.label.value_counts().to_dict())
    libs = lazy_libs()
    train_model(train, seed, save_dir, epochs=5, libs=libs)
    (save_dir / "frozen_data_sha256.json").write_text(
        json.dumps({"manifest_sha256": checksum(MANIFEST), "variant": name,
                    "seed": seed, "epochs": 5}, indent=2), encoding="utf-8")
    print("Saved:", save_dir)


def do_compare(test, hv):
    for name, model_dir in [("baseline", BASELINE), ("human", HUMAN)]:
        if not (model_dir / "frozen_data_sha256.json").is_file():
            raise SystemExit(f"Missing trained checkpoint: {model_dir}. Train both models first.")
        recorded = json.loads((model_dir / "frozen_data_sha256.json").read_text(encoding="utf-8"))
        if recorded != {"manifest_sha256": checksum(MANIFEST), "variant": name,
                        "seed": 42, "epochs": 5}:
            raise SystemExit(f"STOP: incompatible checkpoint metadata for {model_dir}")
    libs = lazy_libs()  # same CUDA as previous experiments
    torch, _Dataset, AutoModel, AutoTokenizer, *_ = libs
    predictions = {}
    for name, model_dir in [("baseline", BASELINE), ("human", HUMAN)]:
        p_test = predict_saved_model(model_dir, test, torch, AutoModel, AutoTokenizer)
        p_hold = predict_saved_model(model_dir, hv, torch, AutoModel, AutoTokenizer)
        predictions[(name, "test")] = show_metrics(f"{name.upper()} | Fold0 TEST", test, p_test)
        predictions[(name, "human_holdout")] = show_metrics(f"{name.upper()} | HUMAN HOLDOUT (selected sample)", hv, p_hold)
    for dataset_name, rows in [("test", test), ("human_holdout", hv)]:
        out = rows[["document_id", "text", "label"]].copy()
        out["baseline_prediction"] = [LABELS[i] for i in predictions[("baseline", dataset_name)]]
        out["human_prediction"] = [LABELS[i] for i in predictions[("human", dataset_name)]]
        out["baseline_correct"] = out.label.eq(out.baseline_prediction)
        out["human_correct"] = out.label.eq(out.human_prediction)
        out.to_csv(OUT / f"{dataset_name}_predictions.csv", index=False, encoding="utf-8-sig")
        print("Saved per-example:", OUT / f"{dataset_name}_predictions.csv")
    print("\nCompare fold0 as an exploratory test; do not select final hyperparameters on the 66-sentence holdout repeatedly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "baseline", "human", "compare"], required=True)
    args = parser.parse_args()
    orig, train, test, val, nominal, ht, hv, held_docs = partitions()
    meta = metadata(orig, train, test, val, nominal, ht, hv, held_docs)
    freeze_or_verify(meta, args.mode)
    print("Original:", len(orig), "rows,", orig.document_id.nunique(), "docs")
    print("Nominal train:", len(nominal), "| Purged train:", len(train),
          "| Overlap removed:", len(nominal)-len(train))
    print("Fold0 test:", len(test), "| Validation (unused):", len(val),
          "| Human train:", len(ht), "| Human holdout:", len(hv))
    if args.mode == "audit":
        print("AUDIT COMPLETE: no GPU, no training; re-running will verify this exact frozen split.")
    elif args.mode in {"baseline", "human"}:
        do_train(args.mode, train, ht)
    else:
        do_compare(test, hv)


if __name__ == "__main__":
    main()
