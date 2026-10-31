"""Phân loại câu proposal thành 5 nhãn bằng DeBERTa đã fine-tune."""

from __future__ import annotations

import json
import re as _re2
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "models" / "psai_deberta_ctx_v7"
FALLBACK_MODEL_DIRS = (
    Path(__file__).parent / "models" / "psai_deberta_ctx_v5",
    Path(__file__).parent / "models" / "psai_deberta_ctx_v4",
    Path(__file__).parent / "models" / "psai_deberta_ctx",
)

_tokenizer = None
_model = None
_config = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _tokenizer, _model, _config
    if _model is not None:
        return
    model_dir = next(
        (path for path in (MODEL_DIR, *FALLBACK_MODEL_DIRS)
         if (path / "inference_config.json").is_file()),
        None,
    )
    if model_dir is None:
        raise FileNotFoundError(
            "No classifier inference_config.json found in "
            + ", ".join(str(path) for path in (MODEL_DIR, *FALLBACK_MODEL_DIRS))
        )
    _config = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
    _tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    _model.to(_device).eval()


_BULLET = _re2.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
_MODAL = _re2.compile(r"\b(will|must|should|shall|need to|needs to)\b", _re2.I)
_IMPER = _re2.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b",
    _re2.I)
_TIME = _re2.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", _re2.I)


def _context(section, prev_text, pos=0.0, text="", nxt=""):
    ver = _config.get("ctx_version")
    bullet = 1 if _BULLET.match(text) else 0
    if ver == 3:
        cue = f"{int(bool(_MODAL.search(text)))}{int(bool(_IMPER.match(text)))}{int(bool(_TIME.search(text)))}"
        return (f"Section: {section.strip()[-120:]} | Pos: {int(pos * 10)} | "
                f"Bullet: {bullet} | Cue: {cue} | Previous: {prev_text.strip()[-150:]} | "
                f"Next: {nxt.strip()[:150]}")
    if ver == 2:
        return (f"Section: {section.strip()[-120:]} | Pos: {int(pos * 10)} | "
                f"Bullet: {bullet} | Previous: {prev_text.strip()[-200:]}")
    legacy_section = section.strip().split(" > ")[-1]
    return f"Section: {legacy_section[:120]} | Previous: {prev_text.strip()[-200:]}"


def classify(items: list[dict], batch_size: int = 32) -> list[dict]:
    """items: [{"text": ..., "section": ..., "prev_text": ...}, ...]
    section và prev_text có thể bỏ trống."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not items:
        return []
    _load()
    labels = _config["labels"]
    other_id = labels.index("OTHER")
    bias = _config.get("other_bias", 0.0)
    max_len = _config.get("max_len", 192)

    results = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        enc = _tokenizer(
            [b["text"][:_config.get("text_max_chars", 600)] for b in batch],
            [_context(b.get("section", ""), b.get("prev_text", ""), b.get("pos", 0.0),
                      b["text"], b.get("next_text", "")) for b in batch],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(_device)
        with torch.no_grad():
            logits = _model(**enc).logits.float().cpu().numpy()
        if logits.shape != (len(batch), len(labels)) or not np.isfinite(logits).all():
            raise FloatingPointError('Classifier returned invalid or nonfinite logits')
        logits[:, other_id] += bias
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        if not np.isfinite(probs).all():
            raise FloatingPointError('Classifier probabilities are nonfinite')
        for row in probs:
            idx = int(row.argmax())
            results.append({"label": labels[idx], "score": round(float(row[idx]), 3)})
    return results


if __name__ == "__main__":
    samples = [
        {"text": "The goal of this project is to build a monitoring dashboard for miners."},
        {"text": "The system must support Python 3.10 or higher."},
        {"text": "We will develop a script that collects metrics from the node."},
        {"text": "The final deliverable is a Grafana dashboard and documentation."},
        {"text": "Contact: team@example.com"},
    ]
    for s, r in zip(samples, classify(samples)):
        print(f"{r['label']:16s} {r['score']:.2f}  {s['text']}")
