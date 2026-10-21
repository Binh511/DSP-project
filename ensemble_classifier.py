"""Ensemble v5 + v6 + v7: trung bình xác suất, giữ nguyên giao diện classify()."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAMES = ["psai_deberta_ctx_v5", "psai_deberta_ctx_v6", "psai_deberta_ctx_v7"]
MODELS_DIR = Path(__file__).parent / "models"
LABELS = ["OBJECTIVE", "REQUIREMENT", "ACTIVITY", "EXPECTED_RESULT", "OTHER"]
_device = "cuda" if torch.cuda.is_available() else "cpu"

_members = None

_BULLET = re.compile(r"^\s*(?:[-*●•]|\d+[.)])\s")
_MODAL = re.compile(r"\b(will|must|should|shall|need to|needs to)\b", re.I)
_IMPER = re.compile(
    r"^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b",
    re.I,
)
_TIME = re.compile(r"\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b", re.I)


def _context(ver, section, prev, pos, text, nxt):
    bullet = 1 if _BULLET.match(text) else 0
    if ver == 3:
        cue = f"{int(bool(_MODAL.search(text)))}{int(bool(_IMPER.match(text)))}{int(bool(_TIME.search(text)))}"
        return (f"Section: {section.strip()[-120:]} | Pos: {int(pos * 10)} | "
                f"Bullet: {bullet} | Cue: {cue} | Previous: {prev.strip()[-150:]} | "
                f"Next: {nxt.strip()[:150]}")
    if ver == 2:
        return (f"Section: {section.strip()[-120:]} | Pos: {int(pos * 10)} | "
                f"Bullet: {bullet} | Previous: {prev.strip()[-200:]}")
    return f"Section: {section.strip()[:120]} | Previous: {prev.strip()[-200:]}"


def _load():
    global _members
    if _members is not None:
        return

    members = []
    for name in MODEL_NAMES:
        model_dir = MODELS_DIR / name
        config = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
        if config["labels"] != LABELS:
            raise ValueError(f"Unexpected label order in {model_dir}: {config['labels']}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device).eval()
        members.append((config, tokenizer, model))
    _members = members


def classify_probs(items: list[dict], batch_size: int = 32) -> np.ndarray:
    _load()
    out = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        acc = None
        for config, tokenizer, model in _members:
            ver = config.get("ctx_version")
            contexts = [
                _context(ver, item.get("section", ""), item.get("prev_text", ""),
                         item.get("pos", 0.0), item["text"], item.get("next_text", ""))
                for item in batch
            ]
            enc = tokenizer(
                [item["text"] for item in batch],
                contexts,
                truncation=True,
                max_length=config.get("max_len", 192),
                padding=True,
                return_tensors="pt",
            ).to(_device)
            with torch.no_grad():
                logits = model(**enc).logits.float().cpu().numpy()
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            acc = probabilities if acc is None else acc + probabilities
        out.append(acc / len(_members))
    return np.vstack(out) if out else np.zeros((0, len(LABELS)))


def classify(items: list[dict], batch_size: int = 32) -> list[dict]:
    probabilities = classify_probs(items, batch_size)
    return [
        {"label": LABELS[int(row.argmax())], "score": round(float(row.max()), 3)}
        for row in probabilities
    ]
