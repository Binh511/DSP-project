from __future__ import annotations

from pathlib import Path
import sys
import extraction

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "data_cleaned"


def _iter_document_ids() -> list[str]:
    if not CLEANED_DIR.is_dir():
        raise FileNotFoundError(f"Cleaned directory not found: {CLEANED_DIR}")

    doc_ids: set[str] = set()
    for path in sorted(CLEANED_DIR.rglob("*.md")) + sorted(CLEANED_DIR.rglob("*.txt")):
        doc_ids.add(path.parent.name)
    return sorted(doc_ids)


def _read_document_text(document_id: str) -> str:
    doc_dir = CLEANED_DIR / document_id
    candidates = sorted(doc_dir.glob("*.md")) + sorted(doc_dir.glob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No cleaned source found for {document_id} in {doc_dir}")
    return candidates[0].read_text(encoding="utf-8-sig", errors="replace")


def compare_document(document_id: str) -> None:
    text = _read_document_text(document_id)

    try:
        model_res = extraction.extract_document(document_id, text, use_model=True)["fields"]
    except FileNotFoundError as exc:
        print(f"[SKIP] Model unavailable for {document_id}: {exc}")
        model_res = {name: [] for name in extraction.FIELD_CUES}

    rule_res = extraction.extract_document(document_id, text, use_model=False)["fields"]

    print(f"\n=== Document: {document_id} ===")
    print(f"{'field':18s} {'model':>8s} {'regex':>8s}")
    for name in sorted(set(model_res) | set(rule_res)):
        print(f"{name:18s} {len(model_res.get(name, [])):8d} {len(rule_res.get(name, [])):8d}")

    for name in ("objectives", "requirements", "activities", "expected_results"):
        print(f"\n--- {name} (model) ---")
        items = model_res.get(name, [])
        if not items:
            print("  (no model predictions)")
            continue
        for item in items[:12]:
            value = item.get("value", "")
            label_score = item.get("label_score", "")
            print(f"  [{label_score}] {value[:110]}")


if __name__ == "__main__":
    # dùng: python compare_extraction.py PSAI150 PSAI084
    document_ids = sys.argv[1:] or ["PSAI150"]
    for document_id in document_ids:
        compare_document(document_id)

    print(f"\nCompared {len(document_ids)} document(s) from {CLEANED_DIR}")