"""Validate Markdown sentence labels against cleaned source text."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _normalize(text: str) -> str:
    text = re.sub(r"\\([|_`*])", r"\1", text)
    text = re.sub(r"`+", "", text)
    return " ".join(text.split()).casefold()


def _annotation_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0].casefold() in {"sentence_id", "sentence id"}:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append({
            "sentence_id": cells[0],
            "text": cells[1],
            "label": cells[2].upper(),
            "source_line": str(line_number),
        })
    return rows


def prepare_segment_classification(
    *,
    label_dir: Path,
    cleaned_dir: Path,
    segment_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_rows: list[dict[str, str]] = []
    details: list[dict[str, str]] = []
    documents_with_labels = 0
    documents_needing_review: list[str] = []
    documents_without_usable_labels: list[str] = []

    for annotation_path in sorted(label_dir.glob("PSAI*.md")):
        document_id = annotation_path.stem
        source_paths = sorted((cleaned_dir / document_id).glob("*.md"))
        source_paths += sorted((cleaned_dir / document_id).glob("*.txt"))
        if not source_paths:
            raise FileNotFoundError(f"Cleaned source is missing for {document_id}.")
        source = source_paths[0].read_text(encoding="utf-8-sig", errors="replace")
        normalized_source = _normalize(source)
        annotation_rows = _annotation_rows(annotation_path)
        matched_count = 0
        for row in annotation_rows:
            normalized_text = _normalize(row["text"])
            matched = bool(normalized_text and normalized_text in normalized_source)
            details.append({
                "document_id": document_id,
                **row,
                "exact_source_match": str(matched).lower(),
            })
            if matched:
                labeled_rows.append({"document_id": document_id, **row})
                matched_count += 1
        if matched_count:
            documents_with_labels += 1
        if annotation_rows and matched_count != len(annotation_rows):
            documents_needing_review.append(document_id)
        if not matched_count:
            documents_without_usable_labels.append(document_id)

    classification_path = output_dir / "segment_classification.csv"
    with classification_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "document_id", "sentence_id", "text", "label", "source_line",
        ])
        writer.writeheader()
        writer.writerows(labeled_rows)
    validation_path = output_dir / "annotation_validation.csv"
    with validation_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "document_id", "sentence_id", "text", "label", "source_line",
            "exact_source_match",
        ])
        writer.writeheader()
        writer.writerows(details)
    (output_dir / "annotation_validation_details.csv").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split_manifest = {
        "task": "segment_classification",
        "bio_ner_training_ready": False,
        "documents_with_exact_source_labels": documents_with_labels,
        "exact_source_labeled_rows": len(labeled_rows),
        "documents_needing_review": documents_needing_review,
        "documents_without_usable_labels": documents_without_usable_labels,
        "classification_file": str(classification_path.resolve()),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "valid_documents": documents_with_labels,
        "valid_rows": len(labeled_rows),
        "documents_needing_review": documents_needing_review,
        "documents_without_usable_labels": documents_without_usable_labels,
    }
