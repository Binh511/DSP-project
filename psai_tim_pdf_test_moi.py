# -*- coding: utf-8 -*-
"""PSAI: list *candidate* new PDF documents for a fresh, blind test.

Usage: python psai_tim_pdf_test_moi.py
No GPU required. Never changes source datasets, models or checkpoints.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIG = ROOT / "data_segment_cls" / "segment_classification.csv"
HUMAN = ROOT / "extra_human.csv"
PDF_FOLDERS = [
    ROOT / "data" / "raw" / "pdf",
    ROOT / "data" / "raw" / "pdfs",
    ROOT / "data" / "raw",
    ROOT / "input_pdf",
    ROOT / "data" / "input_pdf",
    ROOT / "raw_pdf",
]
OUT = ROOT / "runs" / "independent_test" / "fresh_pdf_candidates.csv"
ID_RE = re.compile(r"(?<![A-Za-z0-9])PSAI[_-]?(\d{1,5})(?!\d)", re.IGNORECASE)


def normalize_id(x: str) -> str:
    x = (x or "").strip().upper()
    match = ID_RE.search(x)
    if match:
        return f"PSAI{int(match.group(1)):03d}"
    return x


def read_used_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy file dữ liệu để loại tài liệu đã dùng: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "document_id" not in (reader.fieldnames or []):
            raise ValueError(f"File không có cột document_id: {path}")
        return {normalize_id(row["document_id"]) for row in reader if row.get("document_id")}


def infer_id(pdf: Path) -> tuple[str, bool]:
    # A project ID in the filename is preferable; a parent ID can work too.
    for name in [pdf.stem] + [p.name for p in list(pdf.parents)[:2]]:
        m = ID_RE.search(name)
        if m:
            return f"PSAI{int(m.group(1)):03d}", True
    return normalize_id(pdf.stem), False


def main() -> None:
    used_original = read_used_ids(ORIG)
    used_human = read_used_ids(HUMAN)
    used_all = used_original | used_human
    folders = [p for p in PDF_FOLDERS if p.is_dir()]
    if not folders:
        print("Không tìm thấy thư mục PDF trong các vị trí dự kiến.")
        print("Hãy cho biết đường dẫn thư mục đang lưu các PDF gốc để chỉnh PDF_FOLDERS.")
        return
    paths = {p.resolve() for base in folders for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"}
    rows = []
    excluded = 0
    for pdf in sorted(paths, key=lambda p: str(p).lower()):
        doc_id, confident_id = infer_id(pdf)
        if doc_id in used_all:
            excluded += 1
            continue
        rows.append({
            "document_id_candidate": doc_id,
            "pdf_path": str(pdf),
            "id_from_PSAl_pattern": "YES" if confident_id else "NO",
            "status": "CANDIDATE_VERIFY_MANUALLY" if confident_id else "UNKNOWN_ID_VERIFY_MANUALLY",
            "notes": "Chỉ là ứng viên: kiểm tra nội dung có trùng/chỉnh sửa từ tài liệu đã dùng không",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["document_id_candidate", "pdf_path", "id_from_PSAl_pattern", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"PDF tìm thấy: {len(paths)} | Loại vì trùng document_id đã dùng: {excluded}")
    print(f"Document ID trong dataset gốc: {len(used_original)} | Trong extra_human: {len(used_human)}")
    print(f"PDF ứng viên chưa trùng document_id: {len(rows)}")
    for row in rows[:12]:
        print(f"  {row['document_id_candidate']:15s} {row['status']:30s} {row['pdf_path']}")
    print(f"\nĐã lưu: {OUT}")
    print("LƯU Ý: chưa chứng minh PDF hoàn toàn mới. Phải kiểm tra trùng nội dung/phiên bản")
    print("với dữ liệu đã dùng trước khi đưa vào bộ TEST độc lập.")


if __name__ == "__main__":
    main()
