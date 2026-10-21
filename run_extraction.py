import json
import sys
from pathlib import Path

import extraction

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "data_cleaned"
OUT_PATH = BASE_DIR / "data_extracted" / "extraction_model.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

doc_dirs = sorted({p.parent for p in CLEANED_DIR.rglob("*.md")} |
                  {p.parent for p in CLEANED_DIR.rglob("*.txt")})
print(f"Tìm thấy {len(doc_dirs)} tài liệu")

totals: dict[str, int] = {}
failed = []
with OUT_PATH.open("w", encoding="utf-8", newline="\n") as stream:
    for doc_dir in doc_dirs:
        document_id = doc_dir.name
        try:
            sources = sorted(doc_dir.glob("*.md")) + sorted(doc_dir.glob("*.txt"))
            text = sources[0].read_text(encoding="utf-8-sig", errors="replace")
            record = extraction.extract_document(document_id, text)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            for name, items in record["fields"].items():
                totals[name] = totals.get(name, 0) + len(items)
        except Exception as exc:
            failed.append((document_id, str(exc)))
            print(f"[LỖI] {document_id}: {exc}")

print()
print("Tổng số câu theo trường:")
for name, count in totals.items():
    print(f"  {name:18s} {count}")
print(f"\nTài liệu thành công: {len(doc_dirs) - len(failed)} / {len(doc_dirs)}")
print("Đã ghi:", OUT_PATH)