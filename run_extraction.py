"""Run extraction explicitly and publish output only after every document succeeds.

Importing this module performs no I/O or inference. Existing outputs require an
explicit --overwrite; failed runs leave the previous artifact unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import extraction

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "data_cleaned"
OUT_PATH = BASE_DIR / "data_extracted" / "extraction_model.jsonl"


def run(*, cleaned_dir: Path = CLEANED_DIR, output_path: Path = OUT_PATH,
        use_model: bool = True, overwrite: bool = False) -> dict:
    cleaned_dir, output_path = Path(cleaned_dir), Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; explicitly allow overwrite: {output_path}")
    # Validate all document IDs/source ambiguities before loading a checkpoint.
    sources = list(extraction.iter_documents(cleaned_dir))
    if not sources:
        raise ValueError(f"No cleaned Markdown/text documents found: {cleaned_dir}")
    if output_path.resolve() in {path.resolve() for _, path in sources}:
        raise ValueError("Extraction output must not overwrite a cleaned source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    totals: dict[str, int] = {}
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent,
                delete=False) as stream:
            temporary = Path(stream.name)
            for document_id, source in sources:
                text = extraction._read_text(source)
                try:
                    record = extraction.extract_document(document_id, text, use_model=use_model)
                except Exception as exc:
                    raise RuntimeError(f"Extraction failed for {document_id} ({source}): {exc}") from exc
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                for name, entries in record["fields"].items():
                    totals[name] = totals.get(name, 0) + len(entries)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, output_path)
        else:
            # Atomic no-clobber publication, including a competing writer that
            # creates output_path after the initial existence check.
            os.link(temporary, output_path)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"documents": len(sources), "field_counts": totals,
            "output_path": str(output_path.resolve()), "use_model": use_model}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--rules-only", action="store_true", help="Do not load classifier checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output after full success")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = run(cleaned_dir=args.cleaned_dir, output_path=args.output,
                     use_model=not args.rules_only, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Extraction aborted: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
