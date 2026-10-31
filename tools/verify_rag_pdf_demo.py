"""Reproduce a local PDF -> BM25 -> cited HTML functional smoke test.

The two-page fixture is authored here, rather than taken from a benchmark or a
held-out evaluation set. This proves source fidelity and CLI wiring only: it
does not measure OCR, classifier accuracy, retrieval relevance, or plan quality.
Run with .venv/psai_runtime/Scripts/python.exe tools/verify_rag_pdf_demo.py.
Every invocation creates a fresh directory and leaves existing outputs intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PAGES = (
    (
        "PSAI authored PDF smoke fixture - page 1",
        "Our objective is to improve the local project evidence workflow.",
        "The application must support exact source citations for every report.",
        "We will develop a Python report module for project planning.",
    ),
    (
        "PSAI authored PDF smoke fixture - page 2",
        "The result will be a source-grounded HTML report for project teams.",
        "The project budget is 1200 USD for two developers over six weeks.",
        "We will create documentation and release the application locally.",
    ),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def _pdf_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_authored_pdf(path: Path) -> None:
    """Write a selectable-text PDF using only the standard library.

    Helvetica's base-14 font permits this ASCII test fixture without a font
    download. Offsets in the PDF cross-reference table count the actual bytes.
    """
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
    ]
    for number, lines in enumerate(FIXTURE_PAGES):
        content_id = 4 + number * 2
        objects.append(
            ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
             "/Resources << /Font << /F1 7 0 R >> >> "
             f"/Contents {content_id} 0 R >>").encode("ascii")
        )
        stream = ("BT\n/F1 10 Tf\n44 740 Td\n18 TL\n" + "\n".join(
            f"({_pdf_literal(line)}) Tj\nT*" for line in lines) + "\nET\n").encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                       + stream + b"endstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    result.extend((f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                   f"startxref\n{xref}\n%%EOF\n").encode("ascii"))
    with path.open("xb") as output:
        output.write(result)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.citations: list[str] = []
        self.text: list[str] = []
        self.active_elements: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and "citation" in attributes.get("class", "").split():
            self.citations.append(attributes.get("href", ""))
        if tag in {"script", "iframe", "object", "embed"}:
            self.active_elements.append(tag)

    def handle_data(self, data):
        self.text.append(data)


_GUARDED_CLI = r'''
import json, os, pathlib, runpy, sys
blocked_imports = {"torch", "transformers", "sentence_transformers", "chromadb", "peft"}
attempts = []
def guard(event, args):
    if event == "import" and args and args[0].split(".")[0] in blocked_imports:
        attempts.append({"event": event, "name": args[0]})
        raise RuntimeError("Functional PDF smoke must not load model packages: " + args[0])
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname", "socket.bind"}:
        attempts.append({"event": event})
        raise RuntimeError("Functional PDF smoke must remain offline: " + event)
sys.addaudithook(guard)
entry = pathlib.Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(entry.parent))
try:
    runpy.run_path(str(entry), run_name="__main__")
finally:
    value = {"blocked_attempts": attempts,
             "model_modules_loaded": sorted(name for name in blocked_imports if name in sys.modules)}
    pathlib.Path(os.environ["PSAI_SMOKE_AUDIT_PATH"]).write_text(json.dumps(value, indent=2), encoding="utf-8")
'''


def run_cli(directory: Path, name: str, input_path: Path, output_path: Path) -> dict:
    command = [sys.executable, "-c", _GUARDED_CLI, str(ROOT / "psai.py"), "analyze",
               "--input", str(input_path), "--output", str(output_path),
               "--classifier", "rules", "--retrieval", "bm25",
               "--query", "exact source citations report documentation budget",
               "--top-k", "3"]
    trace = directory / "traces"
    trace.mkdir(exist_ok=True)
    audit_path = trace / f"{name}_runtime_audit.json"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": "", "PSAI_SMOKE_AUDIT_PATH": str(audit_path)}
    started = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                             encoding="utf-8", errors="replace", timeout=90)
    (trace / f"{name}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (trace / f"{name}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["blocked_attempts"] or audit["model_modules_loaded"]:
        raise AssertionError(f"{name}: pipeline attempted to load a model or use the network")
    return {"returncode": process.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": process.stdout, "stderr": process.stderr,
            "runtime_audit": str(audit_path.relative_to(directory))}


def _snapshot(directory: Path) -> dict:
    return {str(path.relative_to(directory)): _sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_pdf_demo(output_root: Path) -> Path:
    from pypdf import PdfReader, PdfWriter
    import pypdf

    output_root = output_root.resolve()
    if not output_root.is_relative_to(ROOT / "outputs"):
        raise ValueError("Choose an output root under this project's outputs directory")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(tempfile.mkdtemp(prefix=f"pdf_demo_{stamp}_", dir=output_root))
    fixture = directory / "authored_proposal.pdf"
    write_authored_pdf(fixture)
    reader = PdfReader(fixture)
    assert len(reader.pages) == len(FIXTURE_PAGES), "Fixture page count differs"
    page_texts = [page.extract_text() for page in reader.pages]
    for expected, actual in zip(FIXTURE_PAGES, page_texts, strict=True):
        for line in expected:
            assert line in actual, f"PDF fixture lost its authored text: {line}"

    report_dir = directory / "report"
    success = run_cli(directory, "selectable_pdf", fixture, report_dir)
    assert success["returncode"] == 0, success["stderr"]
    required = {"source.txt", "report.md", "report.html", "analysis.json", "manifest.json",
                "retrieval.json", "extraction.json", "plan.json"}
    assert required <= {path.name for path in report_dir.iterdir()}, "Report artifacts missing"
    analysis = json.loads((report_dir / "analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    source = (report_dir / "source.txt").read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert analysis["source"]["method"] == "pypdf_selectable_text"
    assert analysis["source"]["pdf_sha256"] == _sha(fixture)
    assert manifest["input_sha256"] == _sha(fixture)
    assert manifest["source_text_sha256"] == source_hash
    assert manifest["classifier"] == "rules"
    assert manifest["production_model_changed"] is False
    assert manifest["human_content_review_complete"] is False
    assert manifest["pipeline_quality_score"] is None
    assert manifest["retrieval"]["mode"] == "bm25"
    assert analysis["retrieval"]["grounding_validation"] == "source_hash_and_spans"
    for name, digest in manifest["code_sha256"].items():
        assert _sha(ROOT / name) == digest, f"Code changed during demo: {name}"

    pages = analysis["source"]["pages"]
    assert len(pages) == len(page_texts)
    for number, (page, expected_text) in enumerate(zip(pages, page_texts, strict=True), 1):
        assert page["page"] == number
        assert page["selectable_text_present"] is True
        assert source[page["char_start"]:page["char_end"]] == expected_text
    expected_join = "\n\n".join(page_texts)
    assert source == expected_join, "PDF page order or inter-page boundaries differ"

    fields = analysis["extraction"]["fields"]
    evidence = [item for rows in fields.values() for item in rows]
    hits = analysis["retrieval"]["hits"]
    assert evidence and hits, "Authored fixture produced no extraction/retrieval evidence"
    expected_quotes = [line for page in FIXTURE_PAGES for line in page[1:]]
    assert set(expected_quotes) <= {item["quote"] for item in evidence}, "Expected authored evidence omitted"
    quote_checks = []
    for item in evidence + hits:
        start, end = item["char_start"], item["char_end"]
        assert item["verified"] is True
        assert 0 <= start < end <= len(source)
        assert source[start:end] == item["quote"], "Exported quotation differs from source"
        assert item.get("source_sha256", source_hash) == source_hash
        overlapping = [page["page"] for page in pages
                       if start < page["char_end"] and end > page["char_start"]]
        assert overlapping, "Evidence has no source page"
        quote_checks.append({"evidence_id": item["evidence_id"], "pages": overlapping,
                             "char_start": start, "char_end": end,
                             "quote_sha256": hashlib.sha256(item["quote"].encode()).hexdigest()})

    html = (report_dir / "report.html").read_text(encoding="utf-8")
    parsed = _Links()
    parsed.feed(html)
    assert parsed.citations, "No clickable source citations in HTML"
    assert all(href.startswith("#src-") and href[1:] in parsed.ids for href in parsed.citations)
    assert not parsed.active_elements, "Self-contained HTML report has unexpected active elements"
    visible = "".join(parsed.text)
    assert all(quote in visible for quote in expected_quotes), "HTML does not embed expected source text"
    assert source_hash in visible

    before = _snapshot(report_dir)
    preserved = run_cli(directory, "existing_output", fixture, report_dir)
    assert preserved["returncode"] != 0
    assert "existing reports are never overwritten" in preserved["stderr"]
    assert _snapshot(report_dir) == before, "Existing output was modified"

    missing_output = directory / "missing_should_not_exist"
    missing = run_cli(directory, "missing_input", directory / "missing.pdf", missing_output)
    assert missing["returncode"] != 0
    assert "existing .md, .txt or .pdf file" in missing["stderr"]
    assert not missing_output.exists(), "Missing input produced an output directory"

    scanned = directory / "blank_requires_ocr.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with scanned.open("xb") as output:
        writer.write(output)
    scanned_output = directory / "scanned_should_not_exist"
    blank = run_cli(directory, "blank_pdf", scanned, scanned_output)
    assert blank["returncode"] != 0
    assert "no selectable text" in blank["stderr"]
    assert not scanned_output.exists(), "Image-only/blank PDF created a misleading report"

    receipt = {
        "schema_version": 1, "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Functional smoke only; authored fixture, no benchmark/final-test scoring",
        "runtime": {"python": sys.version, "executable": sys.executable, "pypdf": pypdf.__version__},
        "fixture": {"path": fixture.name, "sha256": _sha(fixture), "pages": len(pages),
                    "authored_expected_sentences": expected_quotes},
        "report_directory": "report", "report_files_sha256": _snapshot(report_dir),
        "source_sha256": source_hash, "page_boundaries_verified": pages,
        "extraction_items_verified": len(evidence), "retrieval_hits_verified": len(hits),
        "html_citation_links_verified": len(parsed.citations), "evidence_page_checks": quote_checks,
        "checks": {"pdf_selectable_text": True, "page_text_and_order_exact": True,
                   "expected_sentences_in_extraction_and_html": True,
                   "all_source_quotes_and_spans_exact": True, "all_html_citation_anchors_resolve": True,
                   "manifest_source_input_code_hashes_match": True, "existing_report_preserved": True,
                   "missing_input_rejected_without_output": True,
                   "blank_pdf_requires_ocr_without_output": True, "no_model_import_or_network": True},
        "cli_runs": {"selectable_pdf": success, "existing_output": preserved,
                     "missing_input": missing, "blank_pdf": blank},
        "limits": ["Blank PDF tests OCR-required rejection, not OCR quality or a corpus of scanned PDFs.",
                   "PDF reading order and visual layout quality need human review on real documents.",
                   "Source-exact quotations do not establish correct semantic labels or complete retrieval.",
                   "This demo uses rules and BM25; it does not evaluate a classifier or a trained dense retriever.",
                   "No final-test data, model weights, production index, or classifier configuration are changed."],
        "verifier_sha256": _sha(Path(__file__)),
        "pdf_preprocessing_sha256": _sha(ROOT / "pdf_preprocessing.py"),
    }
    _write_json(directory / "smoke_receipt.json", receipt)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "rag_ready")
    args = parser.parse_args(argv)
    directory = verify_pdf_demo(args.output_root)
    print(json.dumps({"status": "passed", "output_directory": str(directory),
                      "report_html": str(directory / "report" / "report.html"),
                      "receipt": str(directory / "smoke_receipt.json")}, indent=2))


if __name__ == "__main__":
    main()
