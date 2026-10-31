"""Local selectable-text PDF extraction, with explicit OCR limitations.

Offsets downstream refer to the returned text, never to PDF coordinates.
No external model or generative service is called by this adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import hashlib


def extract_pdf(path: str | Path, *, reader_factory: Callable | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != '.pdf':
        raise ValueError('Input must be an existing PDF file')
    if reader_factory is None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError('PDF support requires the optional pypdf dependency. Install requirements-pdf.txt; Markdown/text input is available without it.') from exc
        reader_factory = PdfReader
    reader = reader_factory(str(path))
    if getattr(reader, 'is_encrypted', False):
        raise ValueError('Encrypted PDF requires an explicitly decrypted local copy')
    parts, pages, cursor = [], [], 0
    for number, page in enumerate(reader.pages, 1):
        value = page.extract_text() or ''
        if not isinstance(value, str):
            raise TypeError(f'Page {number} did not return text')
        if parts:
            parts.append('\n\n')
            cursor += 2
        pages.append(dict(page=number, char_start=cursor, char_end=cursor + len(value),
                          selectable_text_present=bool(value.strip())))
        parts.append(value)
        cursor += len(value)
    text = ''.join(parts)
    if not text.strip():
        raise ValueError('PDF contains no selectable text; an OCR step and source review are required')
    return dict(text=text, pages=pages, source_path=str(path),
                pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method='pypdf_selectable_text',
                warnings=['Reading order, tables and headings require comparison with the PDF.',
                          'Empty pages may contain images requiring OCR. No OCR was performed.'],
                pdf_visual_review_complete=False)
