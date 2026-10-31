"""Optional local OCR. Importing this package loads no model or PDF renderer."""

from .extract import OCRDependencyError, extract_pdf_with_ocr

__all__ = ['OCRDependencyError', 'extract_pdf_with_ocr']
