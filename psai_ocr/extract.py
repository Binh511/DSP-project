"""Page-level PDF extraction with explicit, local OCR and source provenance.

No model download, API call or caption generation occurs here. Coordinates are
pixels in the rendered page; character spans refer to the returned text only.
"""
from __future__ import annotations

from contextlib import suppress
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Callable, Literal


class OCRDependencyError(RuntimeError):
    """An optional local OCR component or model is unavailable."""


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unavailable'


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PDFiumRenderer:
    """Render one page at a time, closing native PDFium resources promptly."""

    name = 'pypdfium2'

    def __init__(self, path: Path):
        try:
            import pypdfium2
        except ImportError as exc:
            raise OCRDependencyError(
                'Local PDF OCR needs pypdfium2. Install psai_ocr/requirements.txt '
                'in the project runtime environment; no external Poppler is required.'
            ) from exc
        self.document = pypdfium2.PdfDocument(str(path))
        self.version = _version('pypdfium2')

    def render(self, index: int, *, dpi: int, max_pixels: int):
        page = self.document[index]
        bitmap = None
        try:
            width, height = page.get_size()
            scale = dpi / 72.0
            if width <= 0 or height <= 0 or not math.isfinite(width * height):
                raise ValueError(f'PDF page {index + 1} has invalid dimensions')
            # ceil is used by PDFium. Reserve a small margin to enforce the cap.
            if math.ceil(width * scale) * math.ceil(height * scale) > max_pixels:
                scale = math.sqrt(max_pixels / (width * height)) * 0.99
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert('RGB').copy()
            if image.width * image.height > max_pixels:
                image.close()
                raise ValueError('Rendered page exceeded the configured pixel limit')
            return image
        finally:
            if bitmap is not None:
                bitmap.close()
            page.close()

    def close(self):
        self.document.close()


class RapidOCREngine:
    """CPU ONNX OCR, restricted to already installed/bundled model files.

Bundled Chinese/English recognizers are used for English only in this release.
Vietnamese is deliberately not advertised until a suitable model is supplied
and measured. Line boxes and any available word boxes remain distinct.
"""

    def __init__(self, *, language: str = 'en', model_dir: str | Path | None = None):
        if language != 'en':
            raise ValueError('This OCR release is configured only for English (en). '
                             'Do not use its output as verified Vietnamese OCR.')
        try:
            import rapidocr
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OCRDependencyError(
                'Local OCR needs rapidocr, onnxruntime and Pillow. Install '
                'psai_ocr/requirements.txt in the project runtime environment. '
                'Use the current rapidocr package; rapidocr-onnxruntime 1.4.4 '
                'does not support Python 3.13.'
            ) from exc
        self.model_dir = (Path(model_dir) if model_dir else
                          Path(rapidocr.__file__).resolve().parent / 'models')
        choices = {}
        for task in ('det', 'cls', 'rec'):
            candidates = sorted(path for path in self.model_dir.glob('*.onnx')
                                if re.search(rf'(?:^|_){task}(?:_|\.)', path.name))
            if len(candidates) != 1:
                raise OCRDependencyError(
                    f'Expected one local _{task}.onnx or _{task}_*.onnx model in {self.model_dir}; '
                    f'found {len(candidates)}. Install a RapidOCR wheel with bundled '
                    'models or provide a reviewed local model directory. '
                    'This extractor will not download missing models.'
                )
            choices[task] = candidates[0]
        self.metadata = {
            'name': 'RapidOCR', 'version': _version('rapidocr'),
            'runtime': 'onnxruntime', 'runtime_version': _version('onnxruntime'),
            'device': 'cpu', 'language': language,
            'language_model_validation_complete': False,
            'models': [{'task': task, 'filename': path.name,
                        'sha256': _file_hash(path)} for task, path in choices.items()],
        }
        params = {
            'Global.log_level': 'warning', 'Global.return_word_box': True,
            'Global.model_root_dir': str(self.model_dir),
            'EngineConfig.onnxruntime.use_cuda': False,
            'EngineConfig.onnxruntime.intra_op_num_threads': 2,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        }
        params.update({f'{task.title()}.model_path': str(path)
                       for task, path in choices.items()})
        # Explicit paths prevent RapidOCR's automatic model-resolver download.
        self.engine = RapidOCR(params=params)

    def recognize(self, image) -> dict:
        import numpy as np
        output = self.engine(np.asarray(image.convert('RGB')), return_word_box=True)
        texts = getattr(output, 'txts', None)
        if texts is None:
            return {'segments': [], 'words': [], 'engine': self.metadata}
        scores = getattr(output, 'scores', None)
        boxes = getattr(output, 'boxes', None)
        if scores is None or boxes is None or len(texts) != len(scores) or len(texts) != len(boxes):
            raise RuntimeError('OCR engine returned mismatched texts, scores and boxes')
        segments = [dict(text=str(text), confidence=float(score),
                         polygon=box.tolist() if hasattr(box, 'tolist') else box)
                    for text, score, box in zip(texts, scores, boxes) if str(text).strip()]
        words = []
        # RapidOCR word_results contains lines, each containing (text, score, box).
        for line_index, word_line in enumerate(getattr(output, 'word_results', ()) or ()):
            if not isinstance(word_line, (tuple, list)):
                continue
            for item in word_line:
                if not isinstance(item, (tuple, list)) or len(item) != 3:
                    continue
                word, score, box = item
                if box is not None and str(word).strip():
                    words.append(dict(text=str(word), confidence=float(score),
                                      polygon=box.tolist() if hasattr(box, 'tolist') else box,
                                      segment_index=line_index))
        return {'segments': segments, 'words': words, 'engine': self.metadata}


def _checked_segments(result: dict, image) -> tuple[str, list[dict], list[dict]]:
    segments, cursor = [], 0
    for raw in result.get('segments', []):
        text = raw.get('text')
        score = raw.get('confidence')
        polygon = raw.get('polygon')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('OCR returned an invalid text segment')
        if score is not None and (not isinstance(score, (float, int)) or
                                  not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('OCR confidence must be finite and between 0 and 1')
        if not isinstance(polygon, (list, tuple)) or len(polygon) != 4:
            raise ValueError('OCR polygons must have four points')
        for point in polygon:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
                raise ValueError('OCR polygon coordinates must be finite pixel coordinates')
        if segments:
            cursor += 1
        segments.append(dict(text=text, confidence=float(score) if score is not None else None, polygon=polygon,
                             char_start=cursor, char_end=cursor + len(text),
                             granularity='line', coordinate_space='rendered_page_pixels'))
        cursor += len(text)
    text = '\n'.join(segment['text'] for segment in segments)
    words = []
    for raw in result.get('words', []):
        word = raw.get('text')
        index = raw.get('segment_index')
        if not isinstance(index, int) or not 0 <= index < len(segments) or not isinstance(word, str):
            continue
        line = segments[index]
        # Word geometry can be retained without inventing a text span if the OCR
        # recognizer's token grouping does not map back exactly to the line.
        occurrence = line['text'].find(word)
        item = dict(raw, coordinate_space='rendered_page_pixels', granularity='word')
        if occurrence >= 0 and line['text'].count(word) == 1:
            item.update(char_start=line['char_start'] + occurrence,
                        char_end=line['char_start'] + occurrence + len(word))
        words.append(item)
    return text, segments, words


def extract_pdf_with_ocr(
    path: str | Path, *, mode: Literal['none', 'auto', 'always'] = 'auto',
    backend: Literal['auto', 'rapidocr', 'windows'] = 'auto',
    language: str = 'en', dpi: int = 180, min_text_chars: int = 32,
    max_pixels: int = 20_000_000, max_pages: int = 2000,
    model_dir: str | Path | None = None, cache_dir: str | Path | None = None,
    reader_factory: Callable | None = None, renderer_factory: Callable | None = None,
    engine_factory: Callable | None = None,
) -> dict:
    """Extract text with OCR only where explicitly requested.

    ``auto`` renders pages with fewer than ``min_text_chars`` non-whitespace
    characters; selectable text wins if OCR returns less text. ``always`` uses
    OCR text on each page and records native text as a fallback if OCR is empty.
    OCR recognition is unverified evidence and never silently promoted to gold.
    The optional factories support deterministic dependency-free regression tests.
    """
    started = perf_counter()
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.casefold() != '.pdf':
        raise ValueError('Input must be an existing PDF file')
    if mode not in ('none', 'auto', 'always'):
        raise ValueError('OCR mode must be none, auto or always')
    if backend not in ('auto', 'rapidocr', 'windows'):
        raise ValueError('OCR backend must be auto, rapidocr or windows')
    if language != 'en':
        raise ValueError('Only English OCR is supported by this release; use language=en')
    for name, value, lower, upper in (
        ('dpi', dpi, 72, 600), ('min_text_chars', min_text_chars, 0, 10000),
        ('max_pixels', max_pixels, 10000, 100_000_000), ('max_pages', max_pages, 1, 10000),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer between {lower} and {upper}')
    if reader_factory is None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise OCRDependencyError('PDF reading needs pypdf. Install psai_ocr/requirements.txt.') from exc
        reader_factory = PdfReader
    reader = reader_factory(str(path))
    if getattr(reader, 'is_encrypted', False):
        with suppress(Exception):
            reader.close()
        raise ValueError('Encrypted PDF requires an explicitly decrypted local copy')
    if not 0 < len(reader.pages) <= max_pages:
        with suppress(Exception):
            reader.close()
        raise ValueError(f'PDF must contain between 1 and {max_pages} pages')
    pdf_sha = _file_hash(path)
    renderer = engine = None
    parts, pages, warnings, cursor = [], [], [], 0
    try:
        for index, page in enumerate(reader.pages):
            page_start = perf_counter()
            native = page.extract_text() or ''
            if not isinstance(native, str):
                raise TypeError(f'Page {index + 1} did not return selectable text')
            native_chars = sum(not ch.isspace() for ch in native)
            needs_ocr = mode == 'always' or (mode == 'auto' and native_chars < min_text_chars)
            selected = native
            info = dict(page=index + 1, selectable_text_present=bool(native.strip()),
                        selectable_nonspace_chars=native_chars,
                        selectable_text_sha256=sha256(native.encode('utf-8')).hexdigest(),
                        method='selectable_text', ocr_attempted=needs_ocr,
                        pdf_visual_review_complete=False)
            if needs_ocr:
                if renderer is None and renderer_factory is None and engine_factory is None:
                    if backend == 'windows':
                        from .windows import WindowsPDFRenderer, WindowsOCREngine
                        renderer = WindowsPDFRenderer(path, language=language)
                        engine = WindowsOCREngine(language=language, model_dir=model_dir)
                    else:
                        try:
                            renderer = PDFiumRenderer(path)
                            engine = RapidOCREngine(language=language, model_dir=model_dir)
                        except OCRDependencyError as rapid_error:
                            if backend != 'auto' or model_dir is not None:
                                raise
                            if renderer is not None:
                                renderer.close()
                                renderer = None
                            from .windows import WindowsPDFRenderer, WindowsOCREngine
                            renderer = WindowsPDFRenderer(path, language=language)
                            engine = WindowsOCREngine(language=language)
                            warnings.append('Using installed Windows native OCR because optional '
                                            f'RapidOCR/PDFium dependencies are unavailable: {rapid_error}')
                if renderer is None:
                    renderer = (renderer_factory or PDFiumRenderer)(path)
                if engine is None:
                    engine = (engine_factory or RapidOCREngine)(language=language, model_dir=model_dir)
                image = renderer.render(index, dpi=dpi, max_pixels=max_pixels)
                try:
                    encoded = BytesIO()
                    image.save(encoded, format='PNG')
                    image_bytes = encoded.getvalue()
                    image_sha = sha256(image_bytes).hexdigest()
                    info.update(image_sha256=image_sha, image_sha256_format='PNG',
                                image_width=image.width, image_height=image.height,
                                requested_dpi=dpi, renderer=getattr(renderer, 'name', type(renderer).__name__),
                                renderer_version=getattr(renderer, 'version', 'injected_test_renderer'))
                    if cache_dir is not None:
                        directory = Path(cache_dir).resolve()
                        directory.mkdir(parents=True, exist_ok=True)
                        target = directory / f'{image_sha}.png'
                        if target.exists():
                            if _file_hash(target) != image_sha:
                                raise ValueError('Existing rendered page cache has an invalid hash')
                        else:
                            with target.open('xb') as handle:
                                handle.write(image_bytes)
                        info['image_path'] = str(target)
                    result = engine.recognize(image)
                    ocr_text, segments, words = _checked_segments(result, image)
                    confidences = [segment['confidence'] for segment in segments
                                   if segment['confidence'] is not None]
                    info.update(ocr_text_sha256=sha256(ocr_text.encode('utf-8')).hexdigest(),
                                ocr_engine=result.get('engine', {}),
                                ocr_segments=segments, ocr_words=words,
                                word_boxes_available=bool(words),
                                ocr_confidence_available=bool(confidences),
                                mean_ocr_confidence=(sum(confidences) / len(confidences)
                                                     if confidences else None))
                    if ocr_text.strip() and (mode == 'always' or
                                             sum(not c.isspace() for c in ocr_text) >= native_chars):
                        selected = ocr_text
                        info['method'] = 'local_ocr'
                        warnings.append(f'Page {index + 1}: OCR text requires visual verification against PDF.')
                    else:
                        info['method'] = 'selectable_text_ocr_fallback' if native.strip() else 'empty'
                        warnings.append(f'Page {index + 1}: OCR yielded no usable improvement; review the PDF.')
                finally:
                    image.close()
            elif not native.strip():
                info['method'] = 'empty'
                warnings.append(f'Page {index + 1}: no selectable text and OCR was disabled.')
            if parts:
                parts.append('\n\n')
                cursor += 2
            info.update(char_start=cursor, char_end=cursor + len(selected),
                        source_text_sha256=sha256(selected.encode('utf-8')).hexdigest())
            if info['method'] == 'local_ocr':
                for item in info['ocr_segments'] + info['ocr_words']:
                    if 'char_start' in item:
                        item['char_start'] += cursor
                        item['char_end'] += cursor
                    item['span_space'] = 'extracted_document_text'
            else:
                # OCR attempts remain auditable, but their spans refer to the OCR
                # candidate rather than to the selectable text that was retained.
                for item in info.get('ocr_segments', []) + info.get('ocr_words', []):
                    item['span_space'] = 'page_ocr_candidate'
            parts.append(selected)
            cursor += len(selected)
            info['elapsed_seconds'] = round(perf_counter() - page_start, 6)
            pages.append(info)
    finally:
        if renderer is not None:
            with suppress(Exception):
                renderer.close()
        # PdfReader may keep an input stream open; close it when supported.
        with suppress(Exception):
            reader.close()
    text = ''.join(parts)
    if not text.strip():
        raise ValueError('PDF yielded no usable text. Check OCR dependencies, page language, '
                         'image quality and the original PDF; no text was invented.')
    for info in pages:
        info['line_start'] = text.count('\n', 0, info['char_start']) + 1
        info['line_end'] = text.count('\n', 0, max(info['char_start'], info['char_end'] - 1)) + 1
    ocr_pages = sum(info['method'] == 'local_ocr' for info in pages)
    warnings.append('Reading order, tables and headings require comparison with the PDF.')
    return dict(text=text, pages=pages, source_path=str(path), pdf_sha256=pdf_sha,
                source_text_sha256=sha256(text.encode('utf-8')).hexdigest(),
                method='local_ocr_mixed' if ocr_pages else 'pypdf_selectable_text',
                ocr_mode=mode, ocr_backend=backend, ocr_pages=ocr_pages, language=language,
                pypdf_version=_version('pypdf'), warnings=warnings,
                pdf_visual_review_complete=False,
                elapsed_seconds=round(perf_counter() - started, 6))
