"""Meaningful OCR boundary tests. These do not claim a real OCR engine ran."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psai_ocr import OCRDependencyError, extract_pdf_with_ocr


class Page:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class Reader:
    is_encrypted = False

    def __init__(self, values):
        self.pages = [Page(value) for value in values]
        self.closed = False

    def close(self):
        self.closed = True


class Renderer:
    version = 'test-renderer'

    def __init__(self):
        self.calls = []
        self.closed = False

    def render(self, index, *, dpi, max_pixels):
        from PIL import Image
        self.calls.append(index)
        return Image.new('RGB', (200, 100), 'white')

    def close(self):
        self.closed = True


class Engine:
    def __init__(self, texts):
        self.texts = iter(texts)

    def recognize(self, image):
        text = next(self.texts)
        return {'segments': [dict(text=text, confidence=0.91,
                                 polygon=[[0, 0], [150, 0], [150, 25], [0, 25]])] if text else [],
                'words': [], 'engine': {'name': 'injected_fake', 'device': 'cpu'}}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.directory.name)
        self.pdf = self.root / 'scan.pdf'
        self.pdf.write_bytes(b'%PDF-1.7\nFAKE FOR INJECTED UNIT TEST ONLY\n')

    def tearDown(self):
        self.directory.cleanup()

    def run_extraction(self, native, recognized, **kwargs):
        self.reader = Reader(native)
        self.renderer = Renderer()
        self.engine = Engine(recognized)
        return extract_pdf_with_ocr(
            self.pdf, reader_factory=lambda _: self.reader,
            renderer_factory=lambda _: self.renderer,
            engine_factory=lambda **_: self.engine, **kwargs)

    def test_auto_keeps_native_page_and_only_renders_sparse_page(self):
        native = 'This selectable page contains enough native text to preserve exactly.\nSecond line.'
        recognized = 'The scanned project requires a working offline document retrieval service.'
        result = self.run_extraction([native, ''], [recognized])
        self.assertEqual(result['text'], native + '\n\n' + recognized)
        self.assertEqual(self.renderer.calls, [1])
        self.assertEqual(result['ocr_pages'], 1)
        self.assertTrue(self.reader.closed)
        self.assertTrue(self.renderer.closed)
        self.assertFalse(result['pdf_visual_review_complete'])

    def test_none_never_initializes_ocr(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('OCR must not initialize')
        result = extract_pdf_with_ocr(
            self.pdf, mode='none', reader_factory=lambda _: Reader(['Selectable text']),
            renderer_factory=forbidden, engine_factory=forbidden)
        self.assertEqual(result['ocr_pages'], 0)
        self.assertEqual(result['method'], 'pypdf_selectable_text')

    def test_none_reports_empty_page_without_inventing_text(self):
        result = self.run_extraction(['first', ''], [], mode='none')
        self.assertEqual(result['text'], 'first\n\n')
        self.assertEqual(result['pages'][1]['method'], 'empty')
        self.assertTrue(any('disabled' in value for value in result['warnings']))

    def test_all_empty_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'no usable text'):
            self.run_extraction([''], [''])
        self.assertTrue(self.renderer.closed)
        self.assertTrue(self.reader.closed)

    def test_always_is_explicit_replacement(self):
        result = self.run_extraction(['A complete selectable page text is still present.'], ['OCR output'], mode='always')
        self.assertEqual(result['text'], 'OCR output')
        self.assertEqual(self.renderer.calls, [0])

    def test_auto_retains_native_when_ocr_has_less_text(self):
        result = self.run_extraction(['Native short text'], ['No'])
        self.assertEqual(result['text'], 'Native short text')
        self.assertEqual(result['pages'][0]['method'], 'selectable_text_ocr_fallback')
        self.assertEqual(result['pages'][0]['ocr_segments'][0]['span_space'], 'page_ocr_candidate')

    def test_page_and_ocr_spans_roundtrip_exactly(self):
        result = self.run_extraction(['Header\nNative first page contains a complete selectable paragraph.', ''], ['Scanned requirement line.'])
        text = result['text']
        for page in result['pages']:
            selected = text[page['char_start']:page['char_end']]
            self.assertEqual(sha256(selected.encode()).hexdigest(), page['source_text_sha256'])
            self.assertEqual(page['line_start'], text.count('\n', 0, page['char_start']) + 1)
            for segment in page.get('ocr_segments', []):
                self.assertEqual(text[segment['char_start']:segment['char_end']], segment['text'])

    def test_hashes_match_pdf_text_and_rendered_png(self):
        result = self.run_extraction([''], ['Scanned project information'], cache_dir=self.root / 'cache')
        page = result['pages'][0]
        self.assertEqual(result['pdf_sha256'], sha256(self.pdf.read_bytes()).hexdigest())
        self.assertEqual(result['source_text_sha256'], sha256(result['text'].encode()).hexdigest())
        self.assertEqual(sha256(Path(page['image_path']).read_bytes()).hexdigest(), page['image_sha256'])
        self.assertEqual(page['image_sha256_format'], 'PNG')
        json.dumps(result, allow_nan=False)

    def test_existing_cache_hash_is_verified(self):
        result = self.run_extraction([''], ['Scanned content'], cache_dir=self.root / 'cache')
        Path(result['pages'][0]['image_path']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'cache.*hash'):
            self.run_extraction([''], ['Scanned content'], cache_dir=self.root / 'cache')

    def test_invalid_ocr_confidence_rejected(self):
        class BadEngine:
            def recognize(self, image):
                return {'segments': [dict(text='bad', confidence=float('nan'),
                                          polygon=[[0, 0]] * 4)]}
        self.reader = Reader([''])
        self.renderer = Renderer()
        with self.assertRaisesRegex(ValueError, 'confidence'):
            extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: self.reader,
                                 renderer_factory=lambda _: self.renderer,
                                 engine_factory=lambda **_: BadEngine())
        self.assertTrue(self.renderer.closed)

    def test_model_load_error_closes_renderer(self):
        self.renderer = Renderer()
        def unavailable(**_):
            raise OCRDependencyError('Not installed')
        with self.assertRaisesRegex(OCRDependencyError, 'Not installed'):
            extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: Reader(['']),
                                 renderer_factory=lambda _: self.renderer,
                                 engine_factory=unavailable)
        self.assertTrue(self.renderer.closed)

    def test_encrypted_pdf_rejected(self):
        reader = Reader(['protected'])
        reader.is_encrypted = True
        with self.assertRaisesRegex(ValueError, 'Encrypted'):
            extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: reader)

    def test_invalid_mode_and_limits_rejected(self):
        for config in ({'mode': 'magic'}, {'backend': 'remote'}, {'dpi': 0}, {'max_pixels': 50},
                       {'min_text_chars': -1}, {'dpi': True}, {'max_pages': 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_extraction(['text'], [], **config)

    def test_untested_language_rejected(self):
        with self.assertRaisesRegex(ValueError, 'English'):
            self.run_extraction([''], [''], language='vi')

    def test_page_limit_prevents_unbounded_work(self):
        with self.assertRaisesRegex(ValueError, 'between 1 and 1 pages'):
            self.run_extraction(['one', 'two'], [], max_pages=1)

    def test_non_pdf_input_rejected(self):
        with self.assertRaisesRegex(ValueError, 'existing PDF'):
            extract_pdf_with_ocr(self.root / 'missing.txt')

    def test_real_selectable_pdf_without_any_ocr_dependency(self):
        try:
            from pypdf import PdfWriter
            from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        except ImportError:
            self.skipTest('Optional pypdf is unavailable')
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 14 Tf 50 700 Td (A project shall provide searchable source evidence.) Tj ET')
        page[NameObject('/Contents')] = stream
        real = self.root / 'real_selectable.pdf'
        with real.open('wb') as handle:
            writer.write(handle)
        result = extract_pdf_with_ocr(real, mode='auto')
        self.assertIn('searchable source evidence', result['text'])
        self.assertFalse(result['pages'][0]['ocr_attempted'])

    def test_real_image_only_pdf_fails_clearly_without_ocr(self):
        try:
            import pypdf
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest('Optional pypdf/Pillow is unavailable')
        image = Image.new('RGB', (800, 400), 'white')
        ImageDraw.Draw(image).text((40, 100), 'SCANNED PROJECT REQUIREMENT', fill='black')
        real = self.root / 'real_scan.pdf'
        image.save(real, format='PDF')
        image.close()
        with self.assertRaisesRegex(ValueError, 'no usable text'):
            extract_pdf_with_ocr(real, mode='none')

    def test_auto_uses_windows_only_for_missing_optional_dependency(self):
        renderer = Renderer()
        engine = Engine(['Native Windows recognized text'])
        with patch('psai_ocr.extract.PDFiumRenderer', side_effect=OCRDependencyError('PDFium missing')), \
                patch('psai_ocr.windows.WindowsPDFRenderer', return_value=renderer), \
                patch('psai_ocr.windows.WindowsOCREngine', return_value=engine):
            result = extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: Reader(['']))
        self.assertEqual(result['ocr_pages'], 1)
        self.assertTrue(any('Using installed Windows' in item for item in result['warnings']))

    def test_runtime_render_failure_does_not_silently_switch_backend(self):
        with patch('psai_ocr.extract.PDFiumRenderer', side_effect=RuntimeError('PDF corrupted')), \
                patch('psai_ocr.windows.WindowsPDFRenderer') as native:
            with self.assertRaisesRegex(RuntimeError, 'PDF corrupted'):
                extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: Reader(['']))
            native.assert_not_called()

    def test_native_confidence_is_null_without_invented_scores(self):
        class NativeEngine:
            def recognize(self, image):
                return {'segments': [dict(text='Windows OCR example', confidence=None,
                                          polygon=[[0, 0], [180, 0], [180, 25], [0, 25]])],
                        'words': [dict(text='Windows', confidence=None, segment_index=0,
                                       polygon=[[0, 0], [70, 0], [70, 25], [0, 25]])],
                        'engine': {'name': 'fake_winrt_boundary', 'confidence_available': False}}
        renderer = Renderer()
        result = extract_pdf_with_ocr(self.pdf, reader_factory=lambda _: Reader(['']),
                                      renderer_factory=lambda _: renderer,
                                      engine_factory=lambda **_: NativeEngine())
        page = result['pages'][0]
        self.assertIsNone(page['mean_ocr_confidence'])
        self.assertFalse(page['ocr_confidence_available'])
        self.assertTrue(page['word_boxes_available'])
        word = page['ocr_words'][0]
        self.assertEqual(result['text'][word['char_start']:word['char_end']], 'Windows')
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
