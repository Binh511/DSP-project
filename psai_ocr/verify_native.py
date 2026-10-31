"""Actual local Windows OCR smoke using an authored image-only PDF fixture.

This checks functionality and transcript fidelity for a clear English fixture.
It is not a held-out corpus evaluation or a claim of production OCR accuracy.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile

from . import extract_pdf_with_ocr


def _distance(left, right):
    row = list(range(len(right) + 1))
    for i, lhs in enumerate(left, 1):
        next_row = [i]
        for j, rhs in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (lhs != rhs)))
        row = next_row
    return row[-1]


def _normalize(text):
    return ' '.join(re.findall(r"[a-z0-9]+", text.casefold()))


def verify(output_root: str | Path | None = None) -> dict:
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    root = Path(__file__).resolve().parents[1]
    base = Path(output_root) if output_root else root / 'outputs/ocr_verification'
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = Path(tempfile.mkdtemp(prefix=timestamp + '_', dir=base))
    lines = [
        'PROJECT REQUIREMENTS',
        'The system shall provide searchable project documents.',
        'Users must verify every source citation.',
        'EXPECTED RESULTS',
        'Deliver a working report with evidence links.',
    ]
    image = Image.new('RGB', (1600, 1000), 'white')
    draw = ImageDraw.Draw(image)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    if not font_path.is_file():
        raise RuntimeError('This authored OCR fixture requires the installed Arial font')
    font = ImageFont.truetype(str(font_path), 44)
    for index, line in enumerate(lines):
        draw.text((90, 110 + index * 130), line, font=font, fill='black')
    image.save(directory / 'scan_source.png')
    scan_pdf = directory / 'image_only.pdf'
    image.save(scan_pdf, format='PDF', resolution=144.0)
    image.close()
    truth = '\n'.join(lines)
    (directory / 'authored_transcript.txt').write_text(truth + '\n', encoding='utf-8')
    assert not (PdfReader(str(scan_pdf)).pages[0].extract_text() or '').strip()
    none_rejected = False
    try:
        extract_pdf_with_ocr(scan_pdf, mode='none')
    except ValueError as exc:
        none_rejected = 'no usable text' in str(exc)
    assert none_rejected, 'Image-only PDF must not be treated as selectable text'
    native = 'The project shall retain this original selectable text and its exact source citation.'
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font_object = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                    NameObject('/Subtype'): NameObject('/Type1'),
                                    NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font_object})})
    content = DecodedStreamObject()
    content.set_data(('BT /F1 10 Tf 25 700 Td (' + native + ') Tj ET').encode('ascii'))
    page[NameObject('/Contents')] = content
    scan_reader = PdfReader(str(scan_pdf))
    writer.add_page(scan_reader.pages[0])
    mixed_pdf = directory / 'mixed_selectable_scan.pdf'
    with mixed_pdf.open('wb') as handle:
        writer.write(handle)
    result = extract_pdf_with_ocr(mixed_pdf, mode='auto', backend='windows', cache_dir=directory / 'pages')
    first, second = result['pages']
    assert not first['ocr_attempted'], 'Selectable first page must be preserved'
    assert result['text'][first['char_start']:first['char_end']].strip() == native
    assert second['method'] == 'local_ocr', 'Scanned second page must use real OCR'
    prediction = result['text'][second['char_start']:second['char_end']]
    ref, pred = _normalize(truth), _normalize(prediction)
    cer = _distance(ref, pred) / max(1, len(ref))
    wer = _distance(ref.split(), pred.split()) / max(1, len(ref.split()))
    for segment in second['ocr_segments']:
        assert result['text'][segment['char_start']:segment['char_end']] == segment['text']
    assert second['ocr_engine']['name'] == 'Windows.Media.Ocr'
    assert second['mean_ocr_confidence'] is None
    assert second['word_boxes_available'] and len(second['ocr_words']) > 10
    (directory / 'extraction.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'recognized.txt').write_text(prediction + '\n', encoding='utf-8')
    receipt = {
        'status': 'passed' if cer <= .15 and wer <= .20 else 'quality_review_required',
        'scope': 'authored_clear_English_functional_smoke_not_final_test',
        'real_ocr_engine_executed': True, 'engine': second['ocr_engine'],
        'pdf': str(mixed_pdf), 'directory': str(directory),
        'image_only_none_rejected': none_rejected,
        'selectable_page_retained': True, 'ocr_pages': result['ocr_pages'],
        'word_boxes': len(second['ocr_words']), 'line_boxes': len(second['ocr_segments']),
        'normalized_character_error_rate': cer, 'normalized_word_error_rate': wer,
        'normalization': 'casefold, alphanumeric tokens and whitespace; punctuation excluded',
        'confidence_available': False,
        'pdf_visual_review_complete': False,
        'policy': 'PowerShell process-only ExecutionPolicy Bypass for reviewed workspace helper; no stored policy change',
        'network_used': False, 'dependencies_installed': False,
        'notes': ['One clear authored fixture is not a representative OCR benchmark.',
                  'Windows API does not expose model weights or word confidences.'],
    }
    (directory / 'smoke_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=True, indent=2))
