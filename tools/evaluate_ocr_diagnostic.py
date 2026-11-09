"""Measure installed en-US OCR on AI-authored Vietnamese/English scan fixtures.

The reference is fixed in this source before rendering or recognition. This is
a synthetic language-limit diagnostic, not human gold or an independent test.
No model, Windows language capability, or dependency is installed/downloaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_VERSION = 'psai-synthetic-ocr-language-diagnostic-v1'
REFERENCE_ORIGIN = 'fixed_AI_authored_transcript_before_render_and_OCR'
PAGE_SIZE = (1600, 2100)
RASTER_DPI = 144
FONT_SIZE = 36


def fixture_specs():
    """Return fresh reference content, paired by topic and page layout."""
    return [
        {
            'case_id': 'vi_prose', 'language': 'vi', 'pair_id': 'prose', 'layout': 'paragraphs',
            'title': 'ĐỀ XUẤT BẢO TRÌ THIẾT BỊ PHÒNG THÍ NGHIỆM',
            'paragraphs': [
                'Phòng thí nghiệm cần theo dõi tình trạng thiết bị, yêu cầu bảo trì và lịch sử kiểm tra.',
                'Giảng viên phải biết máy nào không sử dụng được trước khi chuẩn bị buổi thực hành.',
                'Kỹ thuật viên ghi rõ nguyên nhân hỏng, người xử lý và thời điểm thay đổi trạng thái.',
                'Dữ liệu được lưu trên máy tính cục bộ. Người phụ trách kiểm tra bản sao lưu trước khi nghiệm thu.',
                'Đây là tình huống giả lập để đo lỗi nhận dạng chữ, không phải kết quả triển khai thực tế.',
            ],
        },
        {
            'case_id': 'en_prose', 'language': 'en', 'pair_id': 'prose', 'layout': 'paragraphs',
            'title': 'LABORATORY EQUIPMENT MAINTENANCE PROPOSAL',
            'paragraphs': [
                'The laboratory needs to track equipment condition, maintenance requests and inspection history.',
                'Teaching staff must know which instruments are unavailable before preparing a practical class.',
                'A technician records the fault reason, responsible person and time of each status change.',
                'Data is stored on a local computer. The coordinator checks the backup before acceptance.',
                'This synthetic scenario measures recognition errors and is not a real deployment result.',
            ],
        },
        {
            'case_id': 'vi_numbers', 'language': 'vi', 'pair_id': 'numbers', 'layout': 'paragraphs',
            'title': 'YÊU CẦU NGHIỆP VỤ VÀ SỐ LIỆU',
            'paragraphs': [
                'Mã dự án: LAB-2027. Phạm vi thử nghiệm gồm 60 thiết bị trong 2 phòng thực hành.',
                'Đội dự án gồm 2 lập trình viên và 1 cán bộ điều phối. Ngân sách dự kiến là 24.000.000 VND.',
                'Ngày kiểm tra thiết kế: 18/01/2027. Ngày bàn giao dự kiến: 15/02/2027.',
                'Mỗi thiết bị có mã duy nhất, tên thiết bị, số phòng và trạng thái sử dụng.',
                'Hệ thống phải từ chối mã bị trùng, bảo vệ tài khoản và ghi nhật ký thay đổi.',
                'Tiêu chí nghiệm thu: khôi phục đủ 60 bản ghi từ bản sao lưu, không làm mất lịch sử bảo trì.',
            ],
        },
        {
            'case_id': 'en_numbers', 'language': 'en', 'pair_id': 'numbers', 'layout': 'paragraphs',
            'title': 'BUSINESS REQUIREMENTS AND FIGURES',
            'paragraphs': [
                'Project code: LAB-2027. The pilot covers 60 instruments in 2 teaching rooms.',
                'The team has 2 developers and 1 staff coordinator. The proposed budget is 24.000.000 VND.',
                'Design inspection date: 18/01/2027. Proposed handover date: 15/02/2027.',
                'Every instrument has a unique identifier, equipment name, room number and service status.',
                'The system must reject duplicate identifiers, protect accounts and retain a change log.',
                'Acceptance requires restoring all 60 records from backup without losing maintenance history.',
            ],
        },
        {
            'case_id': 'vi_table', 'language': 'vi', 'pair_id': 'table', 'layout': 'table',
            'title': 'BẢNG KẾ HOẠCH KIỂM TRA',
            'paragraphs': ['Đọc bảng theo từng hàng, từ trái sang phải. Các ngày là giả định của tình huống.'],
            'headers': ['Mã', 'Thiết bị', 'Số lượng', 'Ngày kiểm tra'],
            'rows': [
                ['A-01', 'Máy hiện sóng', '12', '18/01/2027'],
                ['A-02', 'Bộ nguồn điện', '8', '25/01/2027'],
                ['A-03', 'Máy phát tín hiệu', '6', '15/02/2027'],
            ],
            'footer': 'Tổng cộng: 26 thiết bị. Người điều phối cần xác nhận lịch trước khi thực hiện.',
        },
        {
            'case_id': 'en_table', 'language': 'en', 'pair_id': 'table', 'layout': 'table',
            'title': 'EQUIPMENT INSPECTION SCHEDULE',
            'paragraphs': ['Read the table one row at a time, left to right. The dates are scenario assumptions.'],
            'headers': ['Code', 'Equipment', 'Quantity', 'Inspection date'],
            'rows': [
                ['A-01', 'Oscilloscope', '12', '18/01/2027'],
                ['A-02', 'Power supply', '8', '25/01/2027'],
                ['A-03', 'Signal generator', '6', '15/02/2027'],
            ],
            'footer': 'Total: 26 instruments. The coordinator must confirm the schedule before implementation.',
        },
    ]


def reference_text(spec):
    sections = [spec['title'], *spec['paragraphs']]
    if spec['layout'] == 'table':
        sections += [' '.join(spec['headers'])]
        sections += [' '.join(row) for row in spec['rows']]
        sections += [spec['footer']]
    return '\n'.join(sections) + '\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_references(directory, specs):
    """Commit known transcripts before any OCR call; preserve literal UTF-8."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for number, spec in enumerate(specs, 1):
        raw = reference_text(spec).encode('utf-8')
        path = directory / (spec['case_id'] + '.txt')
        with path.open('xb') as stream:
            stream.write(raw)
        entries.append({'page': number, **spec, 'transcript': path.name, 'transcript_sha256': sha(raw),
                        'reference_origin': REFERENCE_ORIGIN, 'human_verified': False})
    write_json_new(directory / 'references.json', {'fixture_version': FIXTURE_VERSION, 'pages': entries})
    return entries


def normalize(text, *, remove_diacritics=False):
    text = unicodedata.normalize('NFC', text).casefold()
    if remove_diacritics:
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn').replace('đ', 'd')
        text = unicodedata.normalize('NFC', text)
    return re.sub(r'\s+', ' ', text).strip()


def levenshtein(reference, prediction):
    """Unit-cost insertion/deletion/substitution distance with linear memory."""
    if len(prediction) > len(reference):
        reference, prediction = prediction, reference
    row = list(range(len(prediction) + 1))
    for index, left in enumerate(reference, 1):
        next_row = [index]
        for column, right in enumerate(prediction, 1):
            next_row.append(min(next_row[-1] + 1, row[column] + 1,
                                row[column - 1] + (left != right)))
        row = next_row
    return row[-1]


def error_rates(reference, prediction, *, remove_diacritics=False):
    reference = normalize(reference, remove_diacritics=remove_diacritics)
    prediction = normalize(prediction, remove_diacritics=remove_diacritics)
    reference_words, prediction_words = reference.split(), prediction.split()
    character_edits = levenshtein(reference, prediction)
    word_edits = levenshtein(reference_words, prediction_words)
    return {
        'reference_characters': len(reference), 'prediction_characters': len(prediction),
        'character_edits': character_edits,
        'cer': character_edits / len(reference) if reference else None,
        'reference_words': len(reference_words), 'prediction_words': len(prediction_words),
        'word_edits': word_edits,
        'wer': word_edits / len(reference_words) if reference_words else None,
        'reference_normalized_sha256': sha(reference.encode('utf-8')),
        'prediction_normalized_sha256': sha(prediction.encode('utf-8')),
    }


def aggregate(pages, normalization):
    metrics = [page['metrics'][normalization] for page in pages]
    values = {key: sum(item[key] for item in metrics)
              for key in ['reference_characters', 'character_edits', 'reference_words', 'word_edits']}
    values.update({
        'pages': len(pages),
        'micro_cer': values['character_edits'] / values['reference_characters'] if values['reference_characters'] else None,
        'micro_wer': values['word_edits'] / values['reference_words'] if values['reference_words'] else None,
        'macro_cer': sum(item['cer'] for item in metrics) / len(metrics) if metrics else None,
        'macro_wer': sum(item['wer'] for item in metrics) / len(metrics) if metrics else None,
    })
    return values


def _wrap(text, draw, font, width):
    lines, current = [], ''
    for word in text.split():
        candidate = current + ' ' + word if current else word
        if draw.textbbox((0, 0), candidate, font=font)[2] > width:
            if not current:
                raise ValueError('Fixture contains a word wider than its page')
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_fixtures(output, specs, font_path):
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader

    output, font_path = Path(output), Path(font_path)
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    heading = ImageFont.truetype(str(font_path), FONT_SIZE + 4)
    images, entries = [], []
    try:
        for spec in specs:
            image = Image.new('RGB', PAGE_SIZE, 'white')
            images.append(image)
            draw = ImageDraw.Draw(image)
            y, margin, width = 100, 100, PAGE_SIZE[0] - 200
            for line in _wrap(spec['title'], draw, heading, width):
                draw.text((margin, y), line, font=heading, fill='black')
                y += 62
            y += 45
            for paragraph in spec['paragraphs']:
                for line in _wrap(paragraph, draw, font, width):
                    draw.text((margin, y), line, font=font, fill='black')
                    y += 58
                y += 26
            if spec['layout'] == 'table':
                y += 25
                boundaries = [100, 260, 850, 1060, 1500]
                for row in [spec['headers'], *spec['rows']]:
                    for left, right, cell in zip(boundaries, boundaries[1:], row):
                        if draw.textbbox((0, 0), cell, font=font)[2] > right - left - 30:
                            raise ValueError('Fixture table cell is too wide')
                        draw.rectangle((left, y, right, y + 100), outline='#666666', width=2)
                        draw.text((left + 15, y + 28), cell, font=font, fill='black')
                    y += 100
                y += 50
                for line in _wrap(spec['footer'], draw, font, width):
                    draw.text((margin, y), line, font=font, fill='black')
                    y += 58
            if y > PAGE_SIZE[1] - 100:
                raise ValueError('Fixture text exceeds its image bounds')
            path = output / (spec['case_id'] + '.png')
            image.save(path, format='PNG')
            entries.append({'case_id': spec['case_id'], 'image': path.name,
                            'image_sha256': sha(path.read_bytes()), 'content_bottom_pixel': y})
        pdf = output / 'image_only_fixture.pdf'
        images[0].save(pdf, format='PDF', resolution=RASTER_DPI, save_all=True, append_images=images[1:])
    finally:
        for image in images:
            image.close()
    reader = PdfReader(str(pdf))
    try:
        if len(reader.pages) != len(specs) or any((page.extract_text() or '').strip() for page in reader.pages):
            raise ValueError('OCR diagnostic requires image-only pages without a selectable transcript')
    finally:
        reader.close()
    return {'pdf': pdf.name, 'pdf_sha256': sha(pdf.read_bytes()), 'page_size_pixels': list(PAGE_SIZE),
            'raster_dpi': RASTER_DPI, 'font_size_pixels': FONT_SIZE, 'font_path': str(font_path.resolve()),
            'font_sha256': sha(font_path.read_bytes()), 'all_pages_image_only': True, 'pages': entries}


def run(output, *, font_path=None, dpi=180):
    from psai_ocr import extract_pdf_with_ocr
    from psai_ocr.windows import probe_windows_ocr, WindowsPDFRenderer, WindowsOCREngine

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new OCR diagnostic directory; earlier evidence is preserved')
    font_path = Path(font_path or 'C:/Windows/Fonts/arial.ttf')
    if not font_path.is_file():
        raise ValueError('An existing Unicode font is required; pass --font without installing a model or language pack')
    capability = probe_windows_ocr()
    languages = capability.get('languages', [])
    if isinstance(languages, str):
        languages = [languages]
    if 'en-US' not in languages:
        raise ValueError('This diagnostic requires the already installed en-US OCR capability')
    output.mkdir(parents=True)
    specs = fixture_specs()
    references = write_references(output / 'references', specs)
    write_json_new(output / 'fixture_specs.json', {'version': FIXTURE_VERSION, 'reference_origin': REFERENCE_ORIGIN,
                                                 'human_verified': False, 'pages': specs})
    rendering = render_fixtures(output, specs, font_path)
    write_json_new(output / 'fixture_manifest.json', rendering)
    pdf = output / rendering['pdf']
    unsupported = {'requested_language': 'vi', 'status': 'unexpectedly_accepted'}
    try:
        extract_pdf_with_ocr(pdf, mode='always', backend='windows', language='vi')
    except ValueError as error:
        unsupported.update(status='unsupported_by_current_release', reason=str(error))
    if unsupported['status'] != 'unsupported_by_current_release':
        raise ValueError('Current OCR Vietnamese-language restriction changed; audit before running this protocol')
    write_json_new(output / 'vietnamese_language_request.json', unsupported)
    print('Running installed Windows.Media.Ocr en-US on six synthetic scan pages.', flush=True)
    extraction = extract_pdf_with_ocr(
        pdf, mode='always', backend='windows', language='en', dpi=dpi, cache_dir=output / 'rendered_pages',
        renderer_factory=lambda path: WindowsPDFRenderer(path, language='en-US'),
        engine_factory=WindowsOCREngine,
    )
    if len(extraction['pages']) != len(specs):
        raise ValueError('OCR returned a different number of pages')
    pages = []
    for reference, page in zip(references, extraction['pages']):
        engine = page.get('ocr_engine', {})
        if page.get('method') != 'local_ocr' or engine.get('name') != 'Windows.Media.Ocr' or engine.get('language') != 'en-US':
            raise ValueError('Every scored page must be recognized by the real installed en-US Windows engine')
        raw = (output / 'references' / reference['transcript']).read_bytes()
        if sha(raw) != reference['transcript_sha256']:
            raise ValueError('Reference transcript changed after it was written before OCR')
        prediction = extraction['text'][page['char_start']:page['char_end']]
        prediction_path = output / (reference['case_id'] + '.recognized.txt')
        with prediction_path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(prediction)
        pages.append({
            'case_id': reference['case_id'], 'page': page['page'], 'language': reference['language'],
            'pair_id': reference['pair_id'], 'layout': reference['layout'], 'engine': engine,
            'reference_file': 'references/' + reference['transcript'],
            'reference_sha256': reference['transcript_sha256'],
            'prediction_file': prediction_path.name, 'prediction_sha256': sha(prediction.encode('utf-8')),
            'metrics': {name: error_rates(raw.decode('utf-8'), prediction, remove_diacritics=remove)
                        for name, remove in [('diacritics_preserved', False), ('diacritics_removed', True)]},
        })
    write_json_new(output / 'extraction.json', extraction)
    source_files = ['tools/evaluate_ocr_diagnostic.py', 'psai_ocr/extract.py',
                    'psai_ocr/windows.py', 'psai_ocr/windows_native.ps1']
    dependencies = {}
    for package in ['Pillow', 'pypdf']:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    result = {
        'status': 'diagnostic_completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'fixture_version': FIXTURE_VERSION, 'reference_origin': REFERENCE_ORIGIN,
        'scope': 'Synthetic language-limit diagnostic on three paired Vietnamese/English image-only pages per language',
        'human_verified_references': False, 'independent_final_test': False,
        'vietnamese_support_claimed': False, 'vietnamese_language_request': unsupported,
        'engine_language': 'en-US', 'ocr_api_language_argument': 'en',
        'installed_capability': capability, 'real_local_ocr_executed': True,
        'network_used': False, 'models_downloaded': False, 'dependencies_installed': False,
        'system_language_packs_changed': False, 'gpu_used': False,
        'runtime': {'python': sys.version, 'executable': sys.executable,
                    'platform': platform.platform(), 'packages': dependencies},
        'protocol_source_sha256': {name: sha((ROOT / name).read_bytes()) for name in source_files},
        'fixtures': rendering, 'ocr_requested_dpi': dpi, 'ocr_elapsed_seconds': extraction['elapsed_seconds'],
        'normalization': {
            'diacritics_preserved': 'Unicode NFC, casefold and collapse whitespace; punctuation retained; WER splits on whitespace',
            'diacritics_removed': 'Same, plus NFD combining-mark removal and mapping Vietnamese letter đ to d',
            'undefined_policy': 'CER/WER are null for an empty reference denominator; rates are not clipped to 1',
            'reading_order': 'Whole-page edit distance includes ordering errors; tables reference row-major left-to-right order',
        },
        'pages': pages,
        'aggregate_by_language': {language: {
            mode: aggregate([page for page in pages if page['language'] == language], mode)
            for mode in ['diacritics_preserved', 'diacritics_removed']}
            for language in ['vi', 'en']},
        'limits': [
            'AI-authored clear synthetic fixtures are not representative scanned documents or human expert references.',
            'English controls share layout and topic; translations have different word/character distributions.',
            'Removing diacritics is a separate diagnostic and does not establish correct Vietnamese transcription.',
            'The installed English OCR engine is being tested out of language on Vietnamese, without adding Vietnamese support.',
            'Windows OCR supplies no confidence or model weight hashes through this API.',
            'Tables are scored for text and reading order, not reconstructed cell structure.',
        ],
    }
    write_json_new(output / 'diagnostic.json', result)
    lines = [
        '# Chẩn đoán OCR tiếng Việt bằng engine Windows en-US', '',
        'Sáu trang scan tự tạo bằng AI, gồm ba cặp tiếng Việt/tiếng Anh có cùng loại bố cục.',
        'Transcript gốc được cố định trước khi vẽ ảnh và trước OCR; không phải nhãn người hoặc final test.', '',
        '| Ngôn ngữ | Giữ dấu: CER | Giữ dấu: WER | Bỏ dấu: CER | Bỏ dấu: WER |',
        '|---|---:|---:|---:|---:|',
    ]
    for language in ['vi', 'en']:
        values = result['aggregate_by_language'][language]
        rates = [values['diacritics_preserved']['micro_cer'], values['diacritics_preserved']['micro_wer'],
                 values['diacritics_removed']['micro_cer'], values['diacritics_removed']['micro_wer']]
        lines.append('| ' + language + ' | ' + ' | '.join(f'{value:.2%}' for value in rates) + ' |')
    lines += ['', 'CER/WER là tổng edit distance chia tổng số ký tự/từ chuẩn sau chuẩn hóa (micro).',
              'Chuẩn hóa NFC, chữ thường và khoảng trắng; giữ dấu câu. Bảng đọc từng hàng từ trái sang phải.',
              'CER/WER gồm cả lỗi nhận dạng và thứ tự đọc; bỏ dấu chỉ là phép chẩn đoán riêng.', '',
              'Lệnh gọi `language=vi` bị từ chối đúng giới hạn hiện tại. Chẩn đoán này gọi engine en-US thật với tham số API `en`.',
              'Kết quả không công bố hỗ trợ OCR tiếng Việt, không đo chất lượng trên tài liệu ngoài thực tế.', '',
              'Xem [diagnostic.json](diagnostic.json), [extraction.json](extraction.json),',
              '[ảnh PDF chỉ chứa raster](image_only_fixture.pdf) và [transcript gốc](references/references.json).', '']
    with (output / 'README.md').open('x', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines))
    return {'output': str(output), 'status': result['status'], 'aggregate_by_language': result['aggregate_by_language']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args(argv)
    try:
        result = run(args.output, font_path=args.font, dpi=args.dpi)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
