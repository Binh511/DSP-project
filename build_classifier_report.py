"""Embed local evidence into the classifier report and export a standalone DOCX."""
from pathlib import Path
import ast
import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports' / 'BAO_CAO_TRAIN_CLASSIFIER.md'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows])


def code_function(filename, function):
    text = (ROOT / filename).read_text(encoding='utf-8-sig')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == function)
    return '\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno])


def main():
    summary = json.loads((ROOT / 'reports/model_evidence/summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'runs/fair_fold0/frozen_split.json').read_text(encoding='utf-8'))
    exp = json.loads((ROOT / 'models/validation_selected_f0/experiment.json').read_text(encoding='utf-8'))
    state = json.loads((ROOT / 'models/validation_selected_f0/training_state.json').read_text(encoding='utf-8'))
    from fair_fold0 import partitions
    from train_human_cv import COLS, LABELS, text_key
    import pandas as pd
    orig, train, test, val, _, ht, hv, _ = partitions()
    held = set(pd.concat([test, val, hv]).text.map(text_key))
    combined = pd.concat([train[COLS], ht[COLS]], ignore_index=True)
    forbidden = combined.text.map(text_key).isin(held)
    for c in ('prev_text', 'next_text'):
        forbidden |= combined[c].fillna('').map(text_key).isin(held)
    new_train = combined.loc[~forbidden]
    assert len(new_train) == exp['configuration']['train_rows']
    appendix = ['## Phụ lục A. Kiểm kê model đã lưu',
                'Bảng dưới là bản kiểm kê trực tiếp các thư mục model hiện có. “Chưa có” nghĩa là chưa tìm thấy chỉ số riêng đã lưu trong lần tổng hợp, không phải mô hình đạt điểm 0. Cột trọng số chỉ xác nhận file hiện diện; không chứng minh lịch sử huấn luyện.',
                table(['Model', 'Có trọng số', 'Ngữ cảnh', 'Độ dài', 'OTHER bias', 'Có chỉ số riêng đã lưu'],
                      [[v['model'], 'Có' if v['has_weights'] else 'Không', v['ctx_version'], v['max_len'], v['other_bias'],
                        'Có' if v['saved_metrics'] else 'Chưa có'] for v in summary['inventory']]),
                '## Phụ lục B. Phân bố nhãn và tập chia',
                'Số mẫu từng lớp được tính lại từ dữ liệu hiện tại bằng đúng các hàm chia và lọc dùng trong dự án. Train mới là sau bước lọc bổ sung cả ngữ cảnh.',
                table(['Nhãn', 'Dữ liệu gốc', 'Train baseline', 'Thủ công train', 'Train mới', 'Validation', 'Test', 'Human holdout'],
                      [[lab] + [int(d.label.eq(lab).sum()) for d in [orig, train, ht, new_train, val, test, hv]] for lab in LABELS]),
                '### Mã tài liệu của các tập giữ lại',
                'Test: ' + ', '.join(manifest['test_document_ids']),
                'Validation: ' + ', '.join(manifest['validation_document_ids']),
                'Human holdout: ' + ', '.join(manifest['human_holdout_document_ids']),
                '## Phụ lục C. Ma trận nhầm lẫn từ dự đoán đã lưu',
                'Hàng là nhãn thực tế; cột là nhãn dự đoán. Ô trên đường chéo là số mẫu đúng. Các bảng này được tính từ dự đoán đã lưu của cặp fair_fold0. Không có dự đoán từng mẫu được lưu cho lượt validation_selected_f0 trong nguồn báo cáo này, nên không dựng ma trận cho lượt đó từ các chỉ số tổng hợp.']
    for model in ['fair_fold0_baseline', 'fair_fold0_human_once']:
        for dataset in ['Fold0_test', 'Human_holdout']:
            file = ROOT / f'reports/model_evidence/confusion_{model}_{dataset}.csv'
            with file.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.reader(f))
            appendix += [f'### {model} — {dataset}', table(['Thực tế / Dự đoán'] + rows[0][1:], rows[1:])]
    appendix += ['## Phụ lục D. Trạng thái huấn luyện và kết quả lịch sử',
                table(['Trường', 'Giá trị'], [[k, state.get(k)] for k in ['epoch', 'global_step', 'best_global_step', 'best_metric', 'num_train_epochs']]),
                'Các dòng sau là kết quả lịch sử được giữ lại để minh bạch quá trình thử nghiệm. Điểm của checkpoint cũ chưa xác minh đầy đủ tập chia; không gộp với bảng kết quả chính để xếp hạng model.',
                table(['Model / lượt chạy', 'Tập', 'Số mẫu', 'Accuracy', 'Macro-F1', 'Giới hạn'],
                      [[s['model'], s['dataset'], s['n'] if s['n'] is not None else 'Không xác định',
                        f'{s["accuracy"]:.4f}' if s['accuracy'] is not None else 'Không lưu', f'{s["macro_f1"]:.4f}',
                        'Checkpoint cũ chưa xác minh split' if 'Historical fold' in s['dataset'] else
                        'Mẫu rà soát nhỏ' if 'holdout' in s['dataset'] else 'Validation chọn checkpoint; chưa xác minh model export']
                       for s in summary['metrics'] if 'Historical' in s['dataset'] or s['model'].startswith('psai_cls')]),
                '## Phụ lục E. Cấu hình và dấu vân tay dữ liệu',
                'Dấu SHA256 dùng nhận diện chính xác phiên bản dữ liệu. Đây không phải chữ ký số chứng minh thời điểm hoặc tác giả huấn luyện.',
                table(['Trường', 'Giá trị'], list(exp['configuration'].items())),
                '## Phụ lục F. Trích đoạn mã triển khai quan trọng',
                'Các đoạn mã dưới đây được chèn trực tiếp từ mã nguồn hiện tại để người đọc kiểm tra cách chuẩn bị đầu vào, chia tập và tính loss. Chúng không đại diện cho toàn bộ mã nguồn và không được xem là bằng chứng rằng mọi model lịch sử dùng cùng một cấu hình.']
    for filename, function, title in [
        ('train_human_cv.py', 'make_ds', 'Chuẩn bị đầu vào câu và ngữ cảnh'),
        ('fair_fold0.py', 'partitions', 'Chia dữ liệu và loại câu trùng'),
        ('train_human_cv.py', 'train_model', 'Huấn luyện, loss và lựa chọn checkpoint')]:
        appendix += [f'### {title}', f'Nguồn: {filename}, hàm {function}.', '```python\n' + code_function(filename, function) + '\n```']
    appendix += ['## Phụ lục G. Phạm vi bằng chứng',
                'Báo cáo này tự chứa bảng số liệu, kiểm kê, cấu hình và đoạn mã cần đọc; không yêu cầu mở đường dẫn bên ngoài. Trọng số đầy đủ và toàn bộ dataset không nhúng trong tài liệu. Các chỉ số fair_fold0 được tái tính từ dự đoán đã lưu; kết quả lượt mới dựa trên experiment.json và training_state.json. Việc tổng hợp không thực hiện lại inference trên toàn bộ model.']
    original = REPORT.read_text(encoding='utf-8')
    main_text = original.split('## Phụ lục')[0]
    text = main_text.rstrip() + '\n\n' + '\n\n'.join(appendix) + '\n'
    REPORT.write_text(text, encoding='utf-8')
    export_docx(text, REPORT.with_suffix('.docx'))
    print('Exported self-contained report:', REPORT.with_suffix('.docx'))


def export_docx(text, out):
    ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

    def para(s, style=None, code=False):
        prop = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        if code:
            run_prop = '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="16"/></w:rPr>'
        else:
            s = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', s).replace('**', '').replace('`', '')
            run_prop = ''
        return '<w:p>' + prop + '<w:r>' + run_prop + '<w:t xml:space="preserve">' + escape(s) + '</w:t></w:r></w:p>'

    body, lines, i, code = [], text.splitlines(), 0, False
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if line.startswith('```'):
            code = not code
            i += 1
            continue
        if code:
            body.append(para(raw, code=True))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append(cells)
                i += 1
            borders = ''.join(f'<w:{x} w:val="single" w:sz="4" w:color="AABBCD"/>' for x in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'])
            tbl = '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>' + borders + '</w:tblBorders></w:tblPr>'
            for n, row in enumerate(rows):
                tbl += '<w:tr>' + ('<w:trPr><w:tblHeader/></w:trPr>' if n == 0 else '')
                for cell in row:
                    tbl += '<w:tc><w:tcPr>' + ('<w:shd w:fill="E7EFF8"/>' if n == 0 else '') + '</w:tcPr>' + para(cell) + '</w:tc>'
                tbl += '</w:tr>'
            body += [tbl + '</w:tbl>', para('')]
            continue
        if line:
            heading = re.match(r'^(#{1,3})\s+(.*)', line)
            body.append(para(heading[2], 'Title' if len(heading[1]) == 1 else 'Heading' + str(len(heading[1]) - 1)) if heading else para(line))
        i += 1
    doc = '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="' + ns + '"><w:body>' + ''.join(body) + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1417"/></w:sectPr></w:body></w:document>'
    styles = '<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="' + ns + '"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:lang w:val="vi-VN"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
    for name, size in [('Title', 32), ('Heading1', 29), ('Heading2', 26)]:
        styles += f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    styles += '</w:styles>'
    package = {
        '[Content_Types].xml': '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>',
        '_rels/.rels': '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        'word/document.xml': doc, 'word/styles.xml': styles,
        'word/_rels/document.xml.rels': '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
    }
    for value in package.values():
        ET.fromstring(value)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in package.items():
            z.writestr(name, value)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert 'hyperlink' not in z.read('word/document.xml').decode()


if __name__ == '__main__':
    main()
