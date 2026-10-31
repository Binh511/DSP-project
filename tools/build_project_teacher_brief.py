"""Build a portable teacher brief from existing PSAI code and run receipts.

This script reads evidence only. It does not load models, train, score test data,
or change a release manifest. HTML embeds its diagrams, source excerpts and data.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
import hashlib
from html import escape
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/teacher_brief'
TITLE = 'PSAI — Pipeline chi tiết, đầu ra và tình trạng hiện tại'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def rel(path):
    return Path(os.path.relpath(Path(path), OUT)).as_posix()


def runtime_diagram():
    """Draw the actual data flow; rules are exported after planning, not merged into its prompt."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1380 1710" role="img" aria-labelledby="title desc">',
             '<title id="title">Pipeline runtime thực tế của PSAI</title>',
             '<desc id="desc">PDF hoặc text thành snapshot nguồn. Classifier tạo catalog công việc; RAG cung cấp tham khảo ngoài và context báo cáo. Qwen chọn ID công việc, sau kiểm định mới bổ sung rule facts và xuất report, JSON, workspace. Rule facts chưa đi vào Qwen.</desc>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10Z" fill="#52718a"/></marker></defs>',
             '<rect width="1380" height="1710" fill="#f5f8fc"/>',
             '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#19324b}.title{font-size:28px;font-weight:700}.sub{font-size:19px}.box-title{font-size:22px;font-weight:700}.body{font-size:19px}.note{font-size:18px;fill:#526a80}.edge{fill:none;stroke:#52718a;stroke-width:3;marker-end:url(#arrow)}.dashed{stroke-dasharray:9 7}</style>',
             '<text x="50" y="48" class="title">PSAI · luồng đang chạy trong code</text>',
             '<text x="50" y="80" class="sub">Luồng dữ liệu theo chức năng; ba worker GPU chạy tuần tự: extract → retrieve → plan.</text>']

    def box(x, y, w, h, title, lines, fill='#ffffff', stroke='#c3d4e3'):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x+22}" y="{y+34}" class="box-title">{escape(title)}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x+22}" y="{y+66+i*27}" class="body">{escape(line)}</text>')

    def edge(points, dashed=False):
        parts.append(f'<path d="{points}" class="edge{" dashed" if dashed else ""}"/>')

    box(385, 110, 610, 102, '1 · Đầu vào', ['Một PDF / Markdown / TXT + tham khảo cục bộ tùy chọn'], '#e9f2ff')
    edge('M690 212 V240')
    box(385, 240, 610, 122, '2 · Kiểm tra và đọc / OCR', ['pypdf cho trang có chữ; Windows OCR cho trang ảnh', 'Lưu trang, ảnh, vị trí chữ và hash file gốc'])
    edge('M690 362 V390')
    box(385, 390, 610, 122, '3 · Snapshot nguồn dùng chung', ['Text chính xác + SHA-256; giữ quote và char spans', 'Heading / section / câu trước–sau / vị trí tương đối'])
    edge('M690 512 V542 H335 V570')
    edge('M690 542 H1035 V570')
    box(45, 570, 590, 174, '4A · Classifier + cue extraction', ['mDeBERTa ensemble: final_s11 + s22 + s33', 'Trung bình xác suất 5 lớp → lọc ngưỡng và nhiễu', 'Mục tiêu / yêu cầu / hoạt động / kết quả mong đợi', '+ deliverables, công nghệ, nguồn lực theo cue'], '#e7f5ee', '#94c6b0')
    box(745, 570, 590, 174, '4B · Retrieval', ['Chia đoạn + metadata + tham khảo có nguồn riêng', 'BM25 + BGE-M3 gốc → hợp nhất RRF → top-5', 'Đoạn trong dự án: context để đối chiếu báo cáo', 'Đoạn role=resource: tham khảo vào catalog Qwen'], '#e8f2ff', '#9dbfe1')
    edge('M335 744 V805')
    edge('M1035 744 V775 H690 V805')
    box(280, 805, 820, 146, '5 · Catalog bằng chứng và ứng viên công việc', ['Task từ activities / deliverables, xác minh nguồn và loại trùng', 'Nguồn lực từ cue extraction; tham khảo ngoài từ retrieval', 'Tối đa 12 task + 12 resource + 6 tham khảo; ghi số mục bị bỏ'])
    edge('M690 951 V979')
    box(280, 979, 820, 120, '6 · Qwen2.5-1.5B-Instruct · suy luận cục bộ', ['Chọn / sắp xếp task ID có sẵn, khôi phục mô tả từ nguồn', 'Cho phép gợi ý nhóm cha–con, checklist và dependency để người duyệt'], '#eeeaff', '#b6a7df')
    edge('M690 1099 V1127')
    box(280, 1127, 820, 120, '7 · Kiểm định và tái kiểm định trước export', ['Schema JSON · ID · quote / hash / span · chu trình phụ thuộc', 'Giữ prompt, raw response, catalog và provenance; lỗi giữ chẩn đoán'])
    edge('M690 1247 V1275')
    box(280, 1275, 820, 125, '8 · Rule facts và cảnh báo tại bước xuất', ['Ngày / mã / mốc / thời lượng / ngân sách / công nghệ / nhân sự', 'Các object rule facts hiện chưa được đưa vào prompt Qwen'], '#fff5df', '#dec48e')
    edge('M690 1400 V1430')
    box(45, 1430, 410, 116, '9A · Báo cáo HTML', ['Nội dung chính + task + cảnh báo', 'Trích dẫn và ảnh OCR có nguồn'], '#e9f2ff')
    box(485, 1430, 410, 116, '9B · JSON và log', ['analysis / predictions / plan', 'Snapshot / cấu hình / hash / jobs'], '#e9f2ff')
    box(925, 1430, 410, 116, '9C · Workspace offline', ['Trạng thái, checklist, ghi chú', 'Người duyệt → xuất/nhập tiến độ'], '#e9f2ff')
    edge('M690 1415 H250 V1430')
    edge('M690 1415 H1130 V1430')
    parts += ['<text x="50" y="1590" class="note">Lưu ý: top-k đoạn thuộc tài liệu dự án không trực tiếp tạo task mới trong prompt hiện tại.</text>',
              '<text x="50" y="1622" class="note">Nhánh rule facts đi vào báo cáo sau LLM; không vẽ như đã hợp nhất đầy đủ ba nhánh trước LLM.</text>',
              '<text x="50" y="1654" class="note">Kiểm tra nguồn đúng ≠ nhãn đúng / kế hoạch đủ. Sơ đồ này mô tả implementation, không phải cam kết chất lượng.</text>',
              '</svg>']
    return '\n'.join(parts)


class Brief:
    def __init__(self):
        self.html, self.md, self.text, self.nav = [], [], [], []

    def heading(self, title, identity, level=2):
        self.html.append(f'<h{level} id="{identity}">{escape(title)}</h{level}>')
        self.md += ['#' * level + ' ' + title, '']
        self.text += [title, '']
        if level == 2:
            self.nav.append((identity, title))

    def p(self, value, notice=False):
        self.html.append(f'<p{" class=notice" if notice else ""}>{escape(value)}</p>')
        self.md += [value, '']
        self.text += [value, '']

    def bullets(self, items):
        self.html.append('<ul>' + ''.join('<li>' + escape(s) + '</li>' for s in items) + '</ul>')
        self.md += ['- ' + s for s in items] + ['']
        self.text += ['• ' + s for s in items] + ['']

    def table(self, headers, rows):
        self.html.append('<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
                         + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>')
        self.md += ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
        self.md += ['| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', '<br>') for v in row) + ' |' for row in rows] + ['']
        self.text += [' / '.join(headers)] + [' / '.join(map(str, row)) for row in rows] + ['']

    def code(self, value, language='text'):
        self.html.append('<pre><code>' + escape(value) + '</code></pre>')
        self.md += ['```' + language, value, '```', '']
        self.text += [value, '']

    def details(self, title, value):
        self.html.append('<details><summary>' + escape(title) + '</summary><pre><code>' + escape(value) + '</code></pre></details>')
        self.md += ['### ' + title, '', '```text', value, '```', '']
        self.text += [title, value, '']

    def diagram(self, svg):
        self.html.append('<figure class="diagram">' + svg + '<figcaption>Sơ đồ đối chiếu trực tiếp với runtime. Không dùng dịch vụ render bên ngoài.</figcaption></figure>')
        self.md += ['![Pipeline thực tế](./PIPELINE_PSAI.svg)', '']
        self.text += ['Sơ đồ: PIPELINE_PSAI.svg; thứ tự và vai trò từng bước được trình bày ngay bên dưới.', '']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / 'reports/project_completion/latest_acceptance.json'
    receipt = load(receipt_path)
    run = ROOT / Path(receipt['report'])
    run = run.parent
    analysis = load(run / 'analysis.json')
    preds = load(run / 'classifier_predictions.json')
    plan = analysis['plan']
    fields = analysis['extraction']['fields']
    source = (run / 'source.txt').read_text(encoding='utf-8')
    release_path = ROOT / 'reports/project_completion/PROJECT_RELEASE_MANIFEST.json'
    release = load(release_path)
    classifier_path = ROOT / 'reports/classifier/release_source_clean_v2/FINAL_MODEL_MANIFEST.json'
    classifier = load(classifier_path)
    study_path = ROOT / 'reports/research_review/studies/classifier_source_clean_context_study_v1'
    study = load(study_path / 'summary.json')
    metrics = load(study_path / 'experiments/baseline_seed_2026/validation_metrics.json')
    rag_path = ROOT / 'reports/rag_studies/retrieval_genq_lora_v1'
    base, adapter = load(rag_path / 'base_metrics.json'), load(rag_path / 'adapter_metrics.json')
    paired_path = ROOT / 'psai_rag_quality/artifacts/retrieval_genq_lora_v1/paired_analysis.json'
    paired = load(paired_path)
    ci = paired['summary']['within_doc/dense']['mrr_delta_bootstrap']
    rag_training = load(ROOT / 'models/rag/retrieval_genq_lora_v1/training_receipt.json')
    date = datetime.now(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M UTC+7')
    accept_date = datetime.fromisoformat(receipt['created_utc']).astimezone(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M UTC+7')
    b = Brief()
    b.heading(TITLE, 'cover', 1)
    b.p(f'Đối chiếu code và bằng chứng lưu trên máy lúc {date}. Nghiệm thu đầy đủ gần nhất: {accept_date}.')
    b.p('Kết luận hiện tại: PSAI đã có luồng kỹ thuật chạy xuyên suốt bằng model thật trên PDF mẫu tự tạo. Đầu ra là bản đề xuất kế hoạch có dẫn chứng và workspace cục bộ. Chưa có đánh giá độc lập đủ để kết luận chất lượng đầu cuối trên tài liệu thực tế.', True)
    b.p('Bản này dành cho thầy nhận xét kiến trúc, phạm vi đã làm và ưu tiên hoàn thiện. Nội dung, sơ đồ, số liệu, ví dụ nguồn, ảnh OCR và phụ lục bằng chứng được nhúng vào HTML; thầy không cần project gốc để đọc. Không train thêm hoặc chấm final test khi lập báo cáo.')

    b.heading('1. Mục tiêu, đầu vào và sản phẩm cuối', 'purpose')
    b.p('PSAI hỗ trợ chuyển tài liệu đề xuất dự án thành nội dung có cấu trúc và danh sách công việc đề xuất. Người dùng có thể kiểm tra lại câu nguồn, xem cảnh báo nguồn lực và theo dõi tiến độ thực hiện. Đây là công cụ hỗ trợ người lập kế hoạch; việc chấp thuận kế hoạch vẫn thuộc người dùng.')
    b.table(['Đối tượng', 'Phạm vi hiện có'], [
        ('Đầu vào chính', 'Một PDF, Markdown hoặc TXT mỗi lượt; PDF có text hoặc trang scan.'),
        ('Đầu vào bổ sung', 'Thư mục tài liệu tham khảo Markdown/TXT cục bộ; query truy xuất tùy chọn.'),
        ('Đầu ra cho người dùng', 'Báo cáo HTML có nội dung chính, task và bằng chứng; workspace HTML quản lý trạng thái.'),
        ('Đầu ra cho kiểm chứng', 'JSON phân loại/truy xuất/kế hoạch/dữ kiện, snapshot nguồn, prompt, raw response, cấu hình, hash và log.'),
        ('Vận hành', 'Cục bộ, không gọi API trong inference; extract → retrieve → plan chạy tuần tự để giải phóng GPU.'),
        ('Phạm vi chưa có', 'Chưa có cộng tác nhiều người/cloud, lịch tối ưu tự động, cam kết kế hoạch đầy đủ hay độ chính xác 100%.')])

    b.heading('2. Pipeline thực tế của phiên bản đang chạy', 'pipeline')
    diagram = runtime_diagram()
    b.diagram(diagram)
    b.p('Sơ đồ phân nhánh thể hiện dòng dữ liệu; không có nghĩa tất cả model được nạp đồng thời. Điều phối hiện chạy ba worker tuần tự. Source snapshot và các kiểm tra hash/span là lớp kiểm chứng xuyên suốt các bước.')
    b.table(['Bước', 'Xử lý và thuật toán', 'Đầu ra / giới hạn'], [
        ('1 · Kiểm tra input', 'Kiểm tra file, hậu tố, tham số, thư mục output mới; tính SHA-256 file gốc.', 'Không ghi đè kết quả đã có. Giữ file gốc để đối chiếu.'),
        ('2 · PDF → text/OCR', 'pypdf đọc từng trang. Auto OCR trang có ít hơn 32 ký tự không phải khoảng trắng; backend Windows.Media.Ocr en-US đã chạy thật.', 'Text, metadata trang, ảnh scan, line/word boxes và hash. Bảng và thứ tự đọc chưa được xác nhận trên bộ thực tế.'),
        ('3 · Cấu trúc và context', 'Lưu text nguồn; tách câu, nhận diện Markdown heading, section, câu trước/sau, vị trí, bullet/cue.', 'Mỗi câu có char_start/end. Chưa phải mô hình phục hồi bố cục PDF; tiêu đề không nhận diện được có thể để trống.'),
        ('4A · Classifier', 'Ba mDeBERTa checkpoint, context version 3; lấy trung bình softmax năm nhãn; mỗi model tối đa 256 token.', 'OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT, OTHER; giữ đủ năm xác suất theo đúng thứ tự.'),
        ('4A · Nội dung chính', 'Giữ REQUIREMENT nếu score ≥ 0,60; OBJECTIVE/ACTIVITY/EXPECTED_RESULT ≥ 0,70; lọc nhiễu/reference. Cue bổ sung deliverables, technologies, resources.', 'Các ngưỡng là heuristics giữ từ trước, chưa được xác nhận là operating point tối ưu. OTHER không thành field nội dung chức năng.'),
        ('4B · Chunking và RAG', 'Chunk nguyên văn, heading prefix riêng. BGE-M3 normalized CLS 1.024 chiều + BM25; RRF k=60, hợp nhất nhiều query, top-5 duy nhất mặc định.', 'Hits có role document/resource, quote, nguồn, span, điểm và query_matches. Runtime tạo index tạm; API persistent index có riêng.'),
        ('5 · Catalog và loại trùng', 'Ứng viên task từ activities/deliverables; kiểm tra nguồn, chuẩn hóa khóa để loại trùng; giữ mục có điểm model ≥ 0,80 theo PLAN_MIN_SCORE khi có label_score.', 'Mặc định 12 task, 12 project resource, 6 external reference. Số mục ngoài catalog được ghi lại.'),
        ('6 · LLM kế hoạch', 'Qwen2.5-1.5B-Instruct pretrained chạy greedy cục bộ; chọn, sắp xếp task ID; mô tả được khôi phục từ bằng chứng.', 'Có schema cha/con, checklist trích nguồn, resource association và dependency candidate. Tất cả cần người duyệt.'),
        ('7 · Kiểm định', 'Schema/ID/quote/char span/hash; kiểm tra chu trình và tham chiếu; tái dựng catalog/prompt để kiểm tra lại output trước export.', 'Dữ liệu sai bị từ chối; giữ diagnostics. Đúng nguồn không tự chứng minh nhãn đúng hay task đầy đủ.'),
        ('8 · Rule facts khi export', 'Mẫu xác định ngày, ID, milestone, duration, budget, technology, personnel; chuẩn hóa có chính sách và giữ mơ hồ.', 'Tạo rule_facts và cảnh báo sau bước LLM. Những object này hiện chưa tham gia prompt Qwen.'),
        ('9 · Báo cáo và workspace', 'Xuất HTML/JSON/source snapshots/log; ảnh OCR nhúng; workspace cập nhật tiến độ và duyệt dependency.', 'Mỗi output là một bộ riêng gắn source/analysis hash; xuất JSON để lưu tiến độ chắc chắn.')])
    b.p('Điểm cần phân biệt với sơ đồ thiết kế ban đầu: hiện chưa có LayoutLMv2/BIO tagging trong luồng runtime này. Classifier là phân loại câu; structuring dựa trên heading và quy tắc. Các đoạn RAG role=document được gắn làm related_context trong báo cáo, không trực tiếp trở thành task mới trong prompt. Chỉ hits role=resource được đưa vào catalog tham khảo ngoài. Rule facts chi tiết được bổ sung sau LLM, chưa phải cơ chế hợp nhất ba nhánh đầy đủ trước sinh kế hoạch.', True)
    b.p('Về cleaning, luồng mới giữ snapshot text làm nguồn chuẩn để bảo toàn vị trí trích dẫn; việc bỏ code fence, dòng trích dẫn, mảnh nhiễu hoặc reference nằm trong các bước tách câu/lọc nội dung. Không xem các script preprocessing/cleaning cũ có trong repository là bằng chứng chúng đều được gọi trong runtime mới.')
    b.p('Planner mặc định có tổng context 4.096 token, giới hạn đầu vào 3.072 và đầu ra 1.024 token. Prompt quá dài bị từ chối; catalog vượt số lượng có thống kê mục bị bỏ, nên không được khẳng định đã xử lý đầy đủ mọi task trong tài liệu dài. Bốn query chủ đề trong lần nghiệm thu:')
    b.bullets(analysis['retrieval']['queries'])

    b.heading('3. Model nào đang chạy và model nào đã huấn luyện', 'models')
    b.table(['Thành phần', 'Đang dùng trong demo đầy đủ', 'Nghiên cứu / nguồn gốc'], [
        ('Phân loại câu', 'Ensemble final_s11 + final_s22 + final_s33.', 'Checkpoint nghiên cứu V2 riêng là baseline_seed_2026; chưa thay ensemble production.'),
        ('Embedding/truy xuất', 'BGE-M3 gốc + BM25, hợp nhất RRF.', 'Đã train LoRA retriever; bật riêng để thí nghiệm, chưa chứng minh cải thiện ổn định.'),
        ('LLM lập kế hoạch', 'Qwen2.5-1.5B-Instruct pretrained, suy luận tại máy.', 'Project tích hợp và ràng buộc đầu ra; không huấn luyện Qwen.'),
        ('OCR', 'pypdf + Windows.Media.Ocr en-US đã cài trên hệ điều hành.', 'Không train OCR; backend RapidOCR/PDFium có hỗ trợ tùy chọn nhưng chưa nghiệm thu trên máy này.')])

    b.heading('4. Pipeline nghiên cứu và kết quả Classifier', 'classifier')
    b.p('Dữ liệu câu + nhãn tham chiếu → rà soát trùng nguồn → giữ validation cố định, loại các câu train bị ảnh hưởng → hai cách context × ba seed → chọn checkpoint bằng validation theo cùng giao thức FP32 → lưu model/tokenizer/context/data hashes. Không dùng final test để chọn model. 57 đề xuất sửa nhãn AI là danh sách cần phân xử, không phải ground truth.')
    b.table(['Dữ liệu / lựa chọn', 'Kết quả hiện có'], [
        ('Train sau làm sạch nguồn', f"{classifier['dataset']['train_rows']:,} câu / {classifier['dataset']['train_documents']} ID tài liệu; đã loại 202 câu train bị ảnh hưởng."),
        ('Validation', f"{classifier['dataset']['validation_rows']:,} câu / {classifier['dataset']['validation_documents']} ID tài liệu."),
        ('Checkpoint V2', classifier['selected_run']),
        ('Validation Macro-F1 V2', f"{classifier['validation_macro_f1']:.6f}"),
        ('Validation accuracy V2', f"{classifier['validation_accuracy']:.6f}"),
        ('Điều kiện', 'Đây là validation đã dùng phát triển; nhãn chưa phân xử độc lập đầy đủ; không phải independent final-test score.')])
    b.table(['Seed', 'Context baseline cũ · Macro-F1', 'Context khôi phục · Macro-F1'], [
        (str(row['seed']), f"{row['baseline_f1']:.6f}", f"{row['context_f1']:.6f}")
        for row in sorted(study['paired_seed_deltas'], key=lambda x: x['seed'])])
    b.p('Mean ± độ lệch chuẩn mẫu của ba seed: baseline 0,615912 ± 0,014231; context khôi phục 0,388637 ± 0,213019. Context khôi phục chưa được chọn. Con số lịch sử 0,6544 thuộc lượt trước xử lý trùng nguồn; không gán điểm đó cho V2 hay ensemble đang demo.')
    b.table(['Lớp · V2 đã chọn', 'Precision', 'Recall', 'F1', 'Support'], [
        (label, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}", str(int(row['support'])))
        for label, row in metrics['per_class'].items()])
    b.p('EXPECTED_RESULT vẫn yếu nhất với F1 khoảng 0,4409. OTHER có 843/1.317 câu nên accuracy phải đọc cùng Macro-F1. Source-disjoint theo hash toàn văn không tự chứng minh đã loại hết near-duplicate hoặc mọi tiếp xúc dữ liệu trước đây. Chưa có phép so sánh đủ điều kiện để kết luận V2 tốt hơn ensemble production.')

    b.heading('5. Pipeline nghiên cứu và kết quả RAG', 'rag')
    b.p('254 Markdown → gom 223 nhóm nguồn để tách train/validation/test → chunk theo tokenizer → T5 sinh query → BM25 tìm negative → cross-encoder lọc cặp → 1.536 triplet → BGE-M3 + LoRA rank 8, CachedMNRL batch 16/mini-batch 1 → đánh giá base và adapter trên cùng 54 query validation tự sinh → phân tích ghép cặp theo nhóm nguồn.')
    b.p(f"Lượt LoRA seed 42 đã hoàn thành {rag_training['global_step']}/{rag_training['planned_steps']} bước, một epoch; {rag_training['parameters']['trainable']:,} tham số được huấn luyện. Không có lượt train mới trong lần lập báo cáo này.")
    rag_rows = []
    for scope, method, variant, display in [
        ('within_doc', 'bm25', base, 'BM25'), ('within_doc', 'dense', base, 'BGE-M3 gốc'),
        ('within_doc', 'dense', adapter, 'BGE-M3 + LoRA'),
        ('within_doc', 'bm25_dense_rrf', base, 'Hybrid/RRF gốc'),
        ('within_doc', 'bm25_dense_rrf', adapter, 'Hybrid/RRF + LoRA')]:
        values = variant['metrics'][scope][method]
        rag_rows.append((display, f"{values['MRR@10']:.6f}", f"{values['Recall@5']:.6f}", f"{values['Recall@10']:.6f}"))
    b.table(['Validation trong tài liệu', 'MRR@10', 'Recall@5', 'Recall@10'], rag_rows)
    b.p(f"Dense MRR thay đổi {ci['estimate']:+.6f}; CI95% [{ci['ci95_low']:+.6f}; {ci['ci95_high']:+.6f}], chứa 0. Chưa đủ bằng chứng LoRA cải thiện ổn định. Global validation dense cũng không tăng: {base['metrics']['global_validation']['dense']['MRR@10']:.6f} → {adapter['metrics']['global_validation']['dense']['MRR@10']:.6f}. Default giữ BGE-M3 gốc; hybrid là cấu hình chức năng, chưa thể tuyên bố tốt hơn dense trên mọi truy vấn.")
    b.p('Query và positive trong study là giám sát tự động. Chưa có bộ relevance do người chấm hoàn chỉnh; chỉ số không chứng minh chất lượng lập kế hoạch hoặc OCR. Giao thức chunk của study và runtime có khác biệt, vì vậy không chuyển nguyên các điểm study thành điểm chất lượng runtime. Tài liệu giữ lại của retriever cũng chưa mặc nhiên độc lập với ensemble classifier cũ.')

    b.heading('6. Đầu ra cuối cùng: ví dụ đã chạy thật', 'actual-output')
    b.p('Ví dụ dưới lấy nguyên từ lượt nghiệm thu PDF tự tạo, không phải kế hoạch minh họa viết lại và không phải final test. Input có hai trang text và một trang scan tiếng Anh; có kèm tài liệu hướng dẫn collector làm resource tham khảo.')
    unique_ids = {item['evidence_id'] for entries in fields.values() for item in entries}
    b.table(['Nội dung kiểm tra', 'Giá trị thực tế'], [
        ('Trang PDF / trang OCR', f"{len(analysis['source']['pages'])} / {receipt['ocr_pages']}"),
        ('Câu/đơn vị được classifier dự đoán', len(preds['items'])),
        ('Mục bằng chứng theo field', f"{sum(map(len, fields.values()))} mục; {len(unique_ids)} evidence_id khác nhau."),
        ('Mục mang nhãn model', f"{receipt['source_checks']['model_label_evidence_count']} mục theo field; có thể lặp cùng câu giữa activity/deliverable."),
        ('Đoạn truy xuất', f"{len(analysis['retrieval']['hits'])}: 3 từ tài liệu dự án, 2 từ resource."),
        ('Task LLM chọn', len(plan['steps'])),
        ('Task con / checklist / resource association', f"{sum(x['parent_task_id'] is not None for x in plan['steps'])} / {sum(len(x['checklist']) for x in plan['steps'])} / {sum(len(x['required_resources']) for x in plan['steps'])}"),
        ('Dependency candidates / external references được chọn', f"{len(plan['dependency_candidates'])} / {len(plan['resource_references'])}"),
        ('Trạng thái kế hoạch', plan['status'] + '; human_review_complete=false.'),
        ('Nguồn và checkpoint', 'Các span/hash và liên kết nguồn được kiểm tra; checkpoint production không đổi.')])
    b.p('14 mục field không phải 14 câu độc lập: cùng một evidence_id có thể xuất hiện ở activity, deliverable và technology. Demo chứng minh tạo được task có nguồn; chưa chứng minh model sinh được task con/checklist/phụ thuộc hữu ích vì các danh sách đó đang rỗng.', True)
    b.table(['Task', 'Mô tả nguyên văn từ nguồn', 'Evidence ID'], [
        (step['task_id'], step['description'], ', '.join(step['evidence_ids'])) for step in plan['steps']])
    b.p('Một trường hợp cần thầy nhận xét: T002 “The result will be a source-grounded HTML report…” là câu kết quả mong đợi được cue nhận diện thêm như deliverable và đưa vào catalog. Việc câu đó có đủ tư cách là task thực thi cần người phân xử; validator nguồn không giải quyết được ranh giới ngữ nghĩa này.')
    b.table(['Câu trong input', 'Classifier dự đoán', 'Xác suất lớn nhất'], [
        (item['input']['text'], item['label'], f"{max(item['probabilities']):.4f}") for item in preds['items']])
    title_score = max(next(item for item in preds['items'] if item['input']['text'] == 'PSAI authored PDF smoke fixture - page 2')['probabilities'])
    b.p(f'Bảng trên là dự đoán trên mẫu demo; chưa có nhãn người để tính precision/recall cho mẫu này. Ví dụ tiêu đề trang 2 được argmax thành REQUIREMENT nhưng chỉ đạt {title_score:.4f}, nên không vượt ngưỡng trích nội dung. Xác suất softmax không đồng nghĩa xác suất đúng đã calibration.')
    b.table(['Field nội dung', 'Số mục'], [(name, len(entries)) for name, entries in fields.items()])
    b.p('Quy tắc nhận diện được “1200 USD” và các lần nhắc Python. Cụm “two developers over six weeks” chưa thành personnel/duration do pattern hiện nhận số dạng chữ số. Đây là ví dụ thực tế cho giới hạn coverage; “not_detected” không có nghĩa tài liệu chắc chắn thiếu nhân sự hoặc thời lượng.')
    b.details('Văn bản đã trích từ toàn bộ PDF mẫu — xem trực tiếp', source)
    for page in analysis['source']['pages']:
        if page.get('image_path'):
            path = run / page['image_path']
            if sha(path) != page['image_sha256']:
                raise ValueError('OCR image evidence hash mismatch')
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            b.html.append(f'<details><summary>Ảnh scan nguồn trang {page["page"]} — đã đối chiếu hash</summary><img class="ocr" alt="Trang scan nguồn của PDF mẫu" src="data:image/png;base64,{encoded}"></details>')
    b.md += ['Ảnh scan được nhúng trong bản HTML độc lập; vị trí ảnh gốc: [' + 'trang OCR' + '](' + rel(run / analysis['source']['pages'][-1]['image_path']) + ').', '']
    fixture = ROOT / receipt['fixture']
    b.html.append('<p><a download="complete_project_fixture.pdf" href="data:application/pdf;base64,' + base64.b64encode(fixture.read_bytes()).decode('ascii') + '">Tải PDF mẫu gốc đã nhúng trong báo cáo để đối chiếu</a></p>')
    b.md += ['[PDF mẫu gốc](' + rel(fixture) + ')', '']

    b.heading('7. Cấu trúc bộ đầu ra và cách đọc', 'output-files')
    b.table(['Đầu ra', 'Nội dung và đối tượng sử dụng'], [
        ('report.html', 'Người dùng/thầy xem nội dung chính, task, cảnh báo, text nguồn và ảnh OCR ngay trong một báo cáo.'),
        ('workspace.html', 'Người dùng đổi proposed/in_progress/blocked/done, đánh dấu checklist, ghi chú, duyệt dependency và xuất/nhập tiến độ JSON.'),
        ('analysis.json', 'Bản tổng hợp schema_version=2, gắn source/extraction/retrieval/plan/source_plan/rule_facts/resource_matching/metadata.'),
        ('classifier_predictions.json', 'Đầu vào từng câu và context; nhãn argmax; vector năm xác suất.'),
        ('extraction.json / retrieval.json', 'Nội dung phân loại, đoạn truy xuất và bằng chứng quote/span/source hash.'),
        ('plan.json / source_plan.json', 'Kế hoạch LLM và danh sách ứng viên theo nguồn; có raw response/prompt/catalog/provenance trong plan.'),
        ('rule_facts.json', 'Dữ kiện theo quy tắc, cách chuẩn hóa, ambiguity và vị trí nguồn.'),
        ('source.txt / source_metadata.json / sources / page_images', 'Snapshot text, thông tin trang PDF, nguồn resource và ảnh OCR có hash.'),
        ('manifest.json / run_manifest.json / jobs', 'Cấu hình, checkpoint provenance, request/response, thời gian và stdout/stderr của từng worker.'),
        ('progress_initial.json / JSON tiến độ xuất từ workspace', 'Trạng thái gắn đúng source/analysis hash; tiến độ việc làm, không phải điểm chất lượng model.')])
    example = {'document_id': analysis['document_id'], 'source_sha256': plan['source_sha256'],
               'planning_method': plan['planning_method'], 'status': plan['status'],
               'human_review_complete': plan['human_review_complete'], 'steps': plan['steps'][:1],
               'dependency_candidates': plan['dependency_candidates']}
    b.p('Trích JSON thực tế bên dưới chỉ giữ một task và một số trường để dễ đọc; đây không phải toàn bộ plan.json.')
    b.code(pretty(example), 'json')
    b.p('Workspace hiện là HTML offline và lưu trạng thái cục bộ. Cần xuất JSON trước khi đóng để lưu chắc chắn; không có server phân quyền hoặc đồng bộ nhiều máy. Giữ report và workspace cùng thư mục để các liên kết nguồn hoạt động. Bản báo cáo gửi thầy này tự nhúng nội dung cần đọc; không cần thầy truy cập đường dẫn ổ D của sinh viên.')

    b.heading('8. Bằng chứng kiểm thử và hiệu năng hiện có', 'verification')
    b.table(['Stage model · fixture cụ thể', 'Thời gian worker (giây)', 'VRAM allocated tối đa (GiB)'], [
        (stage['stage'], f"{stage['elapsed_seconds']:.2f}", f"{receipt['gpu_peaks'][stage['stage']]['peak_allocated_bytes'] / 2**30:.2f}") for stage in receipt['stages']])
    b.p(f"Tổng ba worker: {sum(stage['elapsed_seconds'] for stage in receipt['stages']):.2f} giây, chưa cộng OCR/xuất báo cáo. Các số đo thuộc PDF mẫu ngắn trên RTX 4050 Laptop 6 GB, không phải cam kết tốc độ cho mọi tài liệu hoặc máy khác.")
    b.p(f"Log kiểm thử đã lưu: {release['unit_test_count']} test đạt. Những test này kiểm tra logic, tính toàn vẹn nguồn, schema, lỗi và luồng tích hợp; không phải {release['unit_test_count']} mẫu người chấm độ chính xác AI. Trong lần lập báo cáo đã chạy lại verify-only cho release Classifier, RAG và toàn project (bao gồm hash trọng số project); đều đạt. Không chạy lại inference hoặc train để tạo số mới.")
    b.bullets([
        'Lưu prompt, raw LLM response, các checkpoint/tokenizer/config hashes và phiên bản thư viện để kiểm chứng.',
        'Đối chiếu quote với char spans, source snapshot bytes và ảnh OCR trước xuất; kiểm tra anchor HTML.',
        'Worker phải trả đúng stage và source hash; lỗi giữ request/response/log trong thư mục diagnostics, không fallback rồi báo model thành công.',
        'Chặn task ID lạ, nguồn bị thay đổi, checklist/resource không đúng substring, parent/dependency có chu trình hoặc progress JSON gắn sai analysis.',
        'Giữ nguyên checkpoint production; không dùng final test trong lựa chọn model hay lần nghiệm thu chức năng này.'
    ])

    b.heading('9. Tình trạng hiện tại theo từng phần', 'status')
    b.table(['Hạng mục', 'Đã có bằng chứng', 'Còn cần xác nhận'], [
        ('Pipeline cục bộ', 'Chạy xuyên suốt bằng ensemble + hybrid + Qwen trên PDF hỗn hợp.', 'Nhiều tài liệu thực, tài liệu dài và môi trường máy khác.'),
        ('PDF/OCR', 'Đọc selectable pages, OCR một scan tiếng Anh, giữ ảnh và vị trí nguồn.', 'Thứ tự đọc/bảng/layout, scan xấu, tiếng Việt; trang text lẫn ảnh có thể cần OCR always hoặc kiểm tra.'),
        ('Classifier', 'Train/evaluate/infer, sáu lượt study nguồn sạch, checkpoint V2 đã chọn bằng validation.', 'Phân xử nhãn tranh cãi, final test độc lập, so sánh công bằng với ensemble đang chạy.'),
        ('Retriever', 'BM25/dense/hybrid, multi-query, nguồn ngoài riêng, LoRA đã train và có paired analysis.', 'Nhãn relevance do người, query thực tế, kiểm chứng adapter và tác động lên planning.'),
        ('LLM planning', 'Model chạy thật, chọn sáu task có nguồn và schema được kiểm định.', 'Chất lượng ngữ nghĩa, độ đủ task; case end-to-end sinh và người duyệt checklist/task con/dependency hữu ích.'),
        ('Rule/resource', 'Dữ kiện literal và cảnh báo có nguồn, giữ mơ hồ.', 'Hiểu số viết chữ, coverage thực tế; chưa đưa rule facts chi tiết vào LLM.'),
        ('Workspace', 'HTML offline, trạng thái, ghi chú, kiểm định và xuất/nhập tiến độ.', 'Nếu cần triển khai nhóm: backend/auth/sync là phạm vi bổ sung.'),
        ('Đánh giá toàn hệ thống', 'Nghiệm thu chức năng và nguồn đã đạt trên fixture.', 'Chưa có điểm chất lượng đầu cuối độc lập; không nên báo “đã hoàn thiện chất lượng 100%”.')])

    b.heading('10. Các việc ưu tiên để xin thầy góp ý', 'feedback')
    b.table(['Ưu tiên', 'Việc nên thực hiện tiếp', 'Tiêu chí để thầy nhận xét'], [
        ('1', 'Chốt định nghĩa task vs deliverable vs expected result và nhãn tham chiếu tranh cãi.', 'Ví dụ T002 có nên là task? Một câu có nhiều vai trò xử lý thế nào?'),
        ('2', 'Chuẩn bị nguồn độc lập và nhãn người cho cả classifier, retrieval và planning.', 'Số tài liệu, ngôn ngữ, loại PDF, người chấm, thang đánh giá và cách xử lý bất đồng.'),
        ('3', 'Đánh giá nguồn trích đúng/đủ, số task đúng/thiếu/thừa, citation correctness và dependency đúng.', 'Phân biệt lỗi OCR, nhãn, retrieval, task eligibility, LLM và nhãn tham chiếu.'),
        ('4', 'Xác định mức tích hợp ba nhánh mà đồ án cần đạt.', 'Có bắt buộc đưa rule facts vào prompt? RAG chỉ hỗ trợ tham khảo hay phải cải thiện chất lượng task?'),
        ('5', 'Thử case thực có checklist, task con, nguồn lực và phụ thuộc; có người duyệt.', 'Không coi schema có field là đã chứng minh sinh cấu trúc đó có ích.'),
        ('6', 'Sau khi chốt giao thức và model, chấm final test đúng một quy trình đã định.', 'Không dùng kết quả final test để chọn tiếp tham số hoặc đổi model; báo cả hạn chế.')])

    b.heading('11. Khi thầy yêu cầu xem code và demo', 'code')
    code_map = [
        ('Điểm chạy demo', 'tools/demo_project.py', 'def main', 'Chọn đầy đủ hoặc fast; input/output và tham số.'),
        ('Điều phối', 'psai_project/runtime.py', 'def run_project', 'Đọc nguồn → worker extract → retrieve → plan → export.'),
        ('PDF/OCR', 'psai_ocr/extract.py', 'def extract_pdf_with_ocr', 'Quyết định trang nào OCR, metadata và hash.'),
        ('Tạo context và trích nội dung', 'extraction.py', 'def extract_document', 'Năm nhãn, ngưỡng, noise filter và evidence spans.'),
        ('Ensemble', 'ensemble_classifier.py', 'def classify_probs', 'Ba model và trung bình xác suất năm lớp.'),
        ('Retrieval', 'psai_retrieval/runtime.py', 'def retrieve_document', 'BM25/dense/hybrid, top-k, resource scope và provenance.'),
        ('Catalog và LLM', 'psai_planner/planner.py', 'def build_catalog', 'Catalog task có nguồn, prompt, inference và validator.'),
        ('Rule facts', 'psai_rules/facts.py', 'def extract_facts', 'Trích dữ kiện literal và giữ vị trí nguồn.'),
        ('Xuất báo cáo', 'psai_project/report.py', 'def export_project_report', 'Tái kiểm định kế hoạch, thêm rule facts và xuất HTML/JSON.'),
        ('Workspace', 'psai_project/workspace.py', 'def validate_progress', 'Kiểm định trạng thái, checklist, dependency và hash.')]
    line_map = []
    for purpose, file, symbol, explanation in code_map:
        lines = (ROOT / file).read_text(encoding='utf-8').splitlines()
        line_no = next(i+1 for i, line in enumerate(lines) if line.startswith(symbol + '('))
        line_map.append((purpose, file, line_no, explanation))
    b.table(['Phần cần show', 'File và dòng bắt đầu', 'Điểm cần giải thích'], [
        (purpose, f'{file}:{line_no}', explanation) for purpose, file, line_no, explanation in line_map])
    b.p('Bảng trên chỉ vị trí code cần giải thích. Các liên kết trong bản HTML dẫn đến trích code nhúng ngay trong báo cáo; bản Markdown có thêm liên kết tương đối đến file gốc trong workspace. Khi gửi riêng HTML cho thầy, phụ lục vẫn đọc trực tiếp được mà không lệ thuộc đường dẫn máy sinh viên.')
    for purpose, file, line_no, _ in line_map:
        b.html.append(f'<p class="code-link"><a href="#code-{line_no}-{Path(file).stem}">{escape(purpose)} · {escape(file)}:{line_no} · xem trích code nhúng</a></p>')
        b.md += [f'- {purpose}: [{file}]({rel(ROOT / file)}#L{line_no})', '']
    b.p('Demo đầy đủ trên máy hiện tại đã có model:')
    b.code('python tools/demo_project.py\npython tools/demo_project.py --input "DUONG_DAN_DEN_PDF.pdf" --device cuda', 'powershell')
    b.p('Lệnh đầu mặc định dùng ví dụ Markdown; lệnh thứ hai đưa PDF cụ thể đi qua nhánh đọc/OCR. Không dùng --fast để trình bày như đã chạy Classifier/LLM. --fast chỉ dùng kiểm tra giao diện rules + BM25 + source plan. Bản chạy thật PDF đã lưu có thể mở ngay khi không muốn chờ nạp model.')

    b.heading('12. Đoạn trình bày có thể nói trực tiếp với thầy', 'speech')
    b.p('Thưa thầy, hiện PSAI đã kết nối được luồng xử lý cục bộ từ PDF đến báo cáo và không gian theo dõi công việc. Hệ thống đọc văn bản hoặc OCR theo trang, phân loại câu vào năm nhóm bằng ensemble mDeBERTa, truy xuất bằng BM25 kết hợp BGE-M3, rồi dùng Qwen cục bộ chọn và tổ chức công việc từ các bằng chứng có sẵn. Kết quả giữ câu trích, vị trí nguồn và log để kiểm tra lại.')
    b.p('Về kiểm chứng kỹ thuật, lượt thử trên PDF tự tạo ba trang, gồm một trang scan, đã chạy qua các model thật và tạo sáu công việc đề xuất. Bộ kiểm thử phần mềm có 294 test đạt. Tuy nhiên, đây là bằng chứng pipeline hoạt động và truy được nguồn; em chưa dùng nó để khẳng định chất lượng kế hoạch trên tài liệu thực tế.')
    b.p('Về nghiên cứu, Classifier V2 đã được chọn bằng validation sau xử lý trùng nguồn, Macro-F1 khoảng 0,6321; bản đó đang được lưu riêng và chưa thay ensemble của pipeline. Retriever LoRA cũng đã train xong, nhưng mức thay đổi hiện chưa đủ chắc chắn để thay BGE-M3 gốc. Qwen là model pretrained được tích hợp để inference, không phải LLM do project tự train.')
    b.p('Phần em muốn xin thầy nhận xét là tiêu chí xác định một task đúng, cách đánh giá độ đầy đủ của kế hoạch và mức tích hợp cần đạt giữa classifier, retrieval và rule. Hiện các cấu trúc checklist, task con, phụ thuộc đã có schema và kiểm định, nhưng mẫu nghiệm thu chưa sinh ra những cấu trúc đó; phần đánh giá độc lập bằng người và final test vẫn cần hoàn tất theo tiêu chí thống nhất.')

    b.heading('13. Bằng chứng nhúng và đường dẫn đối chiếu', 'evidence')
    artifacts = [
        ('BC01', receipt_path, receipt),
        ('BC02', run / 'plan.json', plan),
        ('BC03', run / 'classifier_predictions.json', preds),
        ('BC04', run / 'rule_facts.json', analysis['rule_facts']),
        ('BC05', classifier_path, {k: v for k, v in classifier.items() if k != 'artifact_sha256'}),
        ('BC06', study_path / 'summary.json', study),
        ('BC07', study_path / 'experiments/baseline_seed_2026/validation_metrics.json', metrics),
        ('BC08', rag_path / 'base_metrics.json', {'metrics': base['metrics'], 'label_kind': base['label_kind'], 'validation_queries': base['validation_queries'], 'weak_label_notice': base['weak_label_notice']}),
        ('BC09', rag_path / 'adapter_metrics.json', {'metrics': adapter['metrics'], 'label_kind': adapter['label_kind'], 'weak_label_notice': adapter['weak_label_notice']}),
        ('BC10', paired_path, paired),
        ('BC11', release_path, {k: v for k, v in release.items() if k != 'files_sha256'}),
        ('BC12', ROOT / 'reports/project_completion/unit_tests.log', (ROOT / 'reports/project_completion/unit_tests.log').read_text(encoding='utf-8')[-2000:]),
        ('BC13', run / 'progress_initial.json', load(run / 'progress_initial.json'))]
    inventory = []
    for identity, path, value in artifacts:
        digest = sha(path)
        relative_root = path.relative_to(ROOT).as_posix()
        inventory.append({'id': identity, 'path': relative_root, 'sha256': digest})
        b.details(identity + ' · ' + relative_root + ' · nội dung hoặc phần liên quan',
                  'SHA-256 file gốc: ' + digest + '\n\n' + (value if isinstance(value, str) else pretty(value)))
        b.md += ['[Mở file bằng chứng gốc](' + rel(path) + ')', '']
    b.p('Một số phụ lục chỉ trích phần liên quan để đọc gọn; SHA-256 trên đầu phụ lục là hash của file gốc đầy đủ. BC02 giữ toàn bộ plan gồm prompt, raw response và catalog. Các đường dẫn là vị trí truy nguyên trên workspace; nội dung cốt lõi đã nằm trong chính bản HTML này.')
    b.heading('14. Trích code trọng tâm để thầy đọc ngay', 'code-evidence')
    snippet_specs = [
        ('psai_project/runtime.py', 'def run_project', 98),
        ('ensemble_classifier.py', 'def classify_probs', 39),
        ('extraction.py', 'def extract_document', 85),
        ('psai_planner/planner.py', 'def build_catalog', 45),
        ('psai_project/report.py', 'def export_project_report', 67)]
    for file, symbol, maximum in snippet_specs:
        lines = (ROOT / file).read_text('utf-8').splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith(symbol + '('))
        code = '\n'.join(f'{i+1:4}  {lines[i]}' for i in range(first, min(first + maximum, len(lines))))
        identity = f'code-{first+1}-{Path(file).stem}'
        b.html.append(f'<div id="{identity}"></div>')
        b.details(f'{file}:{first+1} · trích code hiện có (có thể kết thúc trước cuối hàm)', code)
    # Other code links point to a compact explanation rather than a missing anchor.
    for purpose, file, line_no, explanation in line_map:
        identity = f'code-{line_no}-{Path(file).stem}'
        if any(file == f and symbol + '(' in (ROOT / f).read_text('utf-8') and
               line_no == next(i+1 for i,l in enumerate((ROOT / f).read_text('utf-8').splitlines()) if l.startswith(symbol+'('))
               for f, symbol, _ in snippet_specs):
            continue
        lines = (ROOT / file).read_text('utf-8').splitlines()
        code = '\n'.join(f'{i+1:4}  {lines[i]}' for i in range(line_no-1, min(line_no+18, len(lines))))
        b.html.append(f'<div id="{identity}"></div>')
        b.details(f'{purpose} · {file}:{line_no} · trích đầu hàm', code)

    nav = '<nav aria-label="Mục lục">' + ''.join(f'<a href="#{identity}">{escape(title)}</a>' for identity, title in b.nav) + '</nav>'
    style = '''
    :root{color-scheme:light;--ink:#203447;--blue:#115a83;--muted:#60778a;--line:#d5e0e8}
    *{box-sizing:border-box}body{margin:0;background:#edf2f7;color:var(--ink);font:16px/1.65 "Segoe UI",Arial,sans-serif}
    main{max-width:1170px;background:white;margin:30px auto;padding:44px 52px;border-radius:14px;box-shadow:0 4px 24px #233b5010}
    h1{font-size:34px;line-height:1.25;letter-spacing:-.5px;color:#103c5b}h2{font-size:25px;margin-top:44px;padding-top:14px;border-top:2px solid #dce8f2;color:var(--blue);scroll-margin-top:12px}
    p,li,td{overflow-wrap:anywhere}a{color:#075e93}nav{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0}nav a{font-size:13px;text-decoration:none;padding:6px 10px;border:1px solid var(--line);border-radius:7px;background:#f5f9fc}
    .notice{padding:17px 20px;background:#fff6e5;border-left:5px solid #c79126;border-radius:5px;font-weight:500}.table-wrap{overflow-x:auto}
    table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0 24px}th,td{border:1px solid var(--line);padding:10px 12px;vertical-align:top}th{background:#eaf3fa;text-align:left;color:#194e70}tbody tr:nth-child(even){background:#f8fafc}
    pre{background:#f2f5f8;border:1px solid #dce3ea;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 Consolas,monospace;border-radius:6px}details{border:1px solid var(--line);border-radius:7px;padding:12px 15px;margin:12px 0}summary{cursor:pointer;color:#195b80;font-weight:600;font-size:14px;overflow-wrap:anywhere}
    .diagram{margin:20px 0}.diagram svg{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px}figcaption{font-size:13px;color:var(--muted);margin-top:8px}.ocr{display:block;max-width:100%;height:auto;border:1px solid var(--line);margin-top:15px}.code-link{font-size:14px;margin:5px 0}footer{color:var(--muted);font-size:13px;margin-top:35px}
    @media(max-width:700px){main{padding:20px 16px;margin:0;border-radius:0}h1{font-size:28px}h2{font-size:23px}table{font-size:12px}th,td{padding:7px}}
    @media print{@page{size:A4;margin:14mm}body{background:white;font-size:11px}main{margin:0;padding:0;max-width:none;box-shadow:none}nav{display:none}h1{font-size:25px}h2{font-size:19px;break-after:avoid}table{font-size:9px}th,td{padding:6px}tr,figure,.notice{break-inside:avoid}.diagram{max-width:100%}a{color:inherit;text-decoration:none}details{font-size:10px}pre{font-size:8px}footer{font-size:9px}}
    '''
    html = ('<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'; form-action \'none\'">'
            '<title>' + escape(TITLE) + '</title><style>' + style + '</style></head><body><main>' + b.html[0] + nav
            + ''.join(b.html[1:]) + '<footer>Bản đối chiếu phục vụ giảng viên nhận xét · Chỉ dùng bằng chứng đã lưu · Không thay checkpoint hoặc tự tạo nhãn chuẩn.</footer></main></body></html>')
    (OUT / 'PIPELINE_PSAI.svg').write_text(diagram, encoding='utf-8', newline='')
    (OUT / 'PIPELINE_VA_TINH_TRANG_PSAI.html').write_text(html, encoding='utf-8', newline='')
    (OUT / 'PIPELINE_VA_TINH_TRANG_PSAI.md').write_text('\n'.join(b.md), encoding='utf-8', newline='')
    (OUT / 'PIPELINE_VA_TINH_TRANG_PSAI.txt').write_text('\n'.join(b.text), encoding='utf-8', newline='')
    (OUT / 'EVIDENCE_INVENTORY.json').write_text(pretty({'created_local': date, 'training_performed': False,
        'independent_final_test_scored': False, 'evidence': inventory}), encoding='utf-8', newline='')
    print(OUT / 'PIPELINE_VA_TINH_TRANG_PSAI.html')
    print(OUT / 'PIPELINE_PSAI.svg')


if __name__ == '__main__':
    main()
