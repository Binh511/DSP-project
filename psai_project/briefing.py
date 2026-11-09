"""Formal project brief derived only from the current, source-validated run.

Counts describe extracted content and source traceability. They never stand in
for semantic accuracy, requirement satisfaction, or a human approval decision.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re


FIELD_NAMES = {
    'objectives': 'Mục tiêu', 'requirements': 'Yêu cầu', 'activities': 'Hoạt động',
    'expected_results': 'Kết quả mong đợi', 'deliverables': 'Sản phẩm bàn giao',
    'technologies': 'Công nghệ', 'resources': 'Nguồn lực',
}

STYLE = """
@page{size:A4;margin:17mm 16mm 19mm}
:root{font-family:'Segoe UI',Arial,sans-serif;color:#202c3c;background:#e9edf2;line-height:1.65}
body{max-width:1080px;padding:32px 24px 48px}*{box-sizing:border-box}
header,section{border:1px solid #d9e0e8;border-radius:0;padding:30px 36px;margin:0 0 22px;background:#fff}
header.report-cover{border-top:7px solid #183956;padding:32px 36px 28px}
.report-masthead{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid #d9e0e8;padding-bottom:18px}
.report-brand{font-size:22px;font-weight:800;letter-spacing:3px;color:#183956}
.report-series,.report-kicker{font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:#65758a}
.report-cover h1{font-size:34px;line-height:1.25;letter-spacing:-.6px;margin:26px 0 14px;max-width:780px}
.report-subtitle{font-size:20px;color:#40536b;margin:0 0 20px;overflow-wrap:anywhere}
.report-meta{display:grid;grid-template-columns:1fr 1fr;gap:10px 30px;margin:20px 0}
.report-meta div{border-bottom:1px solid #e8ecf1;padding:6px 0;font-size:13px;overflow-wrap:anywhere}
.report-meta dt{color:#65758a;font-size:11px;text-transform:uppercase;letter-spacing:.7px}
.report-meta dd{margin:3px 0 0;font-weight:600}
.report-mode{display:inline-block;background:#edf3f8;color:#183956;font-size:12px;font-weight:700;padding:5px 10px}
.report-status{display:inline-block;color:#88591a;background:#fff5e4;font-size:12px;padding:5px 10px}
.report-cover nav{border-top:1px solid #d9e0e8;padding-top:16px;margin-top:22px;gap:8px 20px;font-size:12px}
h2{font-size:21px;color:#183956;line-height:1.4;border-bottom:1px solid #e1e7ee;padding-bottom:12px}
h3{color:#304d69}a{color:#205e88;text-underline-offset:3px}
.report-intro{font-size:15px;color:#40536b}.report-summary-quote{margin:12px 0}
.report-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}
.report-metric{border:1px solid #dce4ec;padding:16px;background:#f8fafc}
.report-metric strong{display:block;font-size:30px;line-height:1.2;color:#183956}
.report-metric span{display:block;font-size:12px;color:#53667b;margin-top:8px}
.report-table-wrap{overflow-x:auto}.report-table{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}
.report-table th{text-align:left;color:#304d69;background:#eef3f8;font-weight:600}
.report-table th,.report-table td{border:1px solid #dce4ec;padding:11px 13px;vertical-align:top;overflow-wrap:anywhere}
.report-table .task-column{width:46%}.report-small{font-size:12px;color:#65758a}
.report-reference{font:11px/1.8 Consolas,monospace;display:inline-block;margin-right:8px}
.report-linked-quote{padding:9px 0;border-bottom:1px solid #e6edf2;white-space:pre-wrap}
.report-linked-quote:last-child{border-bottom:0}.report-table h4{margin:6px 0}
.report-extra-quotes{margin:9px 0}.report-extra-quotes summary{font-size:12px}
.report-review{border-left:3px solid #c29042;padding:10px 16px;margin:12px 0;background:#fffaf0}
.report-review strong{font-size:14px}.report-review p{margin:5px 0;font-size:13px}
.report-decision{padding:16px 20px;border:1px solid #d5e1eb;background:#f3f7fb;margin-top:18px}
.report-decision p{margin:6px 0;font-size:13px}.report-number{font-size:11px;color:#7c8b9c;letter-spacing:1px;display:block;margin-bottom:6px}
.report-pipeline{display:flex;flex-wrap:wrap;gap:7px;margin:18px 0}.report-pipeline span{font-size:12px;padding:7px 11px;background:#eef3f8;border:1px solid #dce4ec}
.evidence{padding:14px 0}blockquote{background:#f8fafc;border-left:3px solid #92abc2;font-size:14px}
.notice{border-left:3px solid #c29042;background:#fffaf0;font-size:13px;padding:12px 16px}
.source-text{background:#fbfcfe}.footer{text-align:center;font-size:11px;padding:12px}
@media(max-width:700px){body{padding:12px}header.report-cover,section{padding:22px 18px}.report-cover h1{font-size:26px}.report-subtitle{font-size:17px}.report-metrics{grid-template-columns:1fr 1fr}.report-meta{grid-template-columns:1fr}.report-series{max-width:150px;text-align:right}}
@media print{
 :root,body{background:white;color:#182735}body{max-width:none;padding:0;font-size:10pt}
 header.report-cover,section{border:0;border-bottom:1px solid #cbd3dd;padding:0 0 6mm;margin:0 0 8mm;box-shadow:none}
 header.report-cover{border-top:2mm solid #183956;padding-top:7mm;break-after:page}
 .report-cover h1{font-size:27pt;margin-top:14mm}.report-subtitle{font-size:16pt}
 .report-cover nav{display:none}.report-meta{margin-top:12mm;grid-template-columns:1fr 1fr}
 h2{font-size:15pt;break-after:avoid}h3,h4{break-after:avoid}p{orphans:3;widows:3}
 .report-metrics{grid-template-columns:repeat(4,1fr)}.report-metric{padding:3mm}.report-metric strong{font-size:23pt}
 .report-table{font-size:9pt}.report-table th,.report-table td{padding:2.5mm}thead{display:table-header-group}
 tr,.evidence,.report-review,.report-decision,.report-metrics{break-inside:avoid}
 .report-review-table tr{break-inside:auto}.report-linked-quote{break-inside:avoid}
 .report-extra-quotes::details-content{display:block!important;content-visibility:visible!important}
 .report-extra-quotes>:not(summary){display:block!important}
 .report-table-wrap,.source-text{overflow:visible}.source-text{font-size:8pt}
 .source-line{break-inside:avoid}.source-line:target{outline:none;background:transparent}
 a{color:inherit}.report-small,.muted{font-size:9pt}.footer{font-size:8pt}
 *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


def _safe(value):
    return escape(str(value), quote=True)


def _time(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.strftime('%d/%m/%Y · %H:%M') + ' (không ghi múi giờ)'
        return parsed.astimezone(timezone.utc).strftime('%d/%m/%Y · %H:%M UTC')
    except (TypeError, ValueError):
        return 'Chưa ghi nhận'


def _title(text, document_id):
    for line in text.splitlines():
        if re.match(r'^#{1,2}\s+\S', line):
            return re.sub(r'^#+\s+', '', line).strip()
    return document_id


def _source_reference(item, digest, starts, source_text):
    """Check the quote before introducing any new source link in the brief."""
    start, end, quote = item.get('char_start'), item.get('char_end'), item.get('quote')
    if (item.get('verified') is not True or item.get('source_sha256', digest) != digest
            or type(start) is not int or type(end) is not int or not isinstance(quote, str)
            or not 0 <= start < end <= len(source_text) or source_text[start:end] != quote):
        raise ValueError('Project brief received a quotation that differs from its source')
    line = bisect_right(starts, start)
    return f'<a class="report-reference" href="#src-{digest}-L{line}">dòng {line}</a>'


def _omission_messages(counts):
    """Keep unlike units and catalog/prompt exclusions separate."""
    policies = {
        'tasks': 'công việc ngoài catalog do giới hạn cấu hình',
        'project_resources': 'đoạn nguồn lực ngoài catalog do giới hạn cấu hình',
        'external_references': 'đoạn tham khảo ngoài catalog do giới hạn cấu hình',
        'context_evidence': 'đoạn bối cảnh ngoài catalog do giới hạn cấu hình',
        'document_context': 'đoạn từ kết quả truy xuất ngoài catalog do giới hạn cấu hình',
        'link_candidates': 'cặp liên kết gợi ý không vào danh sách lựa chọn do giới hạn cấu hình',
        'prompt_evidence_without_link_candidate':
            'đoạn bằng chứng có trong catalog nhưng không đưa vào prompt vì không có liên kết ứng viên được giữ lại',
        'source_items': 'đoạn nguồn chưa đưa vào tập rà soát do giới hạn cấu hình',
        'lexical_task_links': 'cặp liên kết theo từ khóa chưa đưa vào tập rà soát do giới hạn cấu hình',
    }
    return [f'{count} ' + policies.get(key, f'mục theo bộ đếm {key}; cần đối chiếu định nghĩa trong dữ liệu lượt chạy')
            + '.' for key, count in counts.items() if count]


def _review_context_html(context, steps, digest, starts, source_text):
    """Present lexical review candidates separately from approved plan claims."""
    if context is None:
        return ''
    if context.get('source_sha256') != digest:
        raise ValueError('Project brief review context differs from the source')
    items = context.get('evidence', [])
    index = {item['evidence_id']: item for item in items}
    if len(index) != len(items):
        raise ValueError('Project brief review evidence IDs must be unique')
    references = {identity: _source_reference(item, digest, starts, source_text)
                  for identity, item in index.items()}
    selected = {}
    context_labels = {}
    for number, step in enumerate(steps, 1):
        label = step.get('task_id', f'S{number:03d}')
        selected[label] = step
        for link in context.get('task_links', []):
            if (step.get('task_id') == link.get('task_id') or
                    (not step.get('task_id') and set(step.get('evidence_ids', []))
                     == set(link.get('task_evidence_ids', [])))):
                context_labels[link['task_id']] = label
    links = [{**link, 'task_id': context_labels[link['task_id']]}
             for link in context.get('task_links', []) if link.get('task_id') in context_labels]

    def excerpt(item):
        return ('<div class="report-linked-quote">' + _safe(item['quote']) + '<br>'
                + references[item['evidence_id']] + '</div>')

    def linked_for(task_id, category):
        ids = {link['context_evidence_id'] for link in links
               if link['task_id'] == task_id and category in link.get('categories', [])}
        return [item for item in items if item['evidence_id'] in ids]

    def excerpts_for(items):
        if not items:
            return 'Chưa có liên kết gợi ý theo quy tắc'
        visible = ''.join(excerpt(item) for item in items[:2])
        if len(items) > 2:
            visible += ('<details class="report-extra-quotes"><summary>'
                        + str(len(items) - 2) + ' đoạn liên quan còn lại — mở để đối chiếu</summary>'
                        + ''.join(excerpt(item) for item in items[2:]) + '</details>')
        return visible

    parts = ['<h3 id="review-links">Gợi ý liên kết yêu cầu – công việc – nghiệm thu</h3>',
             '<p class="notice">Các liên kết dưới đây dựa trên từ khóa chung và vị trí nội dung trong nguồn. '
             'Chúng là ứng viên để người duyệt đối chiếu, chưa xác nhận công việc xử lý đúng yêu cầu, '
             'chưa xác nhận tiêu chí đủ hoặc yêu cầu đã được đáp ứng. '
             'Nội dung này hỗ trợ rà soát độc lập với phần lựa chọn của LLM.</p>']
    if selected:
        parts += ['<div class="report-table-wrap"><table class="report-table report-review-table"><thead><tr>'
                  '<th>Công việc</th><th>Yêu cầu có liên kết gợi ý</th><th>Tiêu chí nghiệm thu có liên kết gợi ý</th>'
                  '</tr></thead><tbody>']
        for identity, step in selected.items():
            requirements, acceptance = linked_for(identity, 'requirements'), linked_for(identity, 'acceptance')
            parts += ['<tr><td><strong>' + _safe(identity) + '</strong><br>' + _safe(step['description'])
                      + '</td><td>' + excerpts_for(requirements)
                      + '</td><td>' + excerpts_for(acceptance) + '</td></tr>']
        parts += ['</tbody></table></div>']

    for category, label in [('requirements', 'Yêu cầu chưa có liên kết gợi ý với công việc đã chọn'),
                            ('acceptance', 'Tiêu chí nghiệm thu chưa có liên kết gợi ý với công việc đã chọn')]:
        linked_ids = {link['context_evidence_id'] for link in links if category in link.get('categories', [])}
        unmatched = [item for item in items if category in item.get('categories', [])
                     and item['evidence_id'] not in linked_ids]
        parts += ['<h4>' + label + f' ({len(unmatched)} đoạn)</h4>']
        if unmatched:
            parts += [excerpt(item) for item in unmatched]
        else:
            parts += ['<p class="report-small">Không ghi nhận đoạn chưa liên kết trong tập đã nhận diện. '
                      'Điều này không xác nhận mức bao phủ toàn bộ tài liệu.</p>']

    constraints = [item for item in items if 'dependency_constraints' in item.get('categories', [])]
    parts += ['<h3>Ràng buộc phụ thuộc được đề cập trong nguồn</h3>',
              '<p class="report-small">Giữ nguyên ý nghĩa điều kiện, phê duyệt và thời điểm trong câu nguồn. '
              'Chưa tự chuyển thành quan hệ kết thúc–bắt đầu, lịch thực hiện hoặc quan hệ đã được duyệt.</p>']
    if not constraints:
        parts += ['<p>Chưa nhận diện ràng buộc theo quy tắc; cần đối chiếu nguồn.</p>']
    for item in constraints:
        candidates = sorted({link['task_id'] for link in links
                             if link['context_evidence_id'] == item['evidence_id']})
        parts += [excerpt(item), '<p class="report-small">'
                  + ('Công việc có từ khóa liên quan, cần xác nhận ánh xạ: ' + _safe(', '.join(candidates))
                     if candidates else 'Chưa có công việc khớp theo quy tắc; cần xác nhận ánh xạ.') + '</p>']
    if context.get('review_questions'):
        parts += ['<h3>Câu hỏi rà soát từ nội dung nguồn</h3><ul>']
        questions = {
            'personnel': 'Nguồn đã có mô tả nhân sự. Ai thực sự sẵn sàng tham gia và ai chịu trách nhiệm từng công việc?',
            'measures': 'Cần xác nhận cách đo, điều kiện, mốc so sánh và mẫu số cho từng chỉ tiêu; chưa có bằng chứng đạt mục tiêu.',
            'dependency_constraints': 'Ràng buộc áp dụng ở giai đoạn nào của công việc? Trước khi phê duyệt chưa chắc đồng nghĩa trước khi bắt đầu.',
            'open_questions': 'Những quyết định nào nguồn đang để mở và ai cần xác nhận trước khi cam kết triển khai?',
        }
        for question in context['review_questions']:
            refs = ' '.join(references[identity] for identity in question.get('evidence_ids', [])
                            if identity in references)
            parts += ['<li>' + _safe(questions.get(question.get('kind'), question['message'])) + ' ' + refs + '</li>']
        parts += ['</ul>']
    omitted = context.get('omitted_counts', {})
    if any(omitted.values()):
        parts += ['<div class="notice">Giới hạn của tập rà soát:<ul>'
                  + ''.join('<li>' + _safe(message) + '</li>' for message in _omission_messages(omitted))
                  + '</ul>Các bộ đếm dùng đơn vị và phạm vi khác nhau; không cộng thành tổng nội dung bị bỏ sót.</div>']
    return ''.join(parts)


def present_project_report(rendered, record, plan, retrieval, facts, warnings, metadata, source_text,
                           *, review_context=None):
    """Add a formal brief after the underlying renderer has validated citations."""
    fields = record.get('fields', {})
    evidence = {item['evidence_id']: item for items in fields.values() for item in items}
    digest = record['source_sha256']
    starts, cursor = [0], 0
    for line in source_text.splitlines(keepends=True):
        cursor += len(line)
        starts.append(cursor)
    citation_evidence = dict(evidence)
    for name in ('context_evidence', 'project_resources', 'document_references'):
        for item in plan.get('catalog', {}).get(name, []):
            citation_evidence.setdefault(item['evidence_id'], {**item, 'verified': True})

    def refs(ids):
        links = []
        for identity in ids:
            item = citation_evidence[identity]
            _source_reference(item, digest, starts, source_text)
            line = bisect_right(starts, item['char_start'])
            links.append(f'<a class="report-reference" href="#src-{digest}-L{line}">'
                         f'{_safe(identity)} · L{line}</a>')
        return ' '.join(links)

    classifier = metadata.get('classifier', 'unknown')
    retriever = metadata.get('retrieval_mode', retrieval.get('mode', 'unknown'))
    planner = metadata.get('planner_mode', 'local' if plan.get('provenance') else 'evidence')
    mode = ('FAST · Quy tắc / BM25 / kế hoạch theo nguồn' if
            (classifier, retriever, planner) == ('rules', 'bm25', 'evidence') else
            'FULL · Ensemble / hybrid / LLM cục bộ' if
            (classifier, retriever, planner) == ('ensemble', 'hybrid', 'local') else
            f'Cấu hình: {classifier} / {retriever} / {planner}')
    document_id = record['document_id']
    title = _title(source_text, document_id)
    source_name = Path(metadata.get('input_path') or document_id).name
    source_method = metadata.get('source', {}).get('method', 'local_text')
    source_label = {'local_text': 'Đọc trực tiếp văn bản'}.get(source_method, source_method)
    cover = f'''<header class="report-cover">
<div class="report-masthead"><div><div class="report-brand">PSAI</div><div class="report-kicker">Project proposal analysis</div></div>
<div class="report-series">Báo cáo phân tích &amp; hỗ trợ lập kế hoạch<br>Hồ sơ {_safe(digest[:12].upper())}</div></div>
<h1>BÁO CÁO PHÂN TÍCH<br>ĐỀ XUẤT DỰ ÁN</h1><p class="report-subtitle">{_safe(title)}</p>
<span class="report-mode">{_safe(mode)}</span> <span class="report-status">Bản đề xuất · cần người duyệt</span>
<dl class="report-meta"><div><dt>Tài liệu đầu vào</dt><dd>{_safe(source_name)}</dd></div>
<div><dt>Thời điểm bắt đầu phân tích</dt><dd>{_safe(_time(metadata.get('created_utc')))}</dd></div>
<div><dt>Phương thức đọc nguồn</dt><dd>{_safe(source_label)}</dd></div>
<div><dt>Phạm vi</dt><dd>Trích xuất, truy xuất và đề xuất công việc</dd></div></dl>
<p class="report-intro">Hồ sơ tổng hợp nội dung nhận diện từ tài liệu, liên kết từng công việc với đoạn nguồn
và chỉ ra các thông tin cần xác nhận trước khi triển khai.</p>
<p class="report-small">Mã hồ sơ nhận diện nội dung nguồn; các lượt chạy cùng tài liệu có thể dùng cùng mã.
Lưu thành PDF bằng chức năng In của trình duyệt, chọn khổ A4.</p>
<nav><a href="workspace.html">Mở không gian tiến độ</a><a href="#executive-summary">01. Tổng quan</a>
<a href="#traceability">02. Truy vết</a><a href="#review-register">03. Điểm cần duyệt</a>
<a href="#run-method">04. Phương pháp</a><a href="#extraction-details">05. Kết quả chi tiết</a>
<a href="#sources">Nguồn đối chiếu</a><a href="#provenance">Thông tin tái lập</a></nav></header>'''

    unique_spans = {(item['char_start'], item['char_end']) for item in evidence.values()}
    steps = plan.get('steps', [])
    metrics = [(len(unique_spans), 'Đoạn nguồn phân biệt đã trích'), (len(steps), 'Công việc được đề xuất'),
               (len(retrieval.get('hits', [])), 'Đoạn truy xuất được trả về'), (len(warnings), 'Cảnh báo theo quy tắc')]
    metric_html = ''.join(f'<div class="report-metric"><strong>{count}</strong><span>{label}</span></div>'
                          for count, label in metrics)
    overview = ['<section id="executive-summary"><h2><span class="report-number">01 / TỔNG QUAN</span>Tóm tắt điều hành</h2>',
                '<p>Bản phân tích hỗ trợ rà soát đề xuất và chuẩn bị danh sách công việc ban đầu. '
                'Các công việc và quan hệ được đề xuất cần được người có chuyên môn xác nhận.</p>',
                '<div class="report-metrics">' + metric_html + '</div>']
    for field, label in [('objectives', 'Mục tiêu nhận diện'), ('expected_results', 'Kết quả kỳ vọng nhận diện')]:
        overview.append('<h3>' + label + '</h3>')
        items = fields.get(field, [])
        for item in items[:2]:
            overview.append('<blockquote class="report-summary-quote">' + _safe(item['quote'])
                            + '</blockquote>' + refs([item['evidence_id']]))
        if not items:
            overview.append('<p class="muted">Chưa nhận diện được; cần kiểm tra lại tài liệu nguồn.</p>')
        if len(items) > 2:
            overview.append('<p class="report-small">Xem các nội dung còn lại trong phần kết quả chi tiết.</p>')
    rows = ''.join('<tr><td>' + _safe(FIELD_NAMES.get(field, field)) + '</td><td>' + str(len(items))
                   + '</td><td>' + ('Có nội dung để đối chiếu' if items else 'Chưa nhận diện; cần kiểm tra')
                   + '</td></tr>' for field, items in fields.items())
    overview.append('<h3>Tình trạng nhận diện thông tin</h3><table class="report-table"><thead><tr>'
                    '<th>Nhóm thông tin</th><th>Số đoạn trích</th><th>Tình trạng</th></tr></thead><tbody>' + rows
                    + '</tbody></table><p class="report-small">Một đoạn có thể xuất hiện ở nhiều nhóm. '
                    'Một đoạn có thể liệt kê nhiều sản phẩm bàn giao; số đoạn trích không phải số sản phẩm. '
                    'Các số đếm mô tả đầu ra của lượt chạy, không đo độ đầy đủ hoặc độ chính xác.</p>')
    overview.append('<h3>Thông tin điều hành được nhắc trong nguồn</h3><table class="report-table"><thead>'
                    '<tr><th>Hạng mục</th><th>Nội dung để xác nhận</th></tr></thead><tbody>')
    for field, label in [('budgets', 'Ngân sách'), ('dates', 'Ngày / thời hạn'), ('personnel', 'Nhân sự')]:
        entries = facts.get('fields', {}).get(field, [])
        values = []
        for item in entries[:4]:
            reference = _source_reference(item, digest, starts, source_text)
            detail = ''
            value = item.get('value', {})
            if field == 'personnel' and isinstance(value, dict):
                summary = [str(value['quantity']) + ' ' + str(value.get('role_mention', 'nhân sự'))]
                if value.get('workload_mention'):
                    summary.append(str(value['workload_mention']))
                detail = '<br><span class="report-small">Nhận diện: ' + _safe(' · '.join(summary)) + '</span>'
            values.append(_safe(item['quote']) + detail + ' · ' + reference)
        if len(entries) > 4:
            values.append('Các mục còn lại được liệt kê trong phụ lục quy tắc.')
        overview.append('<tr><td>' + label + '</td><td>' + ('<br>'.join(values)
                        or 'Chưa nhận diện theo quy tắc; cần đối chiếu nguồn') + '</td></tr>')
    overview.append('</tbody></table><p class="report-small">Đây là các đề cập trong tài liệu. '
                    'Nhân sự bán thời gian được giữ nguyên cách mô tả, không quy đổi thành người toàn thời gian. '
                    'Chưa xác nhận mức sẵn sàng, phân bổ ngân sách, ngày thực hiện hoặc nhân sự cho từng công việc.</p>')
    measures = facts.get('fields', {}).get('measures', [])
    if measures:
        overview.append('<h3>Chỉ tiêu và định lượng được đề cập trong nguồn</h3><table class="report-table"><thead>'
                        '<tr><th>Câu nguồn / điều kiện đo</th><th>Định lượng nhận diện</th><th>Trạng thái cần duyệt</th>'
                        '</tr></thead><tbody>')
        statuses = {'proposed_or_unconfirmed': 'Đề xuất / chưa xác nhận; chưa kiểm chứng',
                    'proposed_target': 'Mục tiêu dự kiến; chưa kiểm chứng',
                    'target': 'Mục tiêu dự kiến; chưa kiểm chứng',
                    'requirement': 'Ràng buộc / yêu cầu trong nguồn; chưa xác nhận đáp ứng',
                    'reported_result': 'Kết quả được nguồn thuật lại; chưa kiểm chứng độc lập',
                    'mention_only': 'Đề cập trong nguồn; cần xác định ý nghĩa và kiểm chứng',
                    'source_mention': 'Đề cập trong nguồn; cần xác định ý nghĩa'}
        comparators = {'at_least': '≥', 'at_most': '≤', 'less_than': '<', 'greater_than': '>',
                       'after': 'sau', 'exact_mention': ''}
        units = {'minute': 'phút', 'second': 'giây', 'hour': 'giờ', 'day': 'ngày', 'week': 'tuần',
                 'month': 'tháng', 'year': 'năm', 'percent': '%'}
        kinds = {'target': 'Chỉ tiêu trong nguồn', 'threshold': 'Ngưỡng trong nguồn',
                 'retention': 'Cửa sổ lưu giữ trong nguồn'}
        for item in measures:
            reference = _source_reference(item, digest, starts, source_text)
            value = item.get('value', {})
            context = value.get('context')
            quote = item['quote']
            if isinstance(context, dict):
                reference = _source_reference({**context, 'verified': True, 'source_sha256': digest},
                                              digest, starts, source_text)
                quote = context['quote']
            quantity = ' '.join(str(part) for part in (
                comparators.get(value.get('comparator'), value.get('comparator', '')),
                value.get('quantity', ''), units.get(value.get('unit'), value.get('unit', ''))) if part != '')
            kind = '<br>' + _safe(kinds.get(value.get('kind'), 'Định lượng được đề cập'))
            overview.append('<tr><td>' + _safe(quote) + '<br>' + reference + '</td><td>'
                            + _safe(quantity) + '</td><td>'
                            + statuses.get(value.get('status'), 'Định lượng trong nguồn; cần xác định điều kiện và kiểm chứng')
                            + kind + '</td></tr>')
        overview.append('</tbody></table><p class="report-small">Các giá trị là nội dung của tài liệu, '
                        'không phải phép đo do lượt phân tích này thực hiện. Cần xác nhận cách đo, mẫu số, '
                        'điều kiện và bằng chứng nghiệm thu trước khi kết luận đạt mục tiêu.</p>')
    overview.append('</section>')

    trace = ['<section id="traceability"><h2><span class="report-number">02 / TRUY VẾT</span>Ma trận công việc và bằng chứng</h2>',
             '<p>Đối chiếu nội dung của từng công việc với câu chữ trong tài liệu trước khi phân công.</p>',
             '<div class="report-table-wrap"><table class="report-table"><thead><tr><th>Mã</th>'
             '<th class="task-column">Công việc đề xuất</th><th>Bằng chứng nguồn</th><th>Nội dung kèm theo</th>'
             '</tr></thead><tbody>']
    for number, step in enumerate(steps, 1):
        task_id = step.get('task_id', f'S{number:03d}')
        attached = [f'{len(step.get("checklist", []))} mục kiểm tra',
                    f'{len(step.get("required_resources", []))} nguồn lực gắn kèm']
        if step.get('selection_origin') == 'source_catalog_preserved':
            attached.append('Code giữ từ catalog nguồn; chưa được model chọn')
        elif step.get('selection_origin') == 'llm_selection':
            attached.append('Có trong lựa chọn của model; nội dung từ nguồn')
        if step.get('parent_task_id'):
            attached.append('Công việc cha đề xuất: ' + str(step['parent_task_id']))
        trace.append('<tr><td><strong>' + _safe(task_id) + '</strong></td><td>' + _safe(step['description'])
                     + '<br><span class="report-small">Cần duyệt nội dung</span></td><td>'
                     + refs(step['evidence_ids']) + '</td><td>' + '<br>'.join(_safe(value) for value in attached)
                     + '</td></tr>')
    if not steps:
        trace.append('<tr><td colspan="4">Chưa có công việc đủ điều kiện để đưa vào đề xuất.</td></tr>')
    trace.append('</tbody></table></div><p class="report-small">Liên kết bằng chứng xác minh xuất xứ câu chữ. '
                 'Hệ thống chưa xác nhận ánh xạ yêu cầu → công việc; bảng này chưa đánh giá yêu cầu đã được đáp ứng. '
                 'Thứ tự liệt kê không tự xác lập quan hệ phụ thuộc hay lịch triển khai. '
                 'Khi chưa có mã công việc trong dữ liệu, S001… là mã hiển thị theo thứ tự, tương ứng với workspace.</p>')
    trace.append(_review_context_html(review_context, steps, digest, starts, source_text))
    trace.append('</section>')

    review_items = []
    labels = {
        'budgets_not_detected': ('Ngân sách cần xác nhận', 'Quy tắc chưa nhận diện ngân sách hoặc đơn vị tiền tệ. Kiểm tra nguồn và xác nhận mức kinh phí trước khi lập kế hoạch.'),
        'dates_not_detected': ('Thời hạn cần xác nhận', 'Quy tắc chưa nhận diện ngày lịch cụ thể. Đối chiếu tài liệu và xác nhận thời hạn với người phụ trách.'),
        'personnel_not_detected': ('Nhân sự cần xác nhận', 'Quy tắc chưa nhận diện số lượng nhân sự. Kiểm tra cách mô tả đội ngũ và xác nhận khả năng phân công.'),
        'budgets_mentions_need_review': ('Phân bổ ngân sách cần xác nhận', 'Nguồn có đề cập kinh phí. Chưa xác nhận cách phân bổ và mức đáp ứng cho từng công việc.'),
        'dates_mentions_need_review': ('Tiến độ cần xác nhận', 'Nguồn có đề cập ngày lịch. Cần xác nhận ý nghĩa từng mốc và lịch thực hiện của từng công việc.'),
        'personnel_mentions_need_review': ('Phân công nhân sự cần xác nhận', 'Nguồn có đề cập số lượng nhân sự. Chưa xác nhận mức sẵn sàng và trách nhiệm cho từng công việc.'),
        'technologies_mentions_need_review': ('Công nghệ cần thẩm định', 'Đã nhận diện tên công nghệ trong nguồn. Cần kiểm tra mức độ phù hợp, khả năng cung cấp và phân bổ cho từng công việc.'),
    }
    for warning in warnings:
        label, message = labels.get(warning.get('code'), ('Cảnh báo cần kiểm tra', warning['message']))
        review_items.append((label, message))
    empty = [FIELD_NAMES.get(field, field) for field in ('objectives', 'requirements', 'activities', 'expected_results')
             if not fields.get(field)]
    if empty:
        review_items.append(('Nhóm thông tin chưa nhận diện', ', '.join(empty) + '. Chưa nhận diện không có nghĩa tài liệu không chứa nội dung này.'))
    omitted = plan.get('catalog_omitted_counts', {})
    unselected = len(plan.get('unselected_task_ids', []))
    for message in _omission_messages(omitted):
        review_items.append(('Giới hạn lựa chọn cần rà soát', message))
    if unselected:
        review_items.append(('Phạm vi kế hoạch cần rà soát',
                             f'{unselected} công việc trong catalog chưa được chọn. Đối chiếu lại toàn bộ nguồn để tránh bỏ sót.'))
    if plan.get('dependency_candidates'):
        review_items.append(('Quan hệ phụ thuộc cần duyệt',
                             f'{len(plan["dependency_candidates"])} quan hệ do bộ lập kế hoạch đề xuất, chưa được người xác nhận.'))
    for warning in metadata.get('source', {}).get('warnings', []):
        review_items.append(('Lưu ý khi đọc tài liệu', str(warning)))
    if metadata.get('source', {}).get('ocr_applied'):
        review_items.append(('Văn bản OCR cần đối chiếu', 'Kiểm tra văn bản với ảnh trang gốc, đặc biệt số liệu, bảng và thứ tự đọc.'))
    review = ['<section id="review-register"><h2><span class="report-number">03 / RÀ SOÁT</span>Các điểm cần xác nhận trước triển khai</h2>',
              '<p>Các mục dưới đây được suy ra từ trạng thái nhận diện và kiểm tra của lượt chạy; '
              'chúng hỗ trợ người duyệt xác định việc cần kiểm tra tiếp.</p>']
    for label, message in review_items:
        review.append('<div class="report-review"><strong>' + _safe(label) + '</strong><p>' + _safe(message) + '</p></div>')
    if any(omitted.values()):
        review.append('<p class="report-small">Các bộ đếm giới hạn dùng đơn vị và phạm vi khác nhau, '
                      'có thể giao nhau; không cộng thành tổng nội dung bị bỏ sót. '
                      'Bằng chứng không vào prompt vẫn có thể hiện trong catalog hoặc phần rà soát nguồn.</p>')
    if unselected:
        catalog_tasks = {task['task_id']: task for task in plan.get('catalog', {}).get('tasks', [])}
        review.append('<h3>Công việc trong catalog chưa được chọn</h3><p class="report-small">'
                      'Chưa được chọn không đồng nghĩa không cần thiết. Kiểm tra từng công việc với nguồn.</p><ul>')
        for identity in plan['unselected_task_ids']:
            task = catalog_tasks.get(identity)
            if task:
                review.append('<li><strong>' + _safe(identity) + '</strong> — ' + _safe(task['description'])
                              + '<br>' + refs(task['evidence_ids']) + '</li>')
            else:
                review.append('<li><strong>' + _safe(identity) + '</strong> — '
                              'Dữ liệu kế hoạch cũ chưa kèm nội dung catalog; cần đối chiếu nguồn.</li>')
        review.append('</ul>')
    if review_context is not None and plan.get('catalog') is not None:
        bounded_ids = {task['task_id'] for task in plan['catalog'].get('tasks', [])}
        outside = [task for task in review_context.get('task_catalog', [])
                   if task['task_id'] not in bounded_ids]
        if outside:
            review.append('<h3>Công việc nguồn ngoài giới hạn catalog của model</h3>'
                          '<p class="report-small">Các công việc này vẫn có trong nguồn, chưa được gửi để model lựa chọn. '
                          'Người duyệt cần xem xét riêng; chính sách giữ task chỉ áp dụng trong catalog.</p><ul>')
            for task in outside:
                review.append('<li><strong>' + _safe(task['task_id']) + '</strong> — '
                              + _safe(task['description']) + '<br>' + refs(task['evidence_ids']) + '</li>')
            review.append('</ul>')
    if plan.get('dependency_candidates'):
        review.append('<h3>Quan hệ phụ thuộc do LLM đề xuất — chưa xác nhận</h3><ul>')
        for candidate in plan['dependency_candidates']:
            review.append('<li>' + _safe(candidate['before_task_id']) + ' → '
                          + _safe(candidate['after_task_id']) + ' · cần người duyệt; '
                          'khác với ràng buộc được trích nguyên văn ở phần truy vết. '
                          + refs(candidate.get('evidence_ids', [])) + '</li>')
        review.append('</ul>')
    if not review_items:
        review.append('<p>Chưa phát sinh cảnh báo theo các quy tắc hiện có. Cần tiếp tục duyệt tính đúng và đủ của nội dung.</p>')
    review.append('<div class="report-decision"><strong>Đề nghị xử lý</strong><p>1. Duyệt nhãn và nội dung công việc dựa trên nguồn.</p>'
                  '<p>2. Xác nhận phân công, thời hạn, ngân sách và tiêu chí nghiệm thu ở mức từng công việc.</p>'
                  '<p>3. Cập nhật trạng thái, ghi chú trong không gian tiến độ và xuất JSON để lưu kết quả rà soát.</p>'
                  '</div></section>')

    pipeline = [str(source_label), str(classifier), str(retriever), str(planner), 'Kiểm tra nguồn & cấu trúc', 'Báo cáo & tiến độ']
    stage_names = {'extract': 'Trích xuất / phân loại', 'retrieve': 'Truy xuất bằng chứng', 'plan': 'Lập kế hoạch'}
    stage_rows = []
    elapsed = []
    for stage in metadata.get('stages', []):
        seconds = stage.get('elapsed_seconds')
        recorded = isinstance(seconds, (float, int)) and not isinstance(seconds, bool)
        if recorded:
            elapsed.append(seconds)
        status = 'Hoàn tất' if stage.get('returncode') == 0 else 'Chưa ghi nhận hoàn tất'
        stage_rows.append('<tr><td>' + _safe(stage_names.get(stage.get('stage'), stage.get('stage', '')))
                          + '</td><td>' + (f'{seconds:.2f} giây' if recorded else 'Chưa ghi nhận')
                          + '</td><td>' + status + '</td></tr>')
    method = ['<section id="run-method"><h2><span class="report-number">04 / PHƯƠNG PHÁP</span>Cấu hình và kiểm chứng của lượt chạy</h2>',
              '<div class="report-pipeline">' + ''.join('<span>' + _safe(item) + '</span>' for item in pipeline) + '</div>']
    if plan.get('provenance', {}).get('model_loaded') is False:
        method.append('<p class="notice">Lượt này không nạp LLM do không có công việc đủ điều kiện trong catalog.</p>')
    if stage_rows:
        method.append('<table class="report-table"><thead><tr><th>Giai đoạn</th><th>Thời gian worker</th>'
                      '<th>Trạng thái</th></tr></thead><tbody>' + ''.join(stage_rows) + '</tbody></table>')
        if elapsed:
            method.append(f'<p class="report-small">Tổng thời gian worker đã ghi nhận: {sum(elapsed):.2f} giây. '
                          'Không bao gồm toàn bộ thời gian đọc nguồn, xuất báo cáo hoặc thao tác của người dùng.</p>')
    human_status = ('Đã ghi nhận duyệt nội dung' if metadata.get('human_content_review_complete') is True
                    else 'Chưa hoàn tất duyệt nội dung bởi người')
    test_status = ('Đã ghi nhận chạy final test; xem hồ sơ đánh giá riêng' if metadata.get('final_test_evaluated') is True
                   else 'Chưa đánh giá final test độc lập trong lượt chạy này')
    method.append('<table class="report-table"><thead><tr><th>Hạng mục kiểm chứng</th><th>Kết quả / phạm vi</th></tr></thead>'
                  '<tbody><tr><td>Xuất xứ trích dẫn</td><td>Đã đối chiếu hash và vị trí ký tự với văn bản nhúng trong báo cáo.</td></tr>'
                  '<tr><td>Duyệt nội dung</td><td>' + human_status + '</td></tr><tr><td>Đánh giá độc lập</td><td>' + test_status
                  + '</td></tr></tbody></table><p class="notice">Kiểm tra nguồn và cấu trúc xác nhận khả năng truy vết. '
                  'Độ đúng nhãn, mức độ liên quan, tính đầy đủ của công việc và chất lượng kế hoạch cần được đánh giá riêng.</p>'
                  '<p class="report-small">SHA-256 văn bản phân tích: <code>' + _safe(digest) + '</code></p></section>')

    # Fixed markers belong to our own portable renderer, never to source text.
    before, marker, remaining = rendered.partition('<header>')
    _, closing, after = remaining.partition('</header>')
    if not marker or not closing or '<main id="content">' not in after:
        raise ValueError('Portable report template is missing its presentation boundary')
    rendered = before + cover + after
    sections = ''.join(overview + trace + review + method)
    rendered = rendered.replace('<main id="content">', '<main id="content">' + sections
                                + '<section id="extraction-details"><h2><span class="report-number">05 / KẾT QUẢ CHI TIẾT</span>'
                                'Nội dung trích xuất và phụ lục đối chiếu</h2><p>Các nhóm dưới đây giữ nguyên câu chữ '
                                'và liên kết về văn bản nguồn để phục vụ kiểm tra chi tiết.</p></section>', 1)
    return rendered.replace('</style>', '</style><style>' + STYLE + '</style>', 1)
