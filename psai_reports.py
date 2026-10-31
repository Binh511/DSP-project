"""Portable, escaped HTML reports with embedded exact-source evidence.

No browser, network, JavaScript dependency, or model is needed to render a report.
Verification here concerns source fidelity; it does not certify semantic labels.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import re


FIELD_NAMES = {
    'objectives': 'Mục tiêu', 'requirements': 'Yêu cầu', 'activities': 'Hoạt động',
    'expected_results': 'Kết quả mong đợi', 'deliverables': 'Sản phẩm bàn giao',
    'technologies': 'Công nghệ', 'resources': 'Nguồn lực',
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _safe(value) -> str:
    return escape(str(value), quote=True)


def _line_index(text: str):
    starts, cursor = [0], 0
    for line in text.splitlines(keepends=True):
        cursor += len(line)
        starts.append(cursor)
    return starts


def _check_span(item: dict, text: str) -> None:
    start, end, quote = item.get('char_start'), item.get('char_end'), item.get('quote')
    if (item.get('verified') is not True or type(start) is not int or type(end) is not int
            or not isinstance(quote, str) or not quote.strip()
            or not 0 <= start < end <= len(text) or text[start:end] != quote):
        raise ValueError('HTML report rejected an invalid or unverified source quotation')


def render_report_html(record: dict, plan: dict, source_text: str, metadata: dict,
                       retrieval: dict | None = None) -> str:
    """Render a self-contained report; reject altered source spans and hashes."""
    from planning import _evidence_index, validate_plan

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError('HTML report needs nonempty exact source text')
    main_hash = _hash(source_text)
    if record.get('source_sha256') != main_hash:
        raise ValueError('HTML report source hash differs from extraction')
    if metadata.get('source_text_sha256', main_hash) != main_hash:
        raise ValueError('HTML report source hash differs from run manifest')
    evidence = _evidence_index(record, source_text)
    validate_plan(plan, set(evidence), evidence_by_id=evidence)
    retrieval = retrieval or {}
    texts = {main_hash: source_text, **retrieval.get('source_texts', {})}
    source_metadata = retrieval.get('sources', {})
    for identity, text in texts.items():
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', identity):
            raise ValueError('HTML report source snapshot IDs must be safe nonempty anchor identifiers')
        expected = source_metadata.get(identity, {}).get('source_sha256', identity)
        if not isinstance(text, str) or _hash(text) != expected:
            raise ValueError('HTML report received an invalid source snapshot hash')
    line_starts = {digest: _line_index(text) for digest, text in texts.items()}

    def citation(item, digest=main_hash):
        if digest not in texts:
            raise ValueError('HTML report is missing a cited source snapshot')
        _check_span(item, texts[digest])
        if item.get('source_sha256', _hash(texts[digest])) != _hash(texts[digest]):
            raise ValueError('HTML citation refers to a different source hash')
        line = bisect_right(line_starts[digest], item['char_start'])
        anchor = f'src-{digest}-L{line}'
        return (f'<a class="citation" href="#{_safe(anchor)}">'
                f'{_safe(item.get("evidence_id", "source"))} · dòng {line} · '
                f'ký tự [{item["char_start"]}, {item["char_end"]})</a>')

    cards = []
    for field, items in record.get('fields', {}).items():
        pieces = [f'<section><h2>{_safe(FIELD_NAMES.get(field, field))}</h2>']
        if not items:
            pieces.append('<p class="muted">Chưa tìm thấy nội dung đủ điều kiện.</p>')
        for item in items:
            link = citation(item)
            pieces += ['<article class="evidence">', f'<blockquote>{_safe(item["quote"])}</blockquote>',
                       f'<p>{link}</p>', '</article>']
        pieces.append('</section>')
        cards.append(''.join(pieces))

    tasks = []
    for step in plan.get('steps', []):
        refs = ' '.join(citation(evidence[eid]) for eid in step['evidence_ids'])
        tasks.append(f'<li><p>{_safe(step["description"])}</p><div>{refs}</div></li>')
    plan_html = ('<ol class="tasks">' + ''.join(tasks) + '</ol>') if tasks else '<p>Chưa có bước đủ điều kiện từ nguồn.</p>'

    hits_html = []
    for hit in retrieval.get('hits', []):
        digest = hit.get('snapshot_id', hit.get('source_sha256'))
        link = citation(hit, digest)
        role = 'Tài nguyên tham khảo' if hit.get('role') == 'resource' else 'Tài liệu dự án'
        hits_html.append('<article class="evidence">' + f'<p class="tag">{role}</p>'
                         + f'<blockquote>{_safe(hit["quote"])}</blockquote><p>{link}</p></article>')
    retrieval_html = ''
    if retrieval:
        query_rows = ''.join('<li>' + _safe(query) + '</li>' for query in retrieval.get('queries', []))
        retrieval_html = ('<section><h2>Truy xuất bằng chứng</h2>'
                          f'<p>Phương pháp: <strong>{_safe(retrieval.get("mode", "unknown"))}</strong>.</p>'
                          f'<details><summary>Truy vấn đã dùng</summary><ul>{query_rows}</ul></details>'
                          + (''.join(hits_html) or '<p>Không có đoạn phù hợp đạt điều kiện truy xuất.</p>')
                          + '<p class="muted">Điểm truy xuất là điểm xếp hạng, không phải xác suất đúng. '
                            'Tài nguyên tham khảo không tự trở thành yêu cầu của dự án.</p></section>')

    source_blocks = []
    for digest, text in texts.items():
        info = source_metadata.get(digest, {})
        title = 'Nguồn tài liệu dự án' if digest == main_hash else 'Nguồn tài nguyên tham khảo'
        origin = info.get('origin_path', '')
        lines = text.splitlines(keepends=True) or ['']
        rendered = ''.join(f'<span class="source-line" id="src-{_safe(digest)}-L{number}">'
                           f'<a class="line-number" href="#src-{_safe(digest)}-L{number}">{number}</a>'
                           f'<span>{_safe(line.rstrip(chr(10)).rstrip(chr(13)))}</span></span>'
                           for number, line in enumerate(lines, 1))
        source_blocks.append(f'<section class="source"><h2>{title}</h2>'
                             f'<p class="muted">{_safe(origin)}<br>SHA-256: <code>{_safe(_hash(text))}</code></p>'
                             f'<div class="source-text">{rendered}</div></section>')

    mode = metadata.get('classifier', 'unknown')
    classifier_name = {'rules': 'Quy tắc — demo chức năng',
                       'ensemble': 'Ensemble classifier hiện hành'}.get(mode, mode)
    generated = metadata.get('created_utc', datetime.now(timezone.utc).isoformat())
    source_method = metadata.get('source', {}).get('method', 'local_text')
    warnings = metadata.get('source', {}).get('warnings', [])
    warning_html = ''.join('<li>' + _safe(value) + '</li>' for value in warnings)
    provenance = {key: metadata.get(key) for key in ('input_sha256', 'source_text_sha256', 'code_sha256',
                  'models', 'retrieval', 'human_content_review_complete', 'pipeline_quality_score') if key in metadata}
    return f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Báo cáo PSAI — {_safe(record.get('document_id', 'project'))}</title>
<style>
:root{{font-family:Segoe UI,Arial,sans-serif;color:#182735;background:#edf2f6;line-height:1.55}}
body{{max-width:1100px;margin:auto;padding:28px 20px}} header,section{{background:white;border:1px solid #dbe4ec;border-radius:12px;padding:24px;margin:18px 0}}
header{{border-top:6px solid #165b7b}} h1{{font-size:28px;margin:0 0 10px}} h2{{font-size:21px;margin:0 0 14px}} h3{{font-size:17px}}
a{{color:#075b93;overflow-wrap:anywhere}} .muted{{color:#526577;font-size:14px}} .notice{{border-left:4px solid #bd8217;background:#fff8e8;padding:12px 16px}}
.evidence{{border-top:1px solid #e6edf2;padding:12px 0}} blockquote{{white-space:pre-wrap;margin:8px 0;padding:10px 16px;border-left:3px solid #277d9a;background:#f5f9fb}}
.citation{{font-size:13px}} .tag{{font-size:12px;font-weight:600;color:#165b7b}} .tasks li{{padding:8px 0}} code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}
.source-text{{font:13px/1.6 Consolas,monospace;overflow:auto;border:1px solid #dbe4ec;padding:10px 0;background:#fafcfd}}
.source-line{{display:flex;white-space:pre-wrap;overflow-wrap:anywhere;padding:1px 10px;scroll-margin:40px}} .source-line:target{{background:#fff0a3;outline:2px solid #e8b847}}
.line-number{{display:inline-block;min-width:45px;color:#64798b;user-select:none;text-decoration:none}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}} summary{{cursor:pointer;color:#075b93}}
nav{{display:flex;gap:16px;flex-wrap:wrap}} .footer{{font-size:12px;color:#526577}}
@media print{{body{{max-width:none;background:white;padding:0}}section,header{{break-inside:auto;border:1px solid #bbb;padding:14px}}.source-line:target{{outline:none}}a{{color:inherit}}}}
</style></head><body>
<header><h1>Báo cáo phân tích tài liệu PSAI</h1><p><strong>{_safe(record.get('document_id', ''))}</strong> · {_safe(classifier_name)}</p>
<p class="muted">Tạo lúc {_safe(generated)} · Đọc nguồn bằng {_safe(source_method)}</p>
<nav><a href="#content">Nội dung chính</a><a href="#plan">Các bước theo nguồn</a><a href="#sources">Văn bản nguồn</a><a href="#provenance">Thông tin tái lập</a></nav>
<p class="notice">Các trích dẫn đã được đối chiếu hash và vị trí ký tự với văn bản nguồn.
Điều này xác nhận nội dung trích dẫn, chưa xác nhận nhãn phân loại, tính đầy đủ hoặc chất lượng kế hoạch.
Đây là kế hoạch lấy câu từ nguồn; không tự tạo cam kết hay phụ thuộc chưa có bằng chứng.</p>
<ul>{warning_html}</ul></header>
<main id="content">{''.join(cards)}{retrieval_html}
<section id="plan"><h2>Các bước lấy trực tiếp từ nguồn</h2>{plan_html}</section>
<div id="sources">{''.join(source_blocks)}</div>
<section id="provenance"><h2>Thông tin tái lập</h2><pre>{_safe(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False))}</pre></section></main>
<p class="footer">Báo cáo chứa văn bản nguồn ngay trong file, mở được khi không có project gốc hoặc kết nối mạng.
Vị trí ký tự tính trên văn bản đã trích; không phải tọa độ trên PDF.</p></body></html>'''
