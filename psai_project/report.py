"""Export a complete local project report while retaining unreviewed status."""
from __future__ import annotations

from bisect import bisect_right
import base64
from html import escape
import hashlib
import json
from pathlib import Path


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _validate_local_plan(plan, record, source_text, retrieval, snapshots):
    """Recheck the public export boundary, including every nested LLM claim."""
    if plan.get('planning_method') != 'local_llm_evidence_catalog_v1':
        return
    from psai_planner.planner import build_catalog, build_prompt, validate_response
    provenance = plan.get('provenance')
    if not isinstance(provenance, dict) or not isinstance(provenance.get('config'), dict):
        raise ValueError('Local plan export requires its exact planner protocol provenance')
    catalog = build_catalog(record, source_text, {**retrieval, 'source_texts': snapshots}, provenance['config'])
    messages = build_prompt(catalog)
    def digest(value):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()
    if (plan.get('catalog') != catalog or plan.get('messages') != messages
            or provenance.get('catalog_sha256') != digest(catalog)
            or provenance.get('prompt_sha256') != digest(messages)):
        raise ValueError('Local plan catalog/prompt differs from the current verified source')
    raw = plan.get('raw_response')
    empty_without_model = raw is None and not catalog['tasks'] and provenance.get('model_loaded') is False
    if empty_without_model:
        raw = json.dumps({'tasks': [], 'dependency_candidates': [], 'resource_references': []})
    elif not isinstance(raw, str) or provenance.get('model_loaded') is not True:
        raise ValueError('Local plan must retain its exact model response and execution state')
    expected = validate_response(raw, catalog)
    if empty_without_model:
        expected['warnings'].append('No eligible task catalog; the LLM was not loaded')
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Local plan was altered after schema/source validation')
    allowed = set(expected) | {'raw_response', 'messages', 'catalog', 'provenance', 'related_context'}
    if set(plan) - allowed:
        raise ValueError('Local plan export contains unsupported fields')


def _planner_html(plan, record, source_text):
    if plan.get('planning_method') != 'local_llm_evidence_catalog_v1':
        return ''
    evidence = {item['evidence_id']: item for items in record['fields'].values() for item in items}
    starts, cursor = [0], 0
    for line in source_text.splitlines(keepends=True):
        cursor += len(line); starts.append(cursor)
    digest = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
    def ref(ids):
        result = []
        for identity in ids:
            item = evidence[identity]
            line = bisect_right(starts, item['char_start'])
            result.append(f'<a href="#src-{digest}-L{line}">{escape(identity)}</a>')
        return ' '.join(result)
    parts = ['<section id="llm-plan"><h2>LLM lập kế hoạch cục bộ</h2>',
             '<p class="notice">LLM chọn và sắp xếp các công việc có bằng chứng. Kiểm tra schema/hash/span '
             'chưa chứng minh nhãn, quan hệ phụ thuộc hoặc kế hoạch đã đủ; cần người duyệt trước sử dụng.</p>',
             f'<p>Trạng thái: {escape(plan["status"])}; mô hình đã nạp: {plan["provenance"]["model_loaded"]}.</p>']
    for step in plan['steps']:
        parts += [f'<article class="evidence"><h3>{escape(step["task_id"])} — {escape(step["description"])}</h3>',
                  '<p>' + ref(step['evidence_ids']) + '</p>']
        if step.get('parent_task_id'):
            parts += ['<p>Công việc cha được đề xuất: ' + escape(step['parent_task_id']) + '</p>']
        for title, entries in [('Checklist trích từ nguồn', step['checklist']),
                               ('Tài nguyên được đề xuất gắn vào công việc', step['required_resources'])]:
            if entries:
                parts += ['<h4>' + title + '</h4><ul>']
                for item in entries:
                    parts += ['<li>' + escape(item['description']) + ' · ' + ref(item['evidence_ids']) + '</li>']
                parts += ['</ul>']
        parts += ['</article>']
    if plan['dependency_candidates']:
        parts += ['<h3>Ứng viên phụ thuộc — chưa xác nhận</h3><ul>']
        for candidate in plan['dependency_candidates']:
            parts += ['<li>' + escape(candidate['before_task_id']) + ' → ' + escape(candidate['after_task_id'])
                      + ' · gợi ý của LLM, cần người kiểm tra</li>']
        parts += ['</ul>']
    if plan['resource_references']:
        parts += ['<p>Tài liệu bên ngoài chỉ được dùng làm tham khảo, không tự tạo yêu cầu dự án.</p>']
    for warning in plan.get('warnings', []):
        parts += ['<p class="muted">' + escape(warning) + '</p>']
    parts += ['</section>']
    return ''.join(parts)


def _facts_html(facts, warnings, text):
    digest = facts['source_sha256']
    starts, cursor = [0], 0
    for line in text.splitlines(keepends=True):
        cursor += len(line); starts.append(cursor)
    parts = ['<section id="rules"><h2>Thông tin theo quy tắc và cảnh báo nguồn lực</h2>',
             '<p>Quy tắc chỉ nhận diện mẫu xuất hiện trong nguồn. Chuẩn hóa còn mơ hồ được giữ để người kiểm tra; '
             'không tự suy ra lịch, ngân sách hoặc phân công.</p>']
    names = {'dates': 'Ngày', 'identifiers': 'Mã định danh', 'milestones': 'Mốc', 'durations': 'Thời lượng',
             'budgets': 'Ngân sách', 'technologies': 'Công nghệ', 'personnel': 'Nhân sự'}
    for field, values in facts['fields'].items():
        if not values: continue
        parts.append('<h3>' + names.get(field, escape(field)) + '</h3><ul>')
        for item in values:
            if (text[item['char_start']:item['char_end']] != item['quote']
                    or item['source_sha256'] != digest):
                raise ValueError('Rule fact differs from its exact source')
            line = bisect_right(starts, item['char_start'])
            value = json.dumps(item['value'], ensure_ascii=False) if isinstance(item['value'], dict) else str(item['value'])
            parts.append('<li><strong>' + escape(item['quote']) + '</strong> · ' + escape(value)
                + (' · cần kiểm tra' if item['needs_review'] else '')
                + f' · <a href="#src-{digest}-L{line}">dòng {line}</a></li>')
        parts.append('</ul>')
    for warning in warnings:
        parts.append('<p class="notice">' + escape(warning['message']) + '</p>')
    parts.append('</section>')
    return ''.join(parts)


def _ocr_images_html(stage, source_metadata):
    parts = []
    for page in source_metadata.get('pages', []):
        if not page.get('image_path'): continue
        path = (stage / page['image_path']).resolve()
        if not path.is_relative_to(stage.resolve()) or not path.is_file():
            raise ValueError('OCR evidence image is missing or outside the report directory')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != page.get('image_sha256'):
            raise ValueError('OCR evidence image differs from its recorded source hash')
        if path.stat().st_size > 12 * 1024 * 1024:
            parts.append('<p>Ảnh trang OCR quá lớn để nhúng; xem ảnh nguồn trong thư mục page_images.</p>')
            continue
        encoded = base64.b64encode(raw).decode('ascii')
        parts.append('<section><h2>Ảnh nguồn OCR — trang ' + escape(str(page.get('page', '')))
            + '</h2><p>Ảnh để đối chiếu bằng mắt; văn bản OCR chưa được người kiểm định.</p>'
            + '<img style="max-width:100%;height:auto" alt="Ảnh nguồn trang OCR" src="data:image/png;base64,'
            + encoded + '"></section>')
    return ''.join(parts)


def export_project_report(stage, source_text, source_metadata, extraction_response,
                          retrieval_response, plan_response, metadata):
    from psai_reports import render_report_html
    from planning import _evidence_index, validate_plan
    from psai_rules import extract_facts, resource_warnings
    from .workspace import render_workspace, initial_progress

    stage = Path(stage)
    record = extraction_response['record']
    retrieved = retrieval_response['retrieval']
    snapshots = retrieval_response['source_texts']
    plan, source_plan = plan_response['plan'], plan_response['source_plan']
    _validate_local_plan(plan, record, source_text, retrieved, snapshots)
    evidence = _evidence_index(record, source_text)
    validate_plan(plan, set(evidence), evidence_by_id=evidence)
    record['related_context'] = retrieved['hits']
    plan['related_context'] = retrieved['hits']
    metadata['source'] = source_metadata
    starts, cursor = [0], 0
    for line in source_text.splitlines(keepends=True):
        cursor += len(line); starts.append(cursor)
    metadata['source_line_starts'] = starts
    metadata['planner'] = {'method': plan['planning_method'], 'status': plan.get('status', 'source_items'),
                           'human_review_complete': False, 'provenance': plan.get('provenance')}
    rendered = render_report_html(record, plan, source_text, metadata,
                                  retrieval={**retrieved, 'source_texts': snapshots})
    extra = _planner_html(plan, record, source_text)
    facts = extract_facts(record['document_id'], source_text)
    warnings = resource_warnings(record, plan, facts)
    extra += _facts_html(facts, warnings, source_text)
    extra += _ocr_images_html(stage, source_metadata)
    rendered = rendered.replace('<div id="sources">', extra + '<div id="sources">', 1)
    rendered = rendered.replace('<nav>', '<nav><a href="workspace.html">Không gian dự án / tiến độ</a>', 1)
    for identity, info in retrieved['sources'].items():
        artifact = (stage / info['artifact']).resolve()
        if not artifact.is_relative_to(stage.resolve()):
            raise ValueError('Source snapshot path escapes project report directory')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        text = snapshots[identity]
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != info['source_sha256']:
            raise ValueError('Project report source snapshot hash differs')
        artifact.write_text(text, encoding='utf-8', newline='')
    (stage / 'source.txt').write_text(source_text, encoding='utf-8', newline='')
    (stage / 'report.html').write_text(rendered, encoding='utf-8')
    analysis = {'schema_version': 2, 'document_id': record['document_id'], 'source': source_metadata,
                'extraction': record, 'retrieval': retrieved, 'plan': plan,
                'source_plan': source_plan, 'rule_facts': facts,
                'resource_matching': {'method': 'source_mentions_and_review_warnings', 'warnings': warnings},
                'metadata': metadata}
    _write(stage / 'analysis.json', analysis)
    _write(stage / 'extraction.json', record)
    _write(stage / 'retrieval.json', retrieved)
    _write(stage / 'plan.json', plan)
    _write(stage / 'source_plan.json', source_plan)
    _write(stage / 'manifest.json', metadata)
    _write(stage / 'rule_facts.json', facts)
    _write(stage / 'progress_initial.json', initial_progress(analysis))
    (stage / 'workspace.html').write_text(render_workspace(analysis), encoding='utf-8')
    _write(stage / 'classifier_predictions.json', extraction_response['classifier_predictions'])
    (stage / 'report.md').write_text('# Báo cáo PSAI\n\n'
        'Mở [báo cáo HTML có nguồn và kế hoạch](report.html).\n\n'
        'Kết quả là đề xuất cần người duyệt. Kiểm tra hash/span xác minh nguồn câu chữ, '
        'không tự chứng minh độ đúng nhãn, tính đầy đủ hoặc độ đúng quan hệ phụ thuộc.\n', encoding='utf-8')
    return stage / 'report.html'
