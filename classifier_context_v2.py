"""Prepare an auditable Section-only dataset revision; never train or infer.

The frozen original-label study remains unchanged. Ambiguous evidence retains
the old Section and is exported for human review. No test/final-test is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'reports/research_review/studies/user_labels_v1/experiments/baseline_original_labels'
DEFAULT_OUTPUT = ROOT / 'reports/research_review/studies/classifier_context_v2_1'
VERSION = 'classifier-context-v2.1-section-from-cleaned-markdown-fence-aware-atx'
NORMALIZATION = 'unescape-pipe-underscore-backtick-star/remove-backticks/casefold/collapse-whitespace-v1'
REQUIRED = ['document_id', 'text', 'label', 'section', 'prev_text', 'pos', 'next_text']
HEADING = re.compile(r'^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$')
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def normalize(value: str) -> str:
    value = re.sub(r'\\([|_`*])', r'\1', str(value))
    value = re.sub(r'`+', '', value)
    return ' '.join(value.split()).casefold()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_hash(frame: pd.DataFrame) -> str:
    """Canonical UTF-8 JSON of ordered columns and ordered string-valued rows."""
    payload = {'columns': list(frame.columns), 'rows': frame.astype(str).values.tolist()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def read_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if list(frame.columns) != REQUIRED:
        raise ValueError(f'Unexpected columns in {path}: {list(frame.columns)}')
    if frame.empty or frame.document_id.str.strip().eq('').any():
        raise ValueError(f'Empty data or document IDs: {path}')
    return frame


def parse_source(text: str) -> dict:
    """Retain actual heading levels, including jumps such as # -> ####."""
    stack, fragments, spans, cursor, fence = [], [], [], 0, None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        inside_fence = fence is not None
        if inside_fence:
            if (marker and marker.group(1)[0] == fence[0]
                    and len(marker.group(1)) >= fence[1] and not marker.group(2).strip()):
                fence = None
        elif marker and not (marker.group(1)[0] == '`' and '`' in marker.group(2)):
            fence = (marker.group(1)[0], len(marker.group(1)))
        match = HEADING.match(line) if not inside_fence and fence is None else None
        if match:
            level = len(match.group(1))
            title = re.sub(r'\s+#+\s*$', '', match.group(2).strip()).strip()
            stack = [(old_level, value, old_line) for old_level, value, old_line in stack if old_level < level]
            stack.append((level, title, line_number))
        fragment = normalize(line)
        if fragment:
            spans.append({'start': cursor, 'end': cursor + len(fragment), 'line': line_number,
                          'section': ' > '.join(value for _, value, _ in stack),
                          'heading_lines': [old_line for _, _, old_line in stack]})
            fragments.append(fragment)
            cursor += len(fragment) + 1
    return {'normalized': ' '.join(fragments), 'spans': spans}


def find_matches(source: dict, text: str) -> list[dict]:
    query = normalize(text)
    if not query:
        return []
    cursor, result = 0, []
    while True:
        start = source['normalized'].find(query, cursor)
        if start < 0:
            return result
        end = start + len(query)
        overlaps = [span for span in source['spans'] if span['end'] > start and span['start'] < end]
        sections = sorted(set(span['section'] for span in overlaps))
        result.append({'normalized_start': start, 'normalized_end': end,
                       'source_start_line': overlaps[0]['line'], 'source_end_line': overlaps[-1]['line'],
                       'section_paths': sections,
                       'heading_lines': sorted(set(number for span in overlaps for number in span['heading_lines']))})
        cursor = start + 1


def load_document(cleaned_root: Path, document_id: str) -> tuple[list[dict], list[dict]]:
    # Document identifiers are IDs, never arbitrary filesystem paths.
    if not document_id or any(char in document_id for char in ('/', '\\', ':')) or document_id in {'.', '..'}:
        raise ValueError(f'Unsafe document_id: {document_id!r}')
    folder = cleaned_root / document_id
    paths = sorted(set(folder.glob('*.md')) | set(folder.glob('*.txt')))
    sources, evidence = [], []
    for path in paths:
        entry = {'path': str(path.resolve()), 'sha256': sha(path), 'status': 'readable_utf8'}
        try:
            parsed = parse_source(path.read_text(encoding='utf-8-sig'))
            sources.append({'path': str(path.resolve()), 'parsed': parsed})
        except UnicodeError as error:
            entry['status'] = 'unreadable_encoding'
            entry['error_type'] = type(error).__name__
        evidence.append(entry)
    return sources, evidence


def resolve_section(sources: list[dict], evidence: list[dict], text: str) -> dict:
    if not evidence:
        return {'status': 'missing_source', 'resolved': False, 'candidate_section': '', 'matches': []}
    matches = []
    for source in sources:
        matches.extend({'source_path': source['path'], **match} for match in find_matches(source['parsed'], text))
    result = {'status': '', 'resolved': False, 'candidate_section': '', 'matches': matches}
    if any(entry['status'] != 'readable_utf8' for entry in evidence):
        result['status'] = 'source_unreadable_needs_review'
    elif not matches:
        result['status'] = 'no_normalized_match_needs_review'
    elif len({match['source_path'] for match in matches}) > 1:
        result['status'] = 'multiple_matching_sources_needs_review'
    else:
        sections = {section for match in matches for section in match['section_paths']}
        if len(sections) > 1:
            result['status'] = 'multiple_section_paths_needs_review'
        elif sections == {''}:
            result['status'] = 'before_first_heading_needs_review'
        elif len(sections) == 1:
            result.update(status='resolved_unique_heading_path', resolved=True, candidate_section=next(iter(sections)))
        else:
            result['status'] = 'no_heading_path_needs_review'
    return result


def revise_rows(frame: pd.DataFrame, cleaned_root: Path, split: str,
                document_cache: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    revised, audit = frame.copy(deep=True), []
    cache = {} if document_cache is None else document_cache
    for index, row in frame.iterrows():
        if row.document_id not in cache:
            cache[row.document_id] = load_document(cleaned_root, row.document_id)
        sources, source_evidence = cache[row.document_id]
        result = resolve_section(sources, source_evidence, row.text)
        if result['resolved']:
            revised.at[index, 'section'] = result['candidate_section']
        new_section = revised.at[index, 'section']
        audit.append({'split': split, 'row_index': int(index), 'document_id': row.document_id, 'text': row.text,
                      'reference_label_unchanged': row.label,
                      'old_section': row.section, 'new_section': new_section,
                      'resolved': result['resolved'], 'changed': row.section != new_section,
                      'status': result['status'],
                      'matches': json.dumps(result['matches'], ensure_ascii=False, separators=(',', ':')),
                      'source_paths': json.dumps([entry['path'] for entry in source_evidence], ensure_ascii=False),
                      'human_verified': False, 'pdf_verified': False,
                      'reviewer': '', 'review_decision': '', 'review_note': ''})
    pd.testing.assert_frame_equal(frame.drop(columns=['section']), revised.drop(columns=['section']))
    manifest = [entry for _, evidence in cache.values() for entry in evidence]
    return revised, pd.DataFrame(audit), manifest


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(training_path: Path, validation_path: Path, cleaned_root: Path, output: Path,
            previous_version: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing version: {output}')
    training, validation = read_rows(training_path), read_rows(validation_path)
    if set(training.document_id) & set(validation.document_id):
        raise ValueError('Training/validation document overlap')
    cache, prepared, audits, split_summaries, previous_comparisons, previous_differences = {}, {}, [], {}, {}, []
    for split, rows, path in [('train', training, training_path), ('validation', validation, validation_path)]:
        new_rows, audit, source_manifest = revise_rows(rows, cleaned_root, split, cache)
        prepared[split] = new_rows
        audits.append(audit)
        split_summaries[split] = {
            'rows': len(rows), 'documents': int(rows.document_id.nunique()),
            'original_input_file': str(path.resolve()), 'original_input_file_sha256': sha(path),
            'original_ordered_row_sha256': row_hash(rows), 'revised_ordered_row_sha256': row_hash(new_rows),
            'unchanged_nonsection_row_sha256': row_hash(rows.drop(columns=['section'])),
            'labels_order_docs_text_and_other_features_unchanged': True,
            'section_blank_before': int(rows.section.str.strip().eq('').sum()),
            'section_blank_after': int(new_rows.section.str.strip().eq('').sum()),
            'section_changed': int(audit.changed.sum()),
            'blank_section_recovered': int((audit.changed & audit.old_section.str.strip().eq('')).sum()),
            'existing_nonblank_section_corrected': int((audit.changed & audit.old_section.str.strip().ne('')).sum()),
            'resolved': int(audit.resolved.sum()), 'unresolved_kept_old_section': int((~audit.resolved).sum()),
            'status_counts': {key: int(value) for key, value in Counter(audit.status).items()},
        }
        if previous_version is not None:
            previous_path = previous_version / ('training_rows.csv' if split == 'train' else 'validation_rows.csv')
            previous_rows = read_rows(previous_path)
            pd.testing.assert_frame_equal(rows.drop(columns='section'), previous_rows.drop(columns='section'))
            difference = previous_rows.section.ne(new_rows.section)
            for index in new_rows.index[difference]:
                previous_differences.append({'split': split, 'row_index': int(index),
                                             'document_id': new_rows.at[index, 'document_id'],
                                             'text': new_rows.at[index, 'text'],
                                             'previous_section': previous_rows.at[index, 'section'],
                                             'new_section': new_rows.at[index, 'section'],
                                             'new_resolution_status': audit.at[index, 'status']})
            previous_comparisons[split] = {'previous_file': str(previous_path.resolve()),
                                          'previous_file_sha256': sha(previous_path),
                                          'previous_ordered_row_sha256': row_hash(previous_rows),
                                          'section_differences': int(difference.sum()),
                                          'nonsection_columns_identical': True}
    # Creating the version directory is exclusive and happens after in-memory checks.
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in prepared.items():
        name = 'training_rows.csv' if split == 'train' else 'validation_rows.csv'
        rows.to_csv(output / name, index=False, encoding='utf-8-sig')
        split_summaries[split]['prepared_file'] = name
        split_summaries[split]['prepared_file_sha256'] = sha(output / name)
        pd.testing.assert_frame_equal(rows, read_rows(output / name))
    trace = pd.concat(audits, ignore_index=True)
    trace.to_csv(output / 'section_changes_and_provenance.csv', index=False, encoding='utf-8-sig')
    trace[~trace.resolved].to_csv(output / 'section_unresolved_for_review.csv', index=False, encoding='utf-8-sig')
    if previous_version is not None:
        pd.DataFrame(previous_differences, columns=['split', 'row_index', 'document_id', 'text', 'previous_section', 'new_section',
                                                  'new_resolution_status']).to_csv(output / 'differences_from_v2.csv', index=False, encoding='utf-8-sig')
    documents = []
    for document_id, (sources, evidence) in cache.items():
        documents.append({'document_id': document_id, 'source_count': len(evidence),
                          'sources': evidence, 'parsed_sources': len(sources)})
    snapshot_path = output / 'source_snapshot' / 'classifier_context_v2.py'
    snapshot_path.parent.mkdir()
    snapshot_path.write_bytes(Path(__file__).read_bytes())
    manifest = {
        'version': VERSION, 'status': 'preparation_only_not_train_evaluated',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'splits': split_summaries,
        'normalization': NORMALIZATION, 'ordered_row_hash_protocol': 'sha256-canonical-json-ordered-columns-and-string-rows-v1',
        'section_protocol': 'actual-heading-level-stack; exactly-one-matching-source-file; exactly-one-nonempty-heading-path',
        'markdown_heading_rule': 'ATX-1-to-6-hashes; max-3-leading-spaces; whitespace-or-end-after-hashes; ignore-backtick-and-tilde-fenced-code',
        'previous_version_comparison': previous_comparisons,
        'repeated_matches_same_path_allowed': True, 'unresolved_rule': 'keep original Section and export for human review',
        'human_verified': False, 'pdf_verified': False, 'training_or_inference_performed': False,
        'test_or_final_test_used': False, 'production_models_changed': False,
        'source_documents': len(documents), 'source_files': sum(item['source_count'] for item in documents),
        'preparation_script': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__)),
                               'snapshot': str(snapshot_path.relative_to(output)).replace('\\', '/')},
        'pandas_version': pd.__version__,
        'evidence_files': {str(path.relative_to(output)).replace('\\', '/'): sha(path) for path in output.rglob('*') if path.is_file()},
    }
    write_json(output / 'source_manifest.json', documents)
    manifest['evidence_files']['source_manifest.json'] = sha(output / 'source_manifest.json')
    write_json(output / 'preparation_manifest.json', manifest)
    report = ['# Dataset context v2.1 — chỉ chuẩn bị', '',
              'Phiên bản mới chỉ thay Section bằng đường dẫn heading có bằng chứng từ Markdown cleaned. '
              'Chưa train hoặc chạy inference với context này; chưa có F1 cho phiên bản này. '
              'Dữ liệu cũ, checkpoint và ensemble production giữ nguyên.', '',
              '| Split | Câu | Section trống trước | Trống sau | Đã thay Section | Khôi phục trống | Sửa Section có sẵn | Cần người kiểm tra |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, summary in split_summaries.items():
        report.append(f"| {split} | {summary['rows']} | {summary['section_blank_before']} | {summary['section_blank_after']} | "
                      f"{summary['section_changed']} | {summary['blank_section_recovered']} | {summary['existing_nonblank_section_corrected']} | "
                      f"{summary['unresolved_kept_old_section']} |")
    report += ['', 'Section chỉ được cập nhật khi text chuẩn hóa khớp đúng một file nguồn và đúng một đường dẫn heading khác rỗng. '
               'Các lần xuất hiện lặp cùng đường dẫn được chấp nhận. Nhiều nguồn khớp, nhiều đường dẫn, trước heading đầu, '
               'nguồn thiếu hoặc không đọc được đều giữ Section cũ và xuất để người kiểm tra. '
               'Vị trí khớp chuỗi không thay thế việc kiểm chứng PDF hoặc ngữ nghĩa heading.', '',
               'Thuật toán lưu cấp heading thực tế; khi gặp heading mới bỏ heading cùng cấp hoặc thấp hơn trong stack. '
               'Ví dụ `# Title → #### First → #### Second` cho `Title > Second`, không giữ `First` làm cha. '
               'Chỉ nhận heading ATX có 1–6 dấu #, theo sau là khoảng trắng hoặc hết dòng, thụt tối đa 3 dấu cách. '
               'Bỏ qua dấu # trong code fence backtick hoặc tilde; fence chỉ đóng khi cùng loại marker và độ dài ít nhất bằng fence mở. '
               'Các chuỗi như `#define`, `#!/bin/bash`, `#1` không được coi là heading. '
               'Không thay prev_text/next_text dù chúng có thể được tạo từ chuỗi CSV thưa; đây là bản thay đổi duy nhất Section.', '',
               'Nhãn, thứ tự, document_id, text, prev_text, pos và next_text được kiểm tra bằng so sánh DataFrame trước/sau. '
               'Manifest lưu hash file đầu vào, hash hàng có thứ tự trước/sau và hash mọi nguồn đọc. '
               'Nhãn tham chiếu vẫn là nhãn frozen; không nhập đề xuất AI hoặc bất đồng annotation chưa phân xử.', '',
               '- [Manifest phiên bản và hash](./preparation_manifest.json)',
               '- [Nguồn Markdown và hash](./source_manifest.json)',
               '- [Toàn bộ thay đổi và vị trí bằng chứng](./section_changes_and_provenance.csv)',
               '- [Các câu cần người kiểm tra](./section_unresolved_for_review.csv)',
               '- [Training rows v2](./training_rows.csv)',
               '- [Validation rows v2](./validation_rows.csv)', '',
               'Không dùng test/final test. Context v2 chưa được dùng để chọn checkpoint. '
               'Mọi so sánh F1 cũ vẫn chỉ áp dụng với context frozen cũ.', '']
    if previous_comparisons:
        report += ['## Khác biệt so với bản v2 được bảo toàn', '',
                   '| Split | Số câu có Section khác v2 |', '|---|---:|']
        for split, comparison in previous_comparisons.items():
            report.append(f"| {split} | {comparison['section_differences']} |")
        report += ['', 'Bản v2 cũ được giữ nguyên. [Các câu khác v2](./differences_from_v2.csv). '
                   'Đây là sửa cách đọc Markdown, chưa đo tác động lên điểm mô hình.', '']
    (output / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=DEFAULT_INPUT / 'training_rows.csv')
    parser.add_argument('--validation', type=Path, default=DEFAULT_INPUT / 'validation_rows.csv')
    parser.add_argument('--cleaned-root', type=Path, default=ROOT / 'data_cleaned')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--previous-version', type=Path, default=ROOT / 'reports/research_review/studies/classifier_context_v2')
    args = parser.parse_args()
    if args.training.resolve() == (DEFAULT_INPUT / 'training_rows.csv').resolve() and args.validation.resolve() == (DEFAULT_INPUT / 'validation_rows.csv').resolve():
        # Fail closed if the expected frozen source has changed in size.
        if len(read_rows(args.training)) != 4144 or len(read_rows(args.validation)) != 1317:
            raise ValueError('Unexpected frozen train/validation counts')
    manifest = prepare(args.training, args.validation, args.cleaned_root, args.output, args.previous_version)
    print(json.dumps({'status': manifest['status'], 'splits': manifest['splits'],
                      'previous_version_comparison': manifest['previous_version_comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
