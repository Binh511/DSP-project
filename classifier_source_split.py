"""Prepare paired classifier data with full-source-disjoint train/validation.

Validation takes priority. Only training documents that duplicate a validation
source are purged. No training, inference, production or test/final-test access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'reports/research_review/studies/user_labels_v1/experiments/baseline_original_labels'
CONTEXT = ROOT / 'reports/research_review/studies/classifier_context_v2_1'
OUTPUT = ROOT / 'reports/research_review/studies/classifier_source_clean_v1'
VERSION = 'classifier-source-clean-v1-validation-priority'
COLS = ['document_id', 'text', 'label', 'section', 'prev_text', 'pos', 'next_text']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if list(rows.columns) != COLS or rows.empty:
        raise ValueError(f'Unexpected or empty classifier data: {path}')
    if rows.document_id.str.strip().eq('').any():
        raise ValueError(f'Missing document ID: {path}')
    return rows


def row_hash(rows: pd.DataFrame) -> str:
    value = {'columns': list(rows.columns), 'rows': rows.astype(str).values.tolist()}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def normalized_source_hash(text: str) -> str:
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC', text).casefold().split()).encode('utf-8')).hexdigest()


def source_records(cleaned_root: Path, training_docs: set[str], validation_docs: set[str],
                   expected_manifest: Path | None = None) -> list[dict]:
    if training_docs & validation_docs:
        raise ValueError('Original training/validation document IDs overlap')
    expected = {}
    if expected_manifest is not None:
        content = json.loads(expected_manifest.read_text(encoding='utf-8'))
        for document in content:
            entries = document['sources']
            if len(entries) != 1:
                raise ValueError(f"Ambiguous prepared context source: {document['document_id']}")
            expected[document['document_id']] = entries[0]
    records = []
    for document_id in sorted(training_docs | validation_docs):
        if not document_id or any(char in document_id for char in ('/', '\\', ':')) or document_id in {'.', '..'}:
            raise ValueError(f'Unsafe document ID: {document_id!r}')
        folder = cleaned_root / document_id
        files = sorted(set(folder.glob('*.md')) | set(folder.glob('*.txt')))
        if len(files) != 1:
            raise ValueError(f'{document_id} must have exactly one Markdown/text source, found {len(files)}')
        path = files[0]
        raw_hash = sha(path)
        if expected_manifest is not None:
            entry = expected.get(document_id)
            if (entry is None or Path(entry['path']).resolve() != path.resolve()
                    or entry['sha256'] != raw_hash):
                raise ValueError(f'Current source differs from prepared context source: {document_id}')
        text = path.read_text(encoding='utf-8-sig')
        if not text.strip():
            raise ValueError(f'Empty source: {document_id}')
        records.append({'document_id': document_id, 'original_split': 'train' if document_id in training_docs else 'validation',
                        'source_path': str(path.resolve()), 'raw_sha256': raw_hash,
                        'normalized_sha256': normalized_source_hash(text)})
    return records


def groups(records: list[dict], field: str) -> list[dict]:
    buckets = defaultdict(list)
    for record in records:
        buckets[record[field]].append(record)
    return [{'sha256': hash_value,
             'members': [{key: member[key] for key in ['document_id', 'original_split', 'retained_split']} for member in members]}
            for hash_value, members in sorted(buckets.items())]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(baseline_training: Path, baseline_validation: Path, context_training: Path,
            context_validation: Path, cleaned_root: Path, output: Path,
            expected_source_manifest: Path | None = None,
            expected_preparation_manifest: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite versioned data: {output}')
    input_paths = {'baseline_training_rows.csv': baseline_training, 'baseline_validation_rows.csv': baseline_validation,
                   'context_training_rows.csv': context_training, 'context_validation_rows.csv': context_validation}
    if expected_preparation_manifest is not None:
        prior = json.loads(expected_preparation_manifest.read_text(encoding='utf-8'))
        for split, old_path, new_path, name in [('train', baseline_training, context_training, 'training_rows.csv'),
                                               ('validation', baseline_validation, context_validation, 'validation_rows.csv')]:
            if sha(old_path) != prior['splits'][split]['original_input_file_sha256'] or sha(new_path) != prior['evidence_files'][name]:
                raise ValueError('Input dataset differs from context preparation evidence: ' + split)
    original = {name: read_rows(path) for name, path in input_paths.items()}
    base_train, base_val = original['baseline_training_rows.csv'], original['baseline_validation_rows.csv']
    new_train, new_val = original['context_training_rows.csv'], original['context_validation_rows.csv']
    pd.testing.assert_frame_equal(base_train.drop(columns='section'), new_train.drop(columns='section'))
    pd.testing.assert_frame_equal(base_val.drop(columns='section'), new_val.drop(columns='section'))
    records = source_records(cleaned_root, set(base_train.document_id), set(base_val.document_id), expected_source_manifest)
    validation_sources = [record for record in records if record['original_split'] == 'validation']
    validation_raw = {record['raw_sha256'] for record in validation_sources}
    validation_normalized = {record['normalized_sha256'] for record in validation_sources}
    removed_documents = {record['document_id'] for record in records
                         if record['original_split'] == 'train'
                         and (record['raw_sha256'] in validation_raw or record['normalized_sha256'] in validation_normalized)}
    for record in records:
        record['retained_split'] = 'removed_train' if record['document_id'] in removed_documents else record['original_split']
        record['duplicate_validation_documents'] = sorted(other['document_id'] for other in validation_sources
                                                         if other['raw_sha256'] == record['raw_sha256']
                                                         or other['normalized_sha256'] == record['normalized_sha256'])
    kept = ~base_train.document_id.isin(removed_documents)
    prepared = {
        'baseline_training_rows.csv': base_train[kept].reset_index(drop=True),
        'baseline_validation_rows.csv': base_val.copy(deep=True),
        'context_training_rows.csv': new_train[kept].reset_index(drop=True),
        'context_validation_rows.csv': new_val.copy(deep=True),
    }
    if prepared['baseline_training_rows.csv'].empty:
        raise ValueError('Purge would remove all training rows')
    pd.testing.assert_frame_equal(prepared['baseline_validation_rows.csv'], base_val)
    pd.testing.assert_frame_equal(prepared['context_validation_rows.csv'], new_val)
    for split in ['training', 'validation']:
        pd.testing.assert_frame_equal(prepared[f'baseline_{split}_rows.csv'].drop(columns='section'),
                                      prepared[f'context_{split}_rows.csv'].drop(columns='section'))
    overlap_groups = []
    for field in ['raw_sha256', 'normalized_sha256']:
        train_hashes = {record[field] for record in records if record['retained_split'] == 'train'}
        val_hashes = {record[field] for record in records if record['retained_split'] == 'validation'}
        overlap_groups.extend({'field': field, 'sha256': value} for value in sorted(train_hashes & val_hashes))
    assert not overlap_groups, 'Full-source hashes still cross train/validation'
    by_doc = {record['document_id']: record for record in records}
    removed_rows = base_train[~kept].copy()
    removed_rows['original_train_row_index'] = removed_rows.index
    removed_rows['removal_reason'] = 'full_source_duplicates_validation'
    removed_rows['duplicate_validation_document_ids'] = removed_rows.document_id.map(
        lambda document: json.dumps(by_doc[document]['duplicate_validation_documents']))
    removed_rows['raw_source_sha256'] = removed_rows.document_id.map(lambda document: by_doc[document]['raw_sha256'])
    removed_rows['normalized_source_sha256'] = removed_rows.document_id.map(lambda document: by_doc[document]['normalized_sha256'])
    source_groups = {'normalization': 'sha256-utf8-nfc-casefold-collapse-whitespace-v1', 'documents': records,
                     'raw_groups': groups(records, 'raw_sha256'), 'normalized_groups': groups(records, 'normalized_sha256'),
                     'train_validation_source_overlap_after_purge': 0}
    # Output is versioned; all checks above finish before exclusive creation.
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in prepared.items():
        rows.to_csv(output / name, index=False, encoding='utf-8-sig')
        pd.testing.assert_frame_equal(read_rows(output / name), rows)
    removed_rows.to_csv(output / 'removed_training_rows.csv', index=False, encoding='utf-8-sig')
    write_json(output / 'source_groups.json', source_groups)
    snapshot = output / 'source_snapshot' / 'classifier_source_split.py'
    snapshot.parent.mkdir()
    snapshot.write_bytes(Path(__file__).read_bytes())
    inputs = {name: {'path': str(path.resolve()), 'file_sha256': sha(path),
                     'ordered_row_sha256': row_hash(original[name]), 'rows': len(original[name])}
              for name, path in input_paths.items()}
    if expected_source_manifest is not None:
        inputs['context_source_manifest'] = {'path': str(expected_source_manifest.resolve()), 'file_sha256': sha(expected_source_manifest)}
    if expected_preparation_manifest is not None:
        inputs['context_preparation_manifest'] = {'path': str(expected_preparation_manifest.resolve()), 'file_sha256': sha(expected_preparation_manifest)}
    class_counts = lambda frame: {key: int(value) for key, value in sorted(Counter(frame.label).items())}
    manifest = {
        'version': VERSION, 'status': 'preparation_only_not_train_evaluated',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purge_rule': 'validation priority; remove whole training document if raw or normalized full-source SHA256 matches validation',
        'source_hashes': {'raw': 'sha256 of source file bytes', 'normalized': 'sha256 of utf8 text after BOM decoding, Unicode NFC, casefold and collapsed whitespace'},
        'train_rows_before': len(base_train), 'train_rows': int(kept.sum()), 'validation_rows': len(base_val),
        'removed_train_rows': int((~kept).sum()), 'removed_train_documents': sorted(removed_documents),
        'train_documents_before': int(base_train.document_id.nunique()),
        'train_documents': int(prepared['baseline_training_rows.csv'].document_id.nunique()),
        'validation_documents': int(base_val.document_id.nunique()),
        'train_validation_source_overlap_after_purge': 0,
        'source_files_bound': len(records),
        'class_counts': {'training_before': class_counts(base_train),
                         'training_after': class_counts(prepared['baseline_training_rows.csv']),
                         'removed_training': class_counts(removed_rows), 'validation_unchanged': class_counts(base_val)},
        'validation_exactly_preserved_in_both_contexts': True,
        'paired_rows_identical_except_section': True, 'row_order_preserved': True,
        'training_or_inference_performed': False, 'production_models_changed': False, 'test_or_final_test_used': False,
        'inputs': inputs,
        'output_rows': {name: {'rows': len(rows), 'ordered_row_sha256': row_hash(rows)} for name, rows in prepared.items()},
        'preparation_script': {'path': str(Path(__file__).resolve()), 'snapshot': str(snapshot.relative_to(output)).replace('\\', '/'),
                               'sha256': sha(snapshot)},
        'evidence_files': {str(path.relative_to(output)).replace('\\', '/'): sha(path) for path in output.rglob('*') if path.is_file()},
    }
    write_json(output / 'preparation_manifest.json', manifest)
    report = ['# Dữ liệu classifier tách theo nguồn — version 1', '',
              'Chỉ chuẩn bị dữ liệu đối chứng cho classifier. Không train hoặc inference; chưa có kết quả mô hình cho split này. '
              'Validation được ưu tiên giữ nguyên. Loại toàn bộ tài liệu train có full-source raw SHA256 hoặc hash nguồn '
              'chuẩn hóa khoảng trắng/casefold trùng với bất kỳ tài liệu validation.', '',
              f"Train: **{len(base_train)} → {int(kept.sum())} câu**, **{base_train.document_id.nunique()} → {prepared['baseline_training_rows.csv'].document_id.nunique()} tài liệu**. "
              f"Loại **{int((~kept).sum())} câu** thuộc **{len(removed_documents)} tài liệu**.", '',
              f"Validation giữ nguyên **{len(base_val)} câu/{base_val.document_id.nunique()} tài liệu** cho cả hai context. "
              'Không còn nhóm raw/normalized full-source hash giao train/validation. '
              'Nhãn, text, thứ tự và mọi trường ngoài Section trùng nhau giữa baseline/context sau lọc.', '',
              'Tài liệu train bị loại: ' + (', '.join(sorted(removed_documents)) or '(không có)') + '.', '',
              '| Nhãn | Train trước | Loại | Train sau | Validation giữ nguyên |', '|---|---:|---:|---:|---:|']
    for label in ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']:
        counts = manifest['class_counts']
        report.append(f"| {label} | {counts['training_before'].get(label, 0)} | {counts['removed_training'].get(label, 0)} | "
                      f"{counts['training_after'].get(label, 0)} | {counts['validation_unchanged'].get(label, 0)} |")
    report += ['', 'Split trước đây có nguồn toàn tài liệu trùng nhau dù các câu được giữ trong CSV không nhất thiết trùng. '
               'Kết quả các model cũ chỉ là tham chiếu lịch sử; không phải đối chứng cùng dữ liệu cho bộ chuẩn bị mới.', '',
               'Cả hai phương án dùng cùng hàng dữ liệu sau lọc; baseline giữ Section frozen cũ, context dùng Section v2.1. '
               'Các trường chưa phân xử của context v2.1 vẫn giữ giá trị cũ và không được coi là người đã kiểm định. '
               'Không xóa trùng giữa các tài liệu cùng phía validation; kiểm tra này chỉ bảo đảm source hash không đi qua biên train/validation.', '',
               '- [Baseline training](./baseline_training_rows.csv)', '- [Baseline validation](./baseline_validation_rows.csv)',
               '- [Context training](./context_training_rows.csv)', '- [Context validation](./context_validation_rows.csv)',
               '- [Các câu train bị loại và nguồn validation tương ứng](./removed_training_rows.csv)',
               '- [Nhóm nguồn và hash](./source_groups.json)', '- [Manifest, hash và kiểm tra bảo toàn dữ liệu](./preparation_manifest.json)',
               '- [Snapshot script](./source_snapshot/classifier_source_split.py)', '',
               'Nguồn thiếu, có nhiều file hoặc thay đổi so với manifest context phải dừng chuẩn bị; không chọn file tùy ý. '
               'Chưa đánh giá chất lượng source PDF hoặc nhãn tham chiếu; không dùng test/final test và không thay production.', '']
    (output / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-training', type=Path, default=BASELINE / 'training_rows.csv')
    parser.add_argument('--baseline-validation', type=Path, default=BASELINE / 'validation_rows.csv')
    parser.add_argument('--context-training', type=Path, default=CONTEXT / 'training_rows.csv')
    parser.add_argument('--context-validation', type=Path, default=CONTEXT / 'validation_rows.csv')
    parser.add_argument('--context-source-manifest', type=Path, default=CONTEXT / 'source_manifest.json')
    parser.add_argument('--context-preparation-manifest', type=Path, default=CONTEXT / 'preparation_manifest.json')
    parser.add_argument('--cleaned-root', type=Path, default=ROOT / 'data_cleaned')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.baseline_training, args.baseline_validation, args.context_training, args.context_validation,
                       args.cleaned_root, args.output, args.context_source_manifest, args.context_preparation_manifest)
    print(json.dumps({key: manifest[key] for key in ['version', 'train_rows_before', 'train_rows', 'validation_rows',
                                                    'removed_train_rows', 'removed_train_documents', 'class_counts',
                                                    'train_validation_source_overlap_after_purge']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
