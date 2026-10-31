"""Validate the 91-item classifier review export without changing any dataset.

Annotations are human-supplied decisions, not independently verified ground
truth. Outputs are review artifacts and a PROPOSED training overlay only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DEFAULT_PACKET = ROOT / 'reports/research_review/studies/classifier_stability_v1/human_review'
LABELS = ('OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER')
IMMUTABLE = ('item_id', 'review_scope', 'document_id', 'text', 'section', 'prev_text', 'next_text')
ANSWERS = ('reviewer', 'annotation', 'reason', 'source_checked', 'source_page')
PUBLIC_COLUMNS = (*IMMUTABLE, 'content_sha256', *ANSWERS)
EXPECTED_SCOPES = {'training_ai_proposal': 57, 'validation_disagreement': 34}
SOURCE_CHOICES = {'', 'unchecked', 'yes', 'unavailable'}
PLACEHOLDER_REVIEWERS = {'ai', 'chatgpt', 'gpt', 'assistant', 'unknown', 'n/a', 'na', 'none', 'todo', 'tbd', '?'}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def content_sha256(row: dict) -> str:
    # Match the original packet builder, using actual submitted context values.
    content = {field: row[field] for field in IMMUTABLE}
    return digest(json.dumps(content, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def _read_csv(raw: bytes, label: str, required: tuple, *, exact: bool = False) -> list[dict]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
        names = reader.fieldnames
        if names is None or len(names) != len(set(names)) or not set(required) <= set(names):
            raise ValueError('missing or duplicate column names')
        if exact and set(names) != set(required):
            raise ValueError('unexpected columns in the review export')
        rows = list(reader)
        if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('empty CSV or malformed row width')
        ids = [row['item_id'] for row in rows]
        if any(not item_id.strip() for item_id in ids) or len(ids) != len(set(ids)):
            raise ValueError('blank or duplicate item_id')
        return rows
    except (csv.Error, UnicodeError, ValueError) as exc:
        raise ValueError(f'{label}: {exc}') from exc


def _source_line(value: str) -> str:
    try:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value() or number < 1:
            raise ValueError('source_line must be a positive integer')
        return str(int(number))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f'Invalid training source_line: {value!r}') from exc


def _load_packet(packet_dir: Path) -> dict:
    files = {name: (packet_dir / name).read_bytes() for name in
             ('PHAN_XU_91_CAU.csv', 'coordinator_key.csv', 'manifest.json')}
    assigned = _read_csv(files['PHAN_XU_91_CAU.csv'], 'assigned packet', PUBLIC_COLUMNS, exact=True)
    key = _read_csv(files['coordinator_key.csv'], 'coordinator key',
                    (*IMMUTABLE, 'reference_label', 'original_label', 'proposed_label', 'source_line'))
    manifest = json.loads(files['manifest.json'])
    if len(assigned) != 91 or Counter(row['review_scope'] for row in assigned) != Counter(EXPECTED_SCOPES):
        raise ValueError('Trusted packet must contain 57 training and 34 validation items')
    indexed_key = {row['item_id']: row for row in key}
    if set(indexed_key) != {row['item_id'] for row in assigned}:
        raise ValueError('Coordinator key IDs differ from the assigned packet')
    public_payload = []
    for row in assigned:
        original = indexed_key[row['item_id']]
        if any(row[field] != original[field] for field in IMMUTABLE):
            raise ValueError(f"Assigned content differs from coordinator key: {row['item_id']}")
        if row['content_sha256'] != content_sha256(row):
            raise ValueError(f"Assigned content hash is stale: {row['item_id']}")
        public_payload.append({field: row[field] for field in (*IMMUTABLE, 'content_sha256')})
        if original['reference_label'] not in LABELS:
            raise ValueError('Coordinator reference label is invalid')
        if row['review_scope'] == 'training_ai_proposal':
            if (original['original_label'] not in LABELS or original['proposed_label'] not in LABELS
                    or original['original_label'] != original['reference_label']
                    or original['proposed_label'] == original['original_label']):
                raise ValueError('Coordinator training proposal is inconsistent')
            _source_line(original['source_line'])
    payload = json.dumps(public_payload, ensure_ascii=False).replace('<', r'\u003c').encode('utf-8')
    if (manifest.get('items') != 91 or manifest.get('training_items') != 57
            or manifest.get('validation_items') != 34 or manifest.get('packet_sha256') != digest(payload)):
        raise ValueError('Packet manifest does not bind the assigned review content')
    source_hashes = manifest.get('source_sha256')
    expected_sources = {'seed_2026_regressions.csv', 'ai_57_effective_proposals.csv'}
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_sources:
        raise ValueError('Packet manifest lacks the expected analysis source hashes')
    source_rows = {}
    for filename, expected in source_hashes.items():
        path = packet_dir.parent / 'analysis' / filename
        source_raw = path.read_bytes() if path.is_file() else None
        if source_raw is None or digest(source_raw) != expected:
            raise ValueError(f'Packet analysis source changed or is missing: {filename}')
        required = tuple(field for field in IMMUTABLE if field != 'review_scope')
        required += (('original_label', 'proposed_label', 'source_line')
                     if filename == 'ai_57_effective_proposals.csv' else ('true_label',))
        rows = _read_csv(source_raw, f'analysis source {filename}', required)
        source_rows[filename] = {row['item_id']: row for row in rows}
    training_source = source_rows['ai_57_effective_proposals.csv']
    validation_source = source_rows['seed_2026_regressions.csv']
    if set(training_source) & set(validation_source):
        raise ValueError('Training/validation analysis sources contain ambiguous shared IDs')
    assigned_training = {row['item_id'] for row in assigned if row['review_scope'] == 'training_ai_proposal'}
    if set(training_source) != assigned_training:
        raise ValueError('Training assignment IDs differ from the 57 effective AI proposals')
    for key_row in key:
        training = key_row['review_scope'] == 'training_ai_proposal'
        analysis_row = (training_source if training else validation_source).get(key_row['item_id'])
        if analysis_row is None:
            raise ValueError(f"Coordinator scope/ID is absent from its analysis source: {key_row['item_id']}")
        if any(key_row[field] != analysis_row[field] for field in IMMUTABLE if field != 'review_scope'):
            raise ValueError(f"Coordinator content differs from analysis source: {key_row['item_id']}")
        if training:
            expected_labels = (analysis_row['original_label'], analysis_row['original_label'], analysis_row['proposed_label'])
            if _source_line(key_row['source_line']) != _source_line(analysis_row['source_line']):
                raise ValueError(f"Coordinator source_line differs from analysis source: {key_row['item_id']}")
        else:
            if analysis_row['true_label'] not in ('REQUIREMENT', 'EXPECTED_RESULT'):
                raise ValueError('Validation assignment is outside the declared review-label strata')
            expected_labels = (analysis_row['true_label'], '', '')
        actual_labels = tuple(key_row[field] for field in ('reference_label', 'original_label', 'proposed_label'))
        if actual_labels != expected_labels:
            raise ValueError(f"Coordinator labels differ from analysis source: {key_row['item_id']}")
    return dict(files=files, assigned=assigned, key=indexed_key, manifest=manifest)


def validate_submission(submission: Path, *, packet_dir: Path = DEFAULT_PACKET,
                        complete: bool = False) -> dict:
    """Check a full exported assignment; partial mode permits undecided labels."""
    packet_dir, submission = Path(packet_dir).resolve(), Path(submission).resolve()
    packet = _load_packet(packet_dir)
    raw = submission.read_bytes()
    submitted = _read_csv(raw, 'submitted review', PUBLIC_COLUMNS, exact=True)
    by_id = {row['item_id']: row for row in submitted}
    assigned_ids = {row['item_id'] for row in packet['assigned']}
    if set(by_id) != assigned_ids:
        raise ValueError('Submission must retain all 91 assigned IDs, with no additions or omissions')
    decisions, overlay = [], []
    for assigned in packet['assigned']:
        row = by_id[assigned['item_id']]
        if any(row[field] != assigned[field] for field in IMMUTABLE):
            raise ValueError(f"Immutable context changed for {row['item_id']}")
        recomputed = content_sha256(row)
        if row['content_sha256'] != recomputed or recomputed != assigned['content_sha256']:
            raise ValueError(f"Submitted content hash does not match actual content: {row['item_id']}")
        answer = {field: row[field].strip() for field in ANSWERS}
        annotation = answer['annotation']
        if annotation not in (*LABELS, ''):
            raise ValueError(f"Invalid annotation for {row['item_id']}: {annotation!r}")
        if answer['source_checked'] not in SOURCE_CHOICES:
            raise ValueError(f"Invalid source_checked for {row['item_id']}")
        if answer['source_checked'] == 'yes' and not answer['source_page']:
            raise ValueError(f"source_page is required when source_checked=yes: {row['item_id']}")
        if annotation and (not answer['reason'] or not answer['reviewer']
                           or ' '.join(answer['reviewer'].casefold().split()) in PLACEHOLDER_REVIEWERS):
            raise ValueError(f"Completed annotation requires a named reviewer and reason: {row['item_id']}")
        if complete and not annotation:
            raise ValueError(f"Complete submission still has an undecided item: {row['item_id']}")
        original = packet['key'][row['item_id']]
        training = row['review_scope'] == 'training_ai_proposal'
        finished = bool(annotation)
        accepted = finished and training and annotation == original['proposed_label']
        retained = finished and annotation == original['reference_label']
        alternative = finished and not accepted and not retained
        decision = ('pending' if not finished else 'accepted_AI' if accepted
                    else 'original_retained' if retained else 'alternative')
        result = {field: row[field] for field in (*IMMUTABLE, 'content_sha256')}
        result.update(answer)
        result.update(reference_label=original['reference_label'],
                      original_label=original['original_label'], ai_proposed_label=original['proposed_label'],
                      source_line=_source_line(original['source_line']) if training else '',
                      annotation_complete=finished, decision=decision,
                      accepted_AI=accepted, rejected_AI=finished and training and not accepted,
                      original_retained=retained, alternative=alternative,
                      evidence_status='human_supplied_not_independently_verified',
                      labels_applied=False)
        decisions.append(result)
        if training and finished:
            overlay.append(dict(overlay_status='PROPOSED_NOT_APPLIED', item_id=row['item_id'],
                                document_id=row['document_id'], source_line=result['source_line'],
                                text=row['text'], annotation=annotation,
                                original_label=original['original_label'],
                                ai_proposed_label=original['proposed_label'], decision=decision,
                                reviewer=answer['reviewer'], reason=answer['reason'],
                                source_checked=answer['source_checked'], source_page=answer['source_page'],
                                content_sha256=recomputed, submission_sha256=digest(raw)))
    training_rows = [row for row in decisions if row['review_scope'] == 'training_ai_proposal']
    validation_rows = [row for row in decisions if row['review_scope'] == 'validation_disagreement']
    def counts(rows):
        return dict(assigned=len(rows), completed=sum(row['annotation_complete'] for row in rows),
                    pending=sum(not row['annotation_complete'] for row in rows),
                    **{name: sum(bool(row[name]) for row in rows) for name in
                       ('accepted_AI', 'rejected_AI', 'original_retained', 'alternative')},
                    source_checked_yes=sum(row['source_checked'] == 'yes' for row in rows))
    summary = dict(schema_version=1, validated_utc=datetime.now(timezone.utc).isoformat(),
                   submission_path=str(submission), submission_sha256=digest(raw),
                   packet_dir=str(packet_dir), packet_sha256=packet['manifest']['packet_sha256'],
                   trusted_input_sha256={name: digest(value) for name, value in packet['files'].items()},
                   complete_required=complete, all_annotations_present=all(row['annotation_complete'] for row in decisions),
                   training=counts(training_rows), validation=counts(validation_rows),
                   proposed_training_overlay_rows=len(overlay), labels_applied=False,
                   validation_labels_modified=False, production_model_changed=False,
                   final_test_approved=False, reviewer_identity_independently_verified=False,
                   warnings=['Human-supplied labels and source-check declarations are not independently verified.',
                             'Targeted packet; do not infer corpus label accuracy or inter-rater Kappa.',
                             'rejected_AI overlaps original_retained and alternative in training counts.',
                             'Proposed overlay only; apply nothing to frozen datasets or historical validation scores.'])
    return dict(summary=summary, training_decisions=training_rows, validation_decisions=validation_rows,
                proposed_overlay=overlay, submission_bytes=raw, trusted_files=packet['files'])


DECISION_COLUMNS = (*IMMUTABLE, 'content_sha256', *ANSWERS, 'reference_label', 'original_label',
                    'ai_proposed_label', 'source_line', 'annotation_complete', 'decision',
                    'accepted_AI', 'rejected_AI', 'original_retained', 'alternative', 'evidence_status', 'labels_applied')
OVERLAY_COLUMNS = ('overlay_status', 'item_id', 'document_id', 'source_line', 'text', 'annotation',
                   'original_label', 'ai_proposed_label', 'decision', 'reviewer', 'reason',
                   'source_checked', 'source_page', 'content_sha256', 'submission_sha256')


def _write_csv(path: Path, rows: list[dict], columns: tuple):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def import_adjudication(submission: Path, output_dir: Path, *, packet_dir: Path = DEFAULT_PACKET,
                         complete: bool = False) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Choose a new output directory; previous imports are never overwritten')
    validated = validate_submission(submission, packet_dir=packet_dir, complete=complete)
    summary = validated['summary']
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.adjudication-', dir=output_dir.parent))
    try:
        (staging / 'submitted.csv').write_bytes(validated['submission_bytes'])
        for name, raw in validated['trusted_files'].items():
            (staging / ('trusted_' + name)).write_bytes(raw)
        _write_csv(staging / 'training_decisions.csv', validated['training_decisions'], DECISION_COLUMNS)
        _write_csv(staging / 'validation_decisions.csv', validated['validation_decisions'], DECISION_COLUMNS)
        _write_csv(staging / 'proposed_reviewed_training_overlay.csv', validated['proposed_overlay'], OVERLAY_COLUMNS)
        (staging / 'validation_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        report = '# Classifier adjudication import\n\n'
        report += 'Structural validation passed. Decisions are human-supplied; identity and semantic correctness have not been independently verified.\n\n'
        report += '| Scope | Assigned | Annotated | Pending | Accepted AI | Rejected AI | Original retained | Alternative |\n'
        report += '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
        for scope in ('training', 'validation'):
            row = summary[scope]
            report += '| ' + scope + ' | ' + ' | '.join(str(row[k]) for k in ('assigned', 'completed', 'pending', 'accepted_AI', 'rejected_AI', 'original_retained', 'alternative')) + ' |\n'
        report += '\nAI rejection counts overlap retained-original/alternative training decisions. Blank annotations remain pending.\n\n'
        report += f"Submission SHA256: `{summary['submission_sha256']}`.\n\n"
        report += 'Evidence: [exact submitted file](submitted.csv), [validation report](validation_report.json), [training decisions](training_decisions.csv), [validation decisions](validation_decisions.csv), [PROPOSED training overlay](proposed_reviewed_training_overlay.csv).\n\n'
        report += 'No dataset, validation reference label, source packet manifest, production model, or final-test gate was changed. The proposed overlay is not approved for automatic application. This targeted packet cannot establish corpus label accuracy or inter-rater Kappa.\n'
        (staging / 'VALIDATION_REPORT.md').write_text(report, encoding='utf-8')
        staging.rename(output_dir)
    finally:
        if staging.exists():
            if staging.parent.resolve() != output_dir.parent or not staging.name.startswith('.adjudication-'):
                raise RuntimeError('Refusing cleanup outside the verified staging directory')
            shutil.rmtree(staging)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CSV exported from the 91-item workbench')
    parser.add_argument('--output', type=Path, required=True, help='New import evidence directory')
    parser.add_argument('--packet-dir', type=Path, default=DEFAULT_PACKET)
    parser.add_argument('--complete', action='store_true', help='Require all 91 items to be annotated')
    args = parser.parse_args(argv)
    try:
        summary = import_adjudication(args.input, args.output, packet_dir=args.packet_dir, complete=args.complete)
    except (OSError, ValueError) as exc:
        print(f'Adjudication import rejected: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
