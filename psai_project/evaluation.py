"""Score explicitly reviewed, source-bound plans without creating reference labels.

Reviewers and independence are declarations, not authenticated identities. Hashes
bind decisions to immutable inputs; they do not establish semantic correctness.
Only accepted one-to-one equivalence decisions contribute task true positives.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EVALUATION_VERSION = 1
DATASET_KINDS = frozenset({'case_study', 'human_review', 'independent_final_test'})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(value, description):
    if not isinstance(value, dict):
        raise ValueError(description + ' must be an object')
    return value


def _list(value, description):
    if not isinstance(value, list):
        raise ValueError(description + ' must be a list')
    return value


def _text(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(description + ' must be nonempty text')
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_json(path):
    raw = path.read_bytes()
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return _object(json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs,
        parse_constant=nonfinite), str(path)), _sha(raw)


def _span(span, source, source_hash):
    _object(span, 'Source span')
    start, end, quote = (span.get(key) for key in ('char_start', 'char_end', 'quote'))
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source)
            or not isinstance(quote, str) or not quote.strip() or source[start:end] != quote
            or span.get('source_sha256') != source_hash):
        raise ValueError('Source span/hash does not match the immutable source')
    return {key: span[key] for key in ('char_start', 'char_end', 'quote', 'source_sha256')}


def reference_task_sha256(task):
    """Bind a reference decision to its ID, wording and exact source spans.

    Call after manually supplying a reference task. This generates no reference
    task, semantic label, reviewer identity, or approval.
    """
    return _digest({'task_id': task['task_id'], 'description': task['description'],
        'source_spans': [{key: span[key] for key in ('char_start', 'char_end', 'quote', 'source_sha256')}
                         for span in task['source_spans']]})


def snapshot_run(run_dir):
    """Read and validate a saved run; no inference, downloads or output writes.

    Rules plans without task IDs use ``step:N``. Task/dependency digests include
    the entire original item, so reviews cannot migrate to a changed prediction.
    ``manifest.json`` is a declared fallback, never invented worker evidence.
    """
    from planning import _evidence_index, validate_plan
    from .report import _validate_local_plan

    root = Path(run_dir).resolve()
    source_raw = (root / 'source.txt').read_bytes()
    source, source_hash = source_raw.decode('utf-8'), _sha(source_raw)
    analysis, analysis_hash = _read_json(root / 'analysis.json')
    record, plan = (_object(analysis.get(key), key) for key in ('extraction', 'plan'))
    document_id = _text(analysis.get('document_id'), 'document_id')
    if any(item.get('document_id') != document_id or item.get('source_sha256') != source_hash
           for item in (record, plan)):
        raise ValueError('Plan/extraction document identity or source hash mismatch')
    metadata = _object(analysis.get('metadata', {}), 'Analysis metadata')
    if 'source_text_sha256' in metadata and metadata['source_text_sha256'] != source_hash:
        raise ValueError('Analysis metadata source hash mismatch')
    binding = {'source_sha256': source_hash, 'analysis_sha256': analysis_hash,
               'run_manifest_sha256': None, 'manifest_sha256': None}
    manifests = []
    for name, key in (('run_manifest.json', 'run_manifest_sha256'), ('manifest.json', 'manifest_sha256')):
        if (root / name).is_file():
            manifest, binding[key] = _read_json(root / name)
            if manifest.get('source_text_sha256') != source_hash:
                raise ValueError(name + ' source hash mismatch')
            manifests.append(name)
    if not manifests:
        raise ValueError('Saved run requires run_manifest.json or manifest.json')
    evidence = _evidence_index(record, source)
    validate_plan(plan, set(evidence), evidence_by_id=evidence)
    retrieval = _object(analysis.get('retrieval', {}), 'retrieval')
    snapshots = {source_hash: source}
    for identity, info in _object(retrieval.get('sources', {}), 'retrieval sources').items():
        _object(info, 'Retrieval source')
        artifact = (root / _text(info.get('artifact'), 'Source artifact')).resolve()
        if not artifact.is_relative_to(root):
            raise ValueError('Retrieval snapshot escapes saved run directory')
        text = artifact.read_bytes().decode('utf-8')
        if _sha(text.encode('utf-8')) != info.get('source_sha256'):
            raise ValueError('Retrieval source snapshot hash mismatch')
        snapshots[identity] = text
    for hit in _list(retrieval.get('hits', []), 'Retrieval hits'):
        _object(hit, 'Retrieval hit')
        identity = hit.get('snapshot_id') or hit.get('source_sha256')
        if (hit.get('verified') is not True or not isinstance(identity, str)
                or identity not in snapshots):
            raise ValueError('Retrieval hit requires a verified, known source snapshot')
        text = snapshots[identity]
        _span(hit, text, _sha(text.encode('utf-8')))
    _validate_local_plan(plan, record, source, retrieval, snapshots)
    if analysis.get('review_context') is not None:
        from psai_planner import validate_review_context
        validate_review_context(analysis['review_context'], record, source,
                                {**retrieval, 'source_texts': snapshots})
    tasks = []
    for item in plan['steps']:
        task_id = _text(item.get('task_id', 'step:' + str(item['step'])), 'Prediction task ID')
        spans = [_span({**evidence[eid], 'source_sha256': source_hash}, source, source_hash)
                 for eid in item['evidence_ids']]
        tasks.append({'task_id': task_id, 'task_sha256': _digest(item),
                      'description': item['description'], 'source_spans': spans})
    ids = [item['task_id'] for item in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prediction task ID')
    dependencies = []
    for item in _list(plan.get('dependency_candidates', []), 'Dependency candidates'):
        _object(item, 'Dependency')
        before, after = item.get('before_task_id'), item.get('after_task_id')
        if before not in ids or after not in ids or before == after:
            raise ValueError('Dependency must join distinct known prediction tasks')
        dependencies.append({'dependency_id': before + '->' + after,
            'dependency_sha256': _digest(item), 'before_task_id': before, 'after_task_id': after})
    if len({item['dependency_id'] for item in dependencies}) != len(dependencies):
        raise ValueError('Duplicate prediction dependency')
    return {'document_id': document_id, 'run_dir': str(root), 'binding': binding,
            'source_eligibility': {'independent_source_declared': metadata.get('independent_source'),
                'independent_declared': metadata.get('independent'),
                'source_exposure': metadata.get('source_exposure', 'not_recorded'),
                'synthetic_fixture_declared': metadata.get('synthetic_fixture') is True},
            'manifest_files': manifests, 'prediction_tasks': tasks,
            'prediction_dependencies': dependencies}


def build_evaluation_bundle(run_dirs, *, dataset_kind='case_study'):
    """Prepare unreviewed packets. Reference tasks and semantic matches stay empty."""
    if dataset_kind not in DATASET_KINDS:
        raise ValueError('Unknown dataset_kind')
    documents = []
    for index, path in enumerate(run_dirs, 1):
        document = snapshot_run(path)
        document.update({'case_id': 'case-' + str(index).zfill(3), 'gold_tasks': [], 'matches': [],
            'reference_review': {'status': 'pending', 'reviewer_id': '', 'rationale': ''},
            'matching_review': {'status': 'pending', 'reviewer_id': '', 'rationale': ''},
            'task_reviews': [{**{key: item[key] for key in ('task_id', 'task_sha256')},
                'evidence_correct': None, 'reviewer_id': '', 'rationale': ''}
                for item in document['prediction_tasks']],
            'dependency_reviews': [{**{key: item[key] for key in ('dependency_id', 'dependency_sha256')},
                'correct': None, 'reviewer_id': '', 'rationale': ''}
                for item in document['prediction_dependencies']]})
        documents.append(document)
    if not documents:
        raise ValueError('At least one saved run is required')
    return {'version': EVALUATION_VERSION, 'dataset_kind': dataset_kind, 'reviewers': [],
        'documents': documents, 'final_test_gate': {'sources_independent': False,
        'selection_frozen_before_review': False, 'no_test_used_for_selection': False,
        'approved': False, 'reviewer_id': '', 'rationale': ''}}


def _prf(tp, fp, fn):
    denominators = {'precision': tp + fp, 'recall': tp + fn, 'f1': 2 * tp + fp + fn}
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / denominators['precision'] if denominators['precision'] else None,
            'recall': tp / denominators['recall'] if denominators['recall'] else None,
            'f1': 2 * tp / denominators['f1'] if denominators['f1'] else None,
            'undefined': [key for key, value in denominators.items() if not value],
            'status': 'defined' if all(denominators.values()) else 'partly_or_fully_undefined'}


def _rate(correct, total):
    return {'correct': correct, 'total': total, 'rate': correct / total if total else None,
            'status': 'defined' if total else 'undefined_no_items'}


def _macro(rows, keys):
    result, counts = {}, {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        counts[key] = len(values)
    return {**result, 'defined_document_counts': counts, 'document_count': len(rows),
            'undefined_policy': 'exclude_undefined_documents_per_metric; never replace with zero'}


def evaluate_bundle(bundle, *, base_dir=None):
    """Validate explicit review decisions and compute separately scoped metrics.

    Incomplete review or unmet scope gates suppress aggregate numeric metrics.
    Invalid hashes, duplicated accepted matches and unsupported gold raise errors.
    Dependency correctness is a rate on predicted edges, not dependency recall.
    Retrieval relevance is deliberately reported as not evaluated.
    """
    _object(bundle, 'Evaluation bundle')
    if type(bundle.get('version')) is not int or bundle['version'] != EVALUATION_VERSION:
        raise ValueError('Unsupported evaluation version')
    kind = bundle.get('dataset_kind')
    if kind not in DATASET_KINDS:
        raise ValueError('Unknown dataset_kind')
    reviewers = {}
    for item in _list(bundle.get('reviewers'), 'Reviewers'):
        _object(item, 'Reviewer')
        rid = _text(item.get('reviewer_id'), 'reviewer_id')
        if rid in reviewers or item.get('kind') not in ('human', 'ai') or type(item.get('independent')) is not bool:
            raise ValueError('Reviewer IDs must be unique with explicit kind and independence')
        reviewers[rid] = item
    used, all_missing = set(), []

    def review_actor(item, location):
        rid = item.get('reviewer_id')
        if not rid or not isinstance(item.get('rationale'), str) or not item['rationale'].strip():
            return False
        if rid not in reviewers:
            raise ValueError(location + ': unknown reviewer_id')
        used.add(rid)
        return True

    def completion(item, location, missing):
        _object(item, location)
        if item.get('status') not in ('pending', 'complete'):
            raise ValueError(location + ': expected pending or complete status')
        if item['status'] != 'complete' or not review_actor(item, location):
            missing.append(location)

    def decisions(entries, known, id_key, hash_key, value_key, location, missing):
        seen, correct = set(), 0
        for entry in _list(entries, location):
            _object(entry, location)
            identity = entry.get(id_key)
            if identity not in known or identity in seen:
                raise ValueError(location + ': unknown or duplicate review target')
            seen.add(identity)
            if entry.get(hash_key) != known[identity][hash_key]:
                raise ValueError(location + ': prediction hash mismatch')
            value = entry.get(value_key)
            if value is not None and type(value) is not bool:
                raise ValueError(location + ': review decision must be boolean or null')
            if value is None or not review_actor(entry, location):
                missing.append(location + ':' + identity)
            elif value:
                correct += 1
        missing.extend(location + ':' + identity for identity in sorted(set(known) - seen))
        return _rate(correct, len(known))

    results, cases, runs, source_hashes, known_development = [], set(), set(), set(), []
    for document in _list(bundle.get('documents'), 'Documents'):
        _object(document, 'Document')
        case = _text(document.get('case_id', document.get('document_id')), 'case_id')
        if case in cases:
            raise ValueError('Duplicate case_id')
        cases.add(case)
        path = Path(_text(document.get('run_dir'), 'run_dir'))
        if not path.is_absolute():
            if base_dir is None:
                raise ValueError('Relative run_dir requires base_dir')
            path = Path(base_dir) / path
        snapshot = snapshot_run(path)
        if any(document.get(key) != snapshot[key] for key in
               ('document_id', 'binding', 'prediction_tasks', 'prediction_dependencies')):
            raise ValueError(case + ': saved run binding/predictions changed or missing')
        run_key = _digest(snapshot['binding'])
        if run_key in runs:
            raise ValueError('The same bound saved run cannot be scored twice')
        runs.add(run_key)
        source = (path / 'source.txt').read_bytes().decode('utf-8')
        source_hash = snapshot['binding']['source_sha256']
        source_hashes.add(source_hash)
        eligibility = snapshot['source_eligibility']
        exposure = eligibility['source_exposure']
        exposed = (exposure in ('known_development', 'known_development_document', 'known_development_source')
                   if isinstance(exposure, str) else isinstance(exposure, dict) and
                   (exposure.get('independent') is False or exposure.get('status') in
                    ('known_development', 'known_development_document', 'known_development_source')))
        if (eligibility['independent_source_declared'] is False
                or eligibility['independent_declared'] is False or exposed
                or eligibility['synthetic_fixture_declared']):
            known_development.append(case)
        missing, gold, reference = [], {}, []
        for task in _list(document.get('gold_tasks'), 'gold_tasks'):
            _object(task, 'Reference task')
            gid = _text(task.get('task_id'), 'Reference task_id')
            if gid in gold:
                raise ValueError('Duplicate reference task ID')
            description = _text(task.get('description'), 'Reference task description')
            spans = [_span(item, source, source_hash)
                     for item in _list(task.get('source_spans'), 'Reference source_spans')]
            if not spans:
                raise ValueError('Reference task must have source spans')
            if task.get('task_sha256') != reference_task_sha256(task):
                raise ValueError('Reference task hash mismatch or missing')
            if task.get('supported') is False:
                raise ValueError('Unsupported reference task cannot be gold')
            if task.get('supported') is not None and type(task.get('supported')) is not bool:
                raise ValueError('Reference supported must be boolean or null')
            if task.get('supported') is not True or not review_actor(task, case + ':gold:' + gid):
                missing.append('gold:' + gid)
            gold[gid] = task
            reference.append({'task_id': gid, 'description': description, 'source_spans': spans})
        completion(document.get('reference_review'), 'reference_review', missing)
        completion(document.get('matching_review'), 'matching_review', missing)
        predictions = {item['task_id']: item for item in snapshot['prediction_tasks']}
        dependencies = {item['dependency_id']: item for item in snapshot['prediction_dependencies']}
        accepted_predictions, accepted_gold, pairs = set(), set(), set()
        for match in _list(document.get('matches'), 'matches'):
            _object(match, 'Match')
            pid, gid = match.get('prediction_task_id'), match.get('gold_task_id')
            if pid not in predictions or gid not in gold:
                raise ValueError('Semantic match must use known prediction and reference IDs')
            if match.get('prediction_task_sha256') != predictions[pid]['task_sha256']:
                raise ValueError('Semantic match prediction hash mismatch')
            if match.get('gold_task_sha256') != gold[gid]['task_sha256']:
                raise ValueError('Semantic match reference hash mismatch')
            if (pid, gid) in pairs:
                raise ValueError('Duplicate semantic match pair')
            pairs.add((pid, gid))
            equivalent = match.get('equivalent')
            if equivalent is not None and type(equivalent) is not bool:
                raise ValueError('Semantic equivalence must be boolean or null')
            actor_present = review_actor(match, case + ':match')
            if equivalent is None or not actor_present:
                missing.append('match:' + pid + ':' + gid)
            if equivalent is True:
                if pid in accepted_predictions or gid in accepted_gold:
                    raise ValueError('Accepted semantic matches must be one-to-one')
                accepted_predictions.add(pid)
                accepted_gold.add(gid)
        evidence_metric = decisions(document.get('task_reviews'), predictions, 'task_id', 'task_sha256',
                                    'evidence_correct', 'task_reviews', missing)
        dependency_metric = decisions(document.get('dependency_reviews'), dependencies, 'dependency_id',
                                      'dependency_sha256', 'correct', 'dependency_reviews', missing)
        matching_metric = _prf(len(accepted_predictions), len(predictions) - len(accepted_predictions),
                               len(gold) - len(accepted_gold))
        metrics = {'task_matching': matching_metric, 'evidence_correctness': evidence_metric,
                   'dependency_correctness': dependency_metric} if not missing else None
        all_missing.extend(case + ':' + item for item in missing)
        results.append({'case_id': case, 'document_id': snapshot['document_id'],
            'binding': snapshot['binding'], 'reference_sha256': _digest({'source_sha256': source_hash,
                'gold_tasks': sorted(reference, key=lambda item: item['task_id'])}),
            'source_eligibility': eligibility,
            'status': 'incomplete' if missing else 'complete', 'missing_reviews': missing,
            'counts': {'prediction_tasks': len(predictions), 'reference_tasks': len(gold),
                       'prediction_dependencies': len(dependencies)}, 'metrics': metrics})
    if not results:
        raise ValueError('At least one document is required')
    gate_passed = False
    if kind != 'case_study' and any(reviewers[rid]['kind'] != 'human' for rid in used):
        all_missing.append('scope_gate:all_reviews_must_be_declared_human')
    if kind == 'independent_final_test':
        gate = _object(bundle.get('final_test_gate'), 'final_test_gate')
        gate_actor = review_actor(gate, 'final_test_gate')
        gate_passed = (gate_actor and len(source_hashes) > 1 and not known_development
            and all(gate.get(key) is True for key in
            ('sources_independent', 'selection_frozen_before_review', 'no_test_used_for_selection', 'approved'))
            and all(reviewers[rid]['kind'] == 'human' and reviewers[rid]['independent'] is True for rid in used))
        if not gate_passed:
            all_missing.append('scope_gate:independent_final_test_explicit_approval_required')
        if len(source_hashes) < 2:
            all_missing.append('scope_gate:independent_final_test_requires_multiple_distinct_sources')
        if known_development:
            all_missing.append('scope_gate:known_development_or_synthetic_sources:' + ','.join(known_development))
    complete = not all_missing
    aggregate = None
    if complete:
        task_rows = [item['metrics']['task_matching'] for item in results]
        aggregate = {'task_matching': {'micro': _prf(*(sum(row[key] for row in task_rows)
                     for key in ('tp', 'fp', 'fn'))), 'macro': _macro(task_rows, ('precision', 'recall', 'f1'))}}
        for key in ('evidence_correctness', 'dependency_correctness'):
            rows = [item['metrics'][key] for item in results]
            aggregate[key] = {'micro': _rate(sum(row['correct'] for row in rows), sum(row['total'] for row in rows)),
                              'macro': _macro(rows, ('rate',))}
    elif any(item.startswith('scope_gate:') for item in all_missing):
        for item in results:
            item['metrics'] = None
            item['status'] = 'incomplete'
    kinds = {reviewers[rid]['kind'] for rid in used}
    scope = ('case_study_' + ('mixed_review' if len(kinds) > 1 else next(iter(kinds)) + '_reviewed')
             if kind == 'case_study' and kinds else kind + '_declared')
    return {'version': EVALUATION_VERSION, 'dataset_kind': kind, 'review_bundle_sha256': _digest(bundle),
        'status': 'complete' if complete else 'incomplete', 'review_scope': scope,
        'identity_verification': 'self_declared_not_authenticated',
        'independence_verified_by_software': False, 'final_test_gate_passed': bool(gate_passed and complete),
        'missing_reviews': all_missing, 'documents': results, 'metrics': aggregate,
        'retrieval_relevance': {'status': 'not_evaluated', 'metrics': None},
        'metric_scope': 'Semantic task matching, task evidence support and predicted dependency correctness are separate; no composite quality score.'}
