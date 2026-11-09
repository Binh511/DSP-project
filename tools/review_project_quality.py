"""Prepare and score explicit human reviews of saved PSAI outputs, offline.

No model is loaded, no reference is inferred, and no frozen release is edited.
CSV files are editable decisions; source/prediction identities stay hash-bound.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CSV_TEXT_PREFIX = 'PSAI_TEXT:'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_project.evaluation import build_evaluation_bundle, evaluate_bundle, reference_task_sha256


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_csv(path, columns, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({key: csv_cell(value) for key, value in row.items()} for row in rows)


def csv_cell(value):
    """Keep source text lossless while preventing spreadsheet formula evaluation."""
    if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(CSV_TEXT_PREFIX)):
        return CSV_TEXT_PREFIX + json.dumps(value, ensure_ascii=False)
    return value


def decoded_cell(value):
    if isinstance(value, str) and value.startswith(CSV_TEXT_PREFIX):
        decoded = json.loads(value[len(CSV_TEXT_PREFIX):])
        if not isinstance(decoded, str):
            raise ValueError('Encoded CSV text must decode to a string')
        return decoded
    return value


def boolean(value):
    if value in (None, ''):
        return None
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Decision cells must be true, false or blank')


def prepare(run_dirs, output, *, dataset_kind='case_study'):
    """Copy immutable run evidence and export pending review tables."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new review folder; existing decisions are preserved')
    # Reject broken inputs before writing any packet or changing source runs.
    bundle = build_evaluation_bundle(run_dirs, dataset_kind=dataset_kind)
    hashes = [doc['binding']['source_sha256'] for doc in bundle['documents']]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Use one saved run per source in each configuration packet')
    configurations = set()
    for doc in bundle['documents']:
        metadata = read(Path(doc['run_dir']) / 'analysis.json').get('metadata', {})
        configurations.add(tuple(metadata.get(key) for key in ('classifier', 'retrieval_mode', 'planner_mode')))
    if len(configurations) != 1:
        raise ValueError('Prepare each pipeline configuration in a separate review packet')
    output.mkdir(parents=True)
    for document in bundle['documents']:
        original = Path(document['run_dir'])
        destination = output / 'runs' / document['case_id']
        destination.mkdir(parents=True)
        required = {'source.txt', 'analysis.json', *document['manifest_files']}
        analysis = read(original / 'analysis.json')
        for info in analysis.get('retrieval', {}).get('sources', {}).values():
            required.add(info['artifact'])
        for name in sorted(required):
            source = (original / name).resolve()
            target = (destination / name).resolve()
            if not source.is_relative_to(original.resolve()) or not target.is_relative_to(destination):
                raise ValueError('Source snapshot path escapes packet directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ['jobs', 'original']:
            if (original / name).is_dir():
                for source in (original / name).rglob('*'):
                    if source.is_file() and source.suffix in {'.json', '.log', '.txt'}:
                        if not source.resolve().is_relative_to(original.resolve()):
                            raise ValueError('Saved log path escapes source run')
                        target = destination / source.relative_to(original)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, target)
        document['original_run_dir'] = str(original)
        document['run_dir'] = destination.relative_to(output).as_posix()
        document['retrieval_reviews'] = [
            {'hit_id': str(number), 'hit_sha256': digest(hit), 'relevant': None,
             'source_support_correct': None, 'reviewer_id': '', 'rationale': ''}
            for number, hit in enumerate(analysis.get('retrieval', {}).get('hits', []), 1)]
    bundle['prepared_utc'] = datetime.now(timezone.utc).isoformat()
    bundle['preparation_scope'] = 'Retrospective saved-output review; no human or independence approvals inferred'
    write_new(output / 'review.json', bundle)
    task_rows, dep_rows, document_rows, retrieval_rows = [], [], [], []
    for document in bundle['documents']:
        case = document['case_id']
        predictions = {item['task_id']: item for item in document['prediction_tasks']}
        for row in document['task_reviews']:
            item = predictions[row['task_id']]
            task_rows.append({**row, 'case_id': case, 'description': item['description'],
                              'evidence_correct': '', 'source_spans': json.dumps(item['source_spans'], ensure_ascii=False)})
        dep_rows.extend({**row, 'case_id': case, 'correct': ''} for row in document['dependency_reviews'])
        document_rows.append({'case_id': case, 'document_id': document['document_id'],
                              'reference_status': 'pending', 'matching_status': 'pending'})
        hits = read(output / document['run_dir'] / 'analysis.json').get('retrieval', {}).get('hits', [])
        for row, hit in zip(document['retrieval_reviews'], hits):
            retrieval_rows.append({**row, 'case_id': case, 'quote': hit['quote'], 'relevant': '',
                                   'source_support_correct': '', 'role': hit.get('role', ''),
                                   'queries': json.dumps([item.get('query') for item in hit.get('query_matches', [])], ensure_ascii=False)})
    write_csv(output / 'reviewers.csv', ['reviewer_id', 'kind', 'independent'], [])
    write_csv(output / 'gold_tasks.csv', ['case_id', 'task_id', 'description', 'char_start', 'char_end',
                                         'quote', 'supported', 'reviewer_id', 'rationale'], [])
    write_csv(output / 'task_matches.csv', ['case_id', 'prediction_task_id', 'prediction_task_sha256',
                                           'gold_task_id', 'gold_task_sha256', 'equivalent', 'reviewer_id', 'rationale'], [])
    write_csv(output / 'task_reviews.csv', ['case_id', 'task_id', 'task_sha256', 'description', 'source_spans',
                                           'evidence_correct', 'reviewer_id', 'rationale'], task_rows)
    write_csv(output / 'dependency_reviews.csv', ['case_id', 'dependency_id', 'dependency_sha256',
                                                 'correct', 'reviewer_id', 'rationale'], dep_rows)
    write_csv(output / 'document_reviews.csv', ['case_id', 'document_id', 'reference_status', 'reference_reviewer_id',
                                               'reference_rationale', 'matching_status', 'matching_reviewer_id',
                                               'matching_rationale'], document_rows)
    write_csv(output / 'retrieval_reviews.csv', ['case_id', 'hit_id', 'hit_sha256', 'quote', 'role', 'queries',
                                                'relevant', 'source_support_correct', 'reviewer_id', 'rationale'], retrieval_rows)
    write_new(output / 'pending_score.json', score(output / 'review.json'))
    write_new(output / 'source_screening.json', screen_sources(bundle, output))
    return {'review_packet': str(output), 'documents': len(bundle['documents']), 'status': 'pending_human_review'}


def import_csv(packet, output):
    """Import explicit decisions into a new JSON, preserving the original packet."""
    packet = Path(packet).resolve()
    bundle = deepcopy(read(packet))
    documents = {item['case_id']: item for item in bundle['documents']}
    if len(documents) != len(bundle['documents']):
        raise ValueError('Duplicate document case ID')
    def rows(name):
        with (packet.parent / name).open(encoding='utf-8-sig', newline='') as stream:
            return [{key: decoded_cell(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    def document(row):
        if row.get('case_id') not in documents:
            raise ValueError('Unknown case ID: ' + str(row.get('case_id')))
        return documents[row['case_id']]
    bundle['reviewers'] = [{**row, 'independent': boolean(row['independent'])}
                            for row in rows('reviewers.csv')]
    for item in documents.values():
        item['gold_tasks'], item['matches'], item['task_reviews'], item['dependency_reviews'] = [], [], [], []
        item['retrieval_reviews'] = []
        item['reference_review'] = {'status': 'pending', 'reviewer_id': '', 'rationale': ''}
        item['matching_review'] = {'status': 'pending', 'reviewer_id': '', 'rationale': ''}
    gold_index = {}
    for row in rows('gold_tasks.csv'):
        doc = document(row)
        key = (row['case_id'], row['task_id'])
        task = {'task_id': row['task_id'], 'description': row['description'], 'supported': boolean(row['supported']),
                'reviewer_id': row['reviewer_id'], 'rationale': row['rationale']}
        if key not in gold_index:
            gold_index[key] = {**task, 'source_spans': []}
            doc['gold_tasks'].append(gold_index[key])
        elif task != {key_: value for key_, value in gold_index[key].items() if key_ != 'source_spans'}:
            raise ValueError('Conflicting rows for the same gold task')
        gold_index[key]['source_spans'].append({'char_start': int(row['char_start']), 'char_end': int(row['char_end']),
            'quote': row['quote'], 'source_sha256': doc['binding']['source_sha256']})
    for task in gold_index.values():
        task['task_sha256'] = reference_task_sha256(task)
    for row in rows('task_matches.csv'):
        document(row)['matches'].append({key: boolean(value) if key == 'equivalent' else value
                                        for key, value in row.items() if key != 'case_id'})
    for name, list_name, decision, keys in [
        ('task_reviews.csv', 'task_reviews', 'evidence_correct', ['task_id', 'task_sha256', 'reviewer_id', 'rationale']),
        ('dependency_reviews.csv', 'dependency_reviews', 'correct', ['dependency_id', 'dependency_sha256', 'reviewer_id', 'rationale']),
        ('retrieval_reviews.csv', 'retrieval_reviews', 'relevant', ['hit_id', 'hit_sha256', 'reviewer_id', 'rationale']),
    ]:
        for row in rows(name):
            value = {key: row[key] for key in keys}
            value[decision] = boolean(row[decision])
            if list_name == 'retrieval_reviews':
                value['source_support_correct'] = boolean(row['source_support_correct'])
            document(row)[list_name].append(value)
    seen = set()
    for row in rows('document_reviews.csv'):
        doc = document(row)
        if row['case_id'] in seen:
            raise ValueError('Duplicate document review')
        seen.add(row['case_id'])
        for kind in ['reference', 'matching']:
            doc[kind + '_review'] = {'status': row[kind + '_status'], 'reviewer_id': row[kind + '_reviewer_id'],
                                     'rationale': row[kind + '_rationale']}
    if seen != set(documents):
        raise ValueError('Document review table must cover each case exactly once')
    result = score_data(bundle, packet.parent)
    # Relative source paths remain anchored to the review packet folder.
    if Path(output).resolve().parent != packet.parent:
        raise ValueError('Reviewed JSON must stay beside its immutable review packet')
    write_new(output, bundle)
    write_csv(Path(output).with_suffix('.gold_catalog.csv'),
              ['case_id', 'task_id', 'task_sha256', 'description'],
              [{**task, 'case_id': doc['case_id']} for doc in bundle['documents'] for task in doc['gold_tasks']])
    return result


def score_data(bundle, base):
    result = evaluate_bundle(bundle, base_dir=base)
    result['task_review_status'] = result['status']
    reviewers = {item['reviewer_id']: item for item in bundle['reviewers']}
    checked, relevance, support, missing = 0, 0, 0, []
    for doc in bundle['documents']:
        run = (Path(base) / doc['run_dir']).resolve()
        hits = read(run / 'analysis.json').get('retrieval', {}).get('hits', [])
        expected = {str(number): digest(hit) for number, hit in enumerate(hits, 1)}
        decisions = {}
        for row in doc.get('retrieval_reviews', []):
            identity = row['hit_id']
            if identity in decisions or identity not in expected or row['hit_sha256'] != expected[identity]:
                raise ValueError('Retrieval decision is duplicated or differs from the saved hit')
            decisions[identity] = row
        for identity in expected:
            row = decisions.get(identity, {})
            reviewer = reviewers.get(row.get('reviewer_id'))
            if (type(row.get('relevant')) is not bool or type(row.get('source_support_correct')) is not bool
                    or reviewer is None or not isinstance(row.get('rationale'), str) or not row['rationale'].strip()):
                missing.append(doc['case_id'] + ': retrieval hit ' + identity)
                continue
            if (bundle['dataset_kind'] != 'case_study' and reviewer['kind'] != 'human'
                    or bundle['dataset_kind'] == 'independent_final_test' and reviewer.get('independent') is not True):
                missing.append(doc['case_id'] + ': retrieval requires human reviewer')
                continue
            if reviewer['kind'] == 'ai':
                result['review_scope'] = 'case_study_ai_or_mixed_reviewed'
            checked += 1
            relevance += row['relevant']
            support += row['source_support_correct']
    result['retrieval_relevance'] = {
        'status': 'incomplete' if missing else 'complete' if checked else 'not_applicable',
        'reviewed_hits': checked, 'missing_reviews': missing,
        'metrics': None if missing or not checked else {
            'relevant_hits': relevance, 'supported_hits': support, 'total': checked,
            'returned_hit_relevance_rate': relevance / checked, 'source_support_rate': support / checked},
        'scope': 'Returned-hit judgments only; no retrieval Recall@k without exhaustive relevance references',
    }
    if missing:
        result['status'] = 'incomplete'
        result['missing_reviews'].extend(missing)
    if bundle['dataset_kind'] == 'independent_final_test':
        screening = screen_sources(bundle, base)
        result['source_screening'] = screening
        rejected = [row['case_id'] for row in screening['documents'] if row['status'] == 'known_development_source']
        if rejected or screening['read_errors']:
            result['status'] = 'incomplete'
            result['missing_reviews'].append('Independent-source gate: known exposure or incomplete development inventory')
        if result['status'] != 'complete':
            result['metrics'] = None
            result['final_test_gate_passed'] = False
            result['retrieval_relevance']['metrics'] = None
            for document in result['documents']:
                document['metrics'] = None
    return result


def score(packet):
    packet = Path(packet).resolve()
    return score_data(read(packet), packet.parent)


def compare(packets):
    results = [score(packet) for packet in packets]
    if len(results) != 2 or any(item['status'] != 'complete' for item in results):
        raise ValueError('Comparison requires exactly two fully reviewed packets')
    def index(result):
        entries = result['documents']
        by_source = {item['binding']['source_sha256']: item for item in entries}
        if len(by_source) != len(entries):
            raise ValueError('Each configuration must have one run per source')
        return by_source
    left, right = (index(item) for item in results)
    if set(left) != set(right) or any(left[key]['reference_sha256'] != right[key]['reference_sha256'] for key in left):
        raise ValueError('Paired comparison requires identical sources and reference tasks')
    if all(left[key]['binding'] == right[key]['binding'] for key in left):
        raise ValueError('Comparison requires two distinct saved configurations, not the same bound outputs twice')
    return {'status': 'complete', 'paired_sources': len(left), 'configurations': results,
            'scope': 'Paired saved-output comparison; changes in several stages do not isolate the effect of RAG'}


def screen_sources(bundle, base):
    """Known-exposure screen only. Absence of a match never grants independence."""
    import re
    import unicodedata
    def normalized(text):
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()
    table = ROOT / 'data_segment_cls/segment_classification.csv'
    ids, inputs, errors, sources = set(), {}, [], {}
    if table.is_file():
        inputs[table.relative_to(ROOT).as_posix()] = hashlib.sha256(table.read_bytes()).hexdigest()
        try:
            with table.open(encoding='utf-8-sig', newline='') as stream:
                ids = {row['document_id'] for row in csv.DictReader(stream)}
        except (KeyError, UnicodeError, csv.Error) as error:
            errors.append(str(error))
    else:
        errors.append('Training CSV not available in this source bundle')
    for folder in ['data_cleaned', 'data_labeled', 'examples']:
        for path in (ROOT / folder).rglob('*.md'):
            try:
                raw = path.read_bytes()
                relative = path.relative_to(ROOT).as_posix()
                inputs[relative] = hashlib.sha256(raw).hexdigest()
                key = hashlib.sha256(normalized(raw.decode('utf-8-sig')).encode('utf-8')).hexdigest()
                sources.setdefault(key, []).append(relative)
            except (OSError, UnicodeError) as error:
                errors.append(str(path) + ': ' + str(error))
    results = []
    for doc in bundle['documents']:
        path = Path(base) / doc['run_dir']
        source = (path / 'source.txt').read_text('utf-8')
        key = hashlib.sha256(normalized(source).encode('utf-8')).hexdigest()
        matches = sources.get(key, [])
        metadata = read(path / 'analysis.json').get('metadata', {})
        exposure = metadata.get('source_exposure')
        known_exposure = (exposure in ('known_development', 'known_development_document') if isinstance(exposure, str)
                          else isinstance(exposure, dict) and
                          (exposure.get('independent') is False or str(exposure.get('status', '')).startswith('known_development')))
        exposed = bool(doc['document_id'] in ids or matches or metadata.get('independent') is False
                       or metadata.get('independent_source') is False or known_exposure)
        results.append({'case_id': doc['case_id'], 'document_id': doc['document_id'],
                        'source_sha256': doc['binding']['source_sha256'],
                        'status': 'known_development_source' if exposed else 'independence_unconfirmed',
                        'training_document_id_match': doc['document_id'] in ids, 'normalized_full_source_matches': matches,
                        'approved_for_final_test': False})
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'documents': results,
            'development_inventory_sha256': digest(inputs), 'inventory_files': inputs, 'read_errors': errors,
            'independence_certified': False,
            'scope': 'Known ID and normalized whole-source checks only; near-duplicates and historical exposure require human review'}


def readiness(*, runtime=None, base_model_dir=None, planner_model_dir=None):
    from psai_project.execution import runtime_python
    from psai_retrieval.runtime import DEFAULT_BASE
    executable = runtime_python(ROOT, runtime)
    probe = ('import importlib.metadata,json,sys; names=["numpy","torch","transformers","sentencepiece","peft"]; '
             'out={};\nfor n in names:\n try: out[n]=importlib.metadata.version(n)\n'
             ' except importlib.metadata.PackageNotFoundError: out[n]=None\n'
             'print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":out}))')
    completed = subprocess.run([executable, '-c', probe], capture_output=True, text=True, check=False,
                               timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode:
        raise ValueError('Runtime metadata probe failed: ' + completed.stderr[-1000:])
    environment = json.loads(completed.stdout)
    directories = {name: ROOT / 'models' / name for name in ['final_s11', 'final_s22', 'final_s33']}
    directories['bge_m3'] = Path(base_model_dir) if base_model_dir else DEFAULT_BASE
    planner_manifest = ROOT / 'reports/project_completion/local_planner_model.json'
    directories['qwen'] = (Path(planner_model_dir) if planner_model_dir else
                           Path(read(planner_manifest)['local_path']) if planner_manifest.is_file() else ROOT / 'models/MISSING_PLANNER')
    assets = {}
    for name, directory in directories.items():
        required = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
        if name.startswith('final_'):
            required.append('inference_config.json')
        missing = [item for item in required if not (directory / item).is_file()]
        weights = (list(directory.glob('*.safetensors')) + list(directory.glob('pytorch_model*.bin')))
        if not weights:
            missing.append('model weights (*.safetensors or pytorch_model*.bin)')
        assets[name] = {'directory': str(directory.resolve()), 'missing_files': missing,
                        'file_presence_ready': not missing, 'model_load_verified': False}
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment,
            'assets': assets, 'file_presence_ready': all(item['file_presence_ready'] for item in assets.values()),
            'dependency_metadata_ready': all(environment['packages'].get(name) for name in ['numpy', 'torch', 'transformers', 'sentencepiece']),
            'full_runtime_verified': False, 'extension_session_pairing_required': True,
            'scope': 'Offline file/package metadata only; CUDA, tokenizer compatibility, inference and PDF OCR require separate execution'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    create.add_argument('--run', type=Path, action='append', required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--dataset-kind', choices=['case_study', 'human_review', 'independent_final_test'], default='case_study')
    import_ = commands.add_parser('import-csv')
    import_.add_argument('--packet', type=Path, required=True)
    import_.add_argument('--output', type=Path, required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--packet', type=Path, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    paired = commands.add_parser('compare')
    paired.add_argument('--packet', type=Path, action='append', required=True)
    paired.add_argument('--output', type=Path, required=True)
    ready = commands.add_parser('readiness')
    ready.add_argument('--output', type=Path, required=True)
    ready.add_argument('--runtime')
    ready.add_argument('--base-model-dir', type=Path)
    ready.add_argument('--planner-model-dir', type=Path)
    screen = commands.add_parser('screen-sources')
    screen.add_argument('--packet', type=Path, required=True)
    screen.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.run, args.output, dataset_kind=args.dataset_kind)
        elif args.command == 'import-csv':
            result = import_csv(args.packet, args.output)
        elif args.command == 'score':
            result = score(args.packet)
            write_new(args.output, result)
        elif args.command == 'compare':
            result = compare(args.packet)
            write_new(args.output, result)
        elif args.command == 'screen-sources':
            result = screen_sources(read(args.packet), args.packet.resolve().parent)
            write_new(args.output, result)
        else:
            result = readiness(runtime=args.runtime, base_model_dir=args.base_model_dir, planner_model_dir=args.planner_model_dir)
            write_new(args.output, result)
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
