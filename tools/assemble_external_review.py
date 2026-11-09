"""Bind explicit AI judgments to frozen sources and score using the public evaluator."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.review_project_quality import prepare, read, score, write_new


def assemble(study, mode, decisions_path, output):
    study = Path(study).resolve()
    decisions_path, output = Path(decisions_path).resolve(), Path(output).resolve()
    protocol_path, reference_path = study / 'protocol_v1.json', study / 'references/rater_b.json'
    decisions, reference = read(decisions_path), read(reference_path)
    actual_protocol = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    actual_reference = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if (decisions.get('mode') != mode or decisions['protocol_sha256'] != actual_protocol
            or decisions['reference_file_sha256'] != actual_reference):
        raise ValueError('AI decisions do not match the frozen protocol/reference')
    reference_by_hash = {item['source_sha256']: item for item in reference['documents']}
    decision_by_hash = {item['source_sha256']: item for item in decisions['documents']}
    if len(decision_by_hash) != len(decisions['documents']) or set(decision_by_hash) != set(reference_by_hash):
        raise ValueError('Decisions must cover every source exactly once')
    protocol = read(protocol_path)
    runs = [study / 'runs_v1' / mode / Path(source).stem for source in protocol['source_files_sha256']]
    prepare(runs, output, dataset_kind='case_study')
    bundle = read(output / 'review.json')
    bundle['reviewers'] = [{'reviewer_id': 'AI_B', 'kind': 'ai', 'independent': False},
                           {'reviewer_id': 'AI_C', 'kind': 'ai', 'independent': False}]
    bundle['primary_reference_sha256'] = actual_reference
    bundle['decision_file_sha256'] = hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    bundle['scope'] = 'Externally sourced technical-design pilot, explicitly AI-annotated and AI-matched; no human final test'
    for document in bundle['documents']:
        source_hash = document['binding']['source_sha256']
        manual, gold = decision_by_hash[source_hash], reference_by_hash[source_hash]
        if manual['binding'] != document['binding'] or manual['prediction_tasks'] != document['prediction_tasks']:
            raise ValueError('Manual review is bound to a different original prediction run')
        document['gold_tasks'] = deepcopy(gold['gold_tasks'])
        document['matches'] = deepcopy(manual['matches'])
        document['task_reviews'] = deepcopy(manual['task_reviews'])
        document['dependency_reviews'] = deepcopy(manual['dependency_reviews'])
        document['retrieval_reviews'] = deepcopy(manual['retrieval_reviews'])
        document['reference_review'] = {'status': 'complete', 'reviewer_id': 'AI_B',
            'rationale': 'Complete primary source-only AI reference fixed before predictions; not human gold. Atomic accepted-design target scope retained unchanged.'}
        document['matching_review'] = {'status': 'complete', 'reviewer_id': 'AI_C',
            'rationale': 'Explicit manual AI one-to-one matching against the fixed reference. Every output and every unmatched reference considered; no gold changed to improve scores.'}
    write_new(output / 'reviewed.json', bundle)
    result = score(output / 'reviewed.json')
    result['accepted_task_evidence_support_rubric'] = decisions['evidence_rubric']
    result['literal_source_matching_is_separate'] = True
    result['literal_source_matches'] = sum(item.get('literal_source_match') is True
                                         for doc in decisions['documents'] for item in doc['task_reviews'])
    write_new(output / 'score.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--mode', choices=['fast', 'full'], required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble(args.study, args.mode, args.decisions, args.output)
    print(json.dumps({'status': result['status'], 'tasks': result['metrics']['task_matching']['micro']
                      if result['metrics'] else None}, indent=2))


if __name__ == '__main__':
    main()
