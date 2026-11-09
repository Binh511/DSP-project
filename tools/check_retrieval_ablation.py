"""Check a fixed-extraction document-retrieval ablation without extra inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_planner.planner import build_catalog, build_prompt, validate_response
from psai_project.evaluation import snapshot_run


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def compare_catalogs(record, text, retrieval, config=None):
    with_retrieval = build_catalog(record, text, retrieval, config)
    without_retrieval = build_catalog(record, text, None, config)
    messages_with, messages_without = build_prompt(with_retrieval), build_prompt(without_retrieval)
    return {'with_retrieval_catalog_sha256': digest(with_retrieval),
            'without_retrieval_catalog_sha256': digest(without_retrieval),
            'with_retrieval_prompt_sha256': digest(messages_with),
            'without_retrieval_prompt_sha256': digest(messages_without),
            'catalog_identical': with_retrieval == without_retrieval,
            'prompt_identical': messages_with == messages_without,
            'tasks_with': len(with_retrieval['tasks']), 'tasks_without': len(without_retrieval['tasks'])}, without_retrieval


def check_saved_run(run):
    run = Path(run)
    binding = snapshot_run(run)['binding']
    analysis = json.loads((run / 'analysis.json').read_text('utf-8'))
    plan = analysis['plan']
    methods = {'local_llm_evidence_catalog_v1': 'psai-local-evidence-catalog-v1',
               'local_llm_evidence_catalog_v2': 'psai-local-evidence-catalog-v2'}
    if plan.get('planning_method') not in methods:
        raise ValueError('Ablation replay requires an actual saved local planner output')
    text = (run / 'source.txt').read_bytes().decode('utf-8')
    retrieval = analysis['retrieval']
    snapshots = {key: (run / info['artifact']).read_bytes().decode('utf-8')
                 for key, info in retrieval.get('sources', {}).items()}
    result, no_retrieval_catalog = compare_catalogs(analysis['extraction'], text,
        {**retrieval, 'source_texts': snapshots},
        {**plan['provenance']['config'], 'protocol_version': methods[plan['planning_method']]})
    if (result['with_retrieval_catalog_sha256'] != plan['provenance']['catalog_sha256']
            or result['with_retrieval_prompt_sha256'] != plan['provenance']['prompt_sha256']):
        raise ValueError('Actual saved planner provenance differs from reconstructed input')
    result.update({'document_id': analysis['document_id'], 'binding': binding,
        'retrieval_hits': len(retrieval.get('hits', [])),
        'external_resource_hits': sum(hit.get('role') == 'resource' for hit in retrieval.get('hits', [])),
        'model_loaded_in_original_run': plan['provenance']['model_loaded'],
        'new_model_inference_performed': False,
        'scope': 'Structural fixed-extraction catalog/prompt ablation and existing-response validation replay; no second inference or latency measurement'})
    if result['catalog_identical'] and result['prompt_identical'] and plan.get('raw_response') is not None:
        replay = validate_response(plan['raw_response'], no_retrieval_catalog)
        result['replay_selected_ids'] = [item['task_id'] for item in replay['steps']]
        result['selected_ids_identical'] = replay['steps'] == plan['steps']
        result['dependencies_identical'] = replay['dependency_candidates'] == plan['dependency_candidates']
    else:
        result['selected_ids_identical'] = None
        result['replay_status'] = 'not_replayed_changed_prompt_or_no_model_response'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = {'kind': 'structural_retrieval_ablation', 'human_quality_validation': False,
              'documents': [check_saved_run(path) for path in args.run],
              'note': 'Protocol v1 excluded document-only hits; v2 admits bounded source-verified document context. '
                      'Prompt changes alone do not prove better planning.'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps([{'document_id': doc['document_id'], 'prompt_identical': doc['prompt_identical']}
                      for doc in result['documents']], indent=2))


if __name__ == '__main__':
    main()
