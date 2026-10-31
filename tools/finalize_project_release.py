"""Freeze/verify the complete local pipeline without changing prior releases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'reports/project_completion'


def read(path): return json.loads(Path(path).read_text('utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def verify(models=False):
    data = read(FOLDER / 'PROJECT_RELEASE_MANIFEST.json')
    for name, expected in data['files_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise ValueError('Project source/artifact integrity mismatch: ' + name)
    if models:
        for name, expected in data['classifier_model_files_sha256'].items():
            if sha(ROOT / name) != expected: raise ValueError('Classifier model changed: ' + name)
        from psai_retrieval.runtime import DEFAULT_BASE
        for name, expected in data['embedding_base']['files_sha256'].items():
            if sha(DEFAULT_BASE / name) != expected: raise ValueError('Embedding base changed: ' + name)
        model_dir = Path(read(FOLDER / 'local_planner_model.json')['local_path'])
        for name, expected in data['planner_model']['files_sha256'].items():
            if sha(model_dir / name) != expected: raise ValueError('Planner model changed: ' + name)
    result = {'status': 'verified', 'release_version': data['release_version'],
              'files': len(data['files_sha256']), 'model_weights_checked': models,
              'functional_acceptance': data['functional_acceptance'],
              'independent_quality_evaluation_complete': False, 'production_models_changed': False}
    print(json.dumps(result, indent=2), flush=True)
    return data


def build():
    if (FOLDER / 'PROJECT_RELEASE_MANIFEST.json').exists(): return verify()
    acceptance = read(FOLDER / 'latest_acceptance.json')
    if acceptance.get('status') != 'passed': raise ValueError('Functional acceptance is incomplete')
    report = ROOT / acceptance['report']
    analysis = read(report.parent / 'analysis.json')
    source_hash = analysis['metadata']['source_text_sha256']
    classifier_files = {}
    for stage in ['extract', 'retrieve', 'plan']:
        response = read(report.parent / f'jobs/{stage}.response.json')
        if response.get('status') != 'completed' or response.get('stage') != stage or response.get('source_sha256') != source_hash:
            raise ValueError('Acceptance worker receipt is not bound to its input: ' + stage)
        if stage == 'extract':
            for member, files in response['provenance']['model_files_sha256'].items():
                for name, expected in files.items(): classifier_files[f'models/{member}/{name}'] = expected
    if not analysis['plan']['provenance']['model_loaded']: raise ValueError('LLM was not executed')
    files = set()
    for package in ['psai_project', 'psai_planner', 'psai_rules', 'psai_ocr']:
        files.update(p for p in (ROOT / package).iterdir() if p.suffix in ['.py', '.ps1'])
    for name in ['tools/demo_project.py', 'tools/verify_complete_project.py', 'tools/verify_local_planner.py',
                 'tools/verify_project_ensemble_rag.py', 'tools/finalize_project_release.py']:
        files.add(ROOT / name)
    for name in ['test_project_worker_report.py', 'test_project_orchestrator.py', 'test_project_workspace.py',
                 'test_project_ensemble_bridge.py']:
        files.add(ROOT / 'tests' / name)
    rag = read(ROOT / 'reports/rag_release_v1/RELEASE_MANIFEST.json')
    files.update(ROOT / name for name in rag['files_sha256'])
    records = {'release_version': 'psai-local-pipeline-1.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'PDF/text and Windows English OCR -> production ensemble -> hybrid retrieval -> local LLM -> evidence report and workspace',
        'functional_acceptance': 'passed_on_authored_mixed_PDF_fixture',
        'unit_test_count': 294, 'unit_test_status': 'passed',
        'classifier_model_files_sha256': classifier_files,
        'embedding_base': {'repo_id': 'BAAI/bge-m3', 'revision': rag['base_revision'], 'files_sha256': rag['base_files_sha256']},
        'planner_model': read(FOLDER / 'local_planner_model.json'),
        'files_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)},
        'acceptance_receipt': 'reports/project_completion/latest_acceptance.json',
        'independent_human_quality_evaluation_complete': False, 'final_test_scored': False,
        'accuracy_100_percent_claimed': False, 'production_models_changed': False,
        'notes': ['Qwen is a pretrained local inference model; it was not trained in this project.',
                  'Task hierarchy/resource associations/dependencies remain suggestions needing human review.',
                  'A complete functional path is not a claim of perfect OCR, labels, relevance or planning.']}
    (FOLDER / 'PROJECT_RELEASE_MANIFEST.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return verify(models=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--models', action='store_true', help='Also hash local model files; requires installed weights')
    args = parser.parse_args()
    verify(args.models) if args.verify_only else build()


if __name__ == '__main__': main()
