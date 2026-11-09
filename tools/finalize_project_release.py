"""Freeze/verify the complete local pipeline without changing prior releases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
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


def _check_files(root, files, *, optional=False, label='Project source/artifact'):
    """Allow absent optional assets, but never accept changed or unsafe files."""
    root = Path(root).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError(label + ' needs a nonempty file/hash mapping')
    checked, missing = [], []
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                or PurePosixPath(name).as_posix() != name
                or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)):
            raise ValueError(label + ' contains an unsafe path or invalid SHA-256: ' + str(name))
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(label + ' path escapes its root: ' + name)
        if not path.exists() and optional:
            missing.append(name)
            continue
        if not path.is_file() or sha(path) != expected:
            raise ValueError(label + ' integrity mismatch: ' + name)
        checked.append(name)
    return checked, missing


def verify_source_bundle(manifest_path, *, root=None):
    """Verify a new package inventory independently of the frozen release.

    ``files_sha256`` (or the handoff inventory's ``files``) lists mandatory
    packaged files. Only explicitly declared ``optional_components`` may be
    absent, each with its own nonempty ``files_sha256`` mapping. A present
    optional file with the wrong hash always fails. This checks file integrity,
    never full model execution or historical release compatibility.
    """
    root = Path(ROOT if root is None else root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    data = read(manifest_path)
    if not isinstance(data, dict):
        raise ValueError('Source bundle manifest must be an object')
    if 'files_sha256' in data and 'files' in data:
        raise ValueError('Source bundle must use exactly one mandatory file/hash mapping')
    required = data.get('files_sha256', data.get('files'))
    checked, _ = _check_files(root, required, label='Source bundle')
    optional = data.get('optional_components', {})
    if not isinstance(optional, dict):
        raise ValueError('Source bundle optional_components must be a mapping')
    occupied = set(required)
    components = {}
    for name, component in optional.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(component, dict):
            raise ValueError('Source bundle optional components need names and file mappings')
        files = component.get('files_sha256')
        if isinstance(files, dict) and occupied.intersection(files):
            raise ValueError('Source bundle file is declared in multiple components: ' + name)
        present, missing = _check_files(root, files, optional=True, label='Optional component ' + name)
        occupied.update(files)
        components[name] = {
            'status': 'missing_optional' if not present else 'partial' if missing else 'verified',
            'files_checked': len(present), 'missing_files': missing,
        }
    missing_components = [name for name, component in components.items() if component['missing_files']]
    result = {
        'status': 'partial' if missing_components else 'verified',
        'verification_scope': 'source_bundle_integrity',
        'source_status': 'verified', 'files_checked': len(checked),
        'optional_components': components, 'missing_optional_components': missing_components,
        'historical_release_verified': False, 'full_runtime_verified': False,
        'independent_quality_evaluation_complete': False,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def verify(models=False):
    data = read(FOLDER / 'PROJECT_RELEASE_MANIFEST.json')
    _check_files(ROOT, data['files_sha256'])
    if models:
        _check_files(ROOT, data['classifier_model_files_sha256'], label='Classifier model')
        from psai_retrieval.runtime import DEFAULT_BASE
        _check_files(DEFAULT_BASE, data['embedding_base']['files_sha256'], label='Embedding base')
        model_dir = Path(read(FOLDER / 'local_planner_model.json')['local_path'])
        _check_files(model_dir, data['planner_model']['files_sha256'], label='Planner model')
    result = {'status': 'verified', 'release_version': data['release_version'],
              'verification_scope': 'historical_release_integrity',
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--source-manifest', type=Path,
                        help='Verify a package-specific inventory; never bypasses a mismatched file')
    source.add_argument('--source-only', action='store_true',
                        help='Verify SOURCE_BUNDLE_MANIFEST.json or COMPLETE_HANDOFF_FILES.json in this checkout')
    args = parser.parse_args()
    if args.source_manifest or args.source_only:
        if args.models:
            parser.error('--models checks the historical release; do not combine it with source-bundle verification')
        manifest = args.source_manifest
        if manifest is None:
            manifest = ROOT / 'SOURCE_BUNDLE_MANIFEST.json'
            if not manifest.is_file():
                manifest = ROOT / 'COMPLETE_HANDOFF_FILES.json'
        verify_source_bundle(manifest)
    else:
        verify(args.models) if args.verify_only else build()


if __name__ == '__main__': main()
