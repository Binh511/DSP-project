"""Freeze and run an explicitly AI-reviewed pilot on unmodified local sources.

The selected production modes remain unchanged. Full failures remain failures;
no automatic fallback, tuning, model downloads or final-test claims are made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_project.execution import offline_environment, runtime_python


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def freeze(sources, references, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Do not replace a frozen pilot protocol')
    sources, references = [Path(path).resolve() for path in sources], [Path(path).resolve() for path in references]
    if (not sources or len({sha(path) for path in sources}) != len(sources)
            or len({path.stem for path in sources}) != len(sources) or not references):
        raise ValueError('Freeze distinct sources and pre-existing reference annotations first')
    from psai_planner.planner import DEFAULT_CONFIG
    from extraction import MIN_SCORE_BY_LABEL
    code = ['extraction.py', 'planning.py', 'ensemble_classifier.py', 'psai_reports.py', 'psai.py',
            'tools/run_external_study.py', 'tools/review_project_quality.py']
    for name in ['psai_project', 'psai_planner', 'psai_retrieval', 'psai_rag']:
        code.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / name).glob('*.py')
                    if not path.name.startswith('test_'))
    models = {}
    for name in ['final_s11', 'final_s22', 'final_s33']:
        for file in ['model.safetensors', 'config.json', 'tokenizer.json', 'inference_config.json']:
            path = ROOT / 'models' / name / file
            models[path.relative_to(ROOT).as_posix()] = sha(path)
    planner_info = json.loads((ROOT / 'reports/project_completion/local_planner_model.json').read_text('utf-8'))
    from psai_retrieval.runtime import DEFAULT_BASE
    rag_info = json.loads((ROOT / 'reports/rag_release_v1/RELEASE_MANIFEST.json').read_text('utf-8'))
    external_models = {}
    for directory, manifest in [(DEFAULT_BASE, rag_info['base_files_sha256']),
                                (Path(planner_info['local_path']), planner_info['files_sha256'])]:
        for name, expected in manifest.items():
            path = directory / name
            actual = sha(path)
            if actual != expected:
                raise ValueError('Local external model differs from its recorded release: ' + str(path))
            external_models[str(path.resolve())] = actual
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'evaluation_kind': 'external_source_AI_reviewed_case_study',
                'human_ground_truth': False, 'independent_final_test': False,
                'source_files_sha256': {str(path): sha(path) for path in sources},
                'reference_files_sha256': {str(path): sha(path) for path in references},
                'code_files_sha256': {name: sha(ROOT / name) for name in sorted(set(code))},
                'production_model_files_sha256': models, 'planner_model_manifest': planner_info,
                'external_model_files_sha256': external_models,
                'base_model_dir': str(DEFAULT_BASE.resolve()),
                'planner_model_dir': str(Path(planner_info['local_path']).resolve()),
                'modes': {'fast': {'classifier': 'rules', 'retrieval': 'bm25', 'planner': 'evidence'},
                          'full': {'classifier': 'ensemble', 'retrieval': 'hybrid', 'planner': 'local'}},
                'planner_config': DEFAULT_CONFIG, 'extraction_thresholds': MIN_SCORE_BY_LABEL,
                'device': 'cuda', 'top_k': 5, 'ocr': 'none',
                'matching_policy': 'Atomic executable proposed engineering changes only; semantic one-to-one matching; grouped predictions cannot collect several true positives',
                'failure_policy': 'Record every requested attempt and diagnostic directory; do not silently replace failed Full with Fast or exclude failures from coverage',
                'ablation_policy': 'Compare catalog/prompt with and without document retrieval using fixed extraction; replay an existing response only when exact prompt/catalog hashes are identical; label replay separately',
                'limitations': ['AI reference and AI adjudication are not independent human ground truth.',
                               'No known development match does not rule out pretraining exposure or unknown histories.',
                               'Small technical-design-source pilot does not establish accuracy on arbitrary project proposals.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, protocol)
    return protocol


def run(protocol_path, output):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol = json.loads(protocol_path.read_text('utf-8'))
    if output.exists():
        raise FileExistsError('Choose a new study output directory')
    bindings = [(Path(path), expected) for key in ['source_files_sha256', 'reference_files_sha256']
                for path, expected in protocol[key].items()]
    bindings += [(ROOT / path, expected) for key in ['code_files_sha256', 'production_model_files_sha256']
                 for path, expected in protocol[key].items()]
    bindings += [(Path(path), expected) for path, expected in protocol['external_model_files_sha256'].items()]
    if any(sha(path) != expected for path, expected in bindings):
        raise ValueError('Source, reference, configuration code or model differs from the frozen protocol')
    output.mkdir(parents=True)
    receipt = {'protocol_sha256': sha(protocol_path), 'started_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'running', 'attempts': [], 'human_ground_truth': False, 'final_test_scored': False}
    (output / 'attempts.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    executable = runtime_python(ROOT)
    for source in protocol['source_files_sha256']:
        source = Path(source)
        for mode, configuration in protocol['modes'].items():
            folder = output / mode / source.stem
            folder.parent.mkdir(exist_ok=True)
            command = [executable, '-u', '-m', 'psai_project', '--input', str(source), '--output', str(folder),
                       '--classifier', configuration['classifier'], '--retrieval', configuration['retrieval'],
                       '--planner', configuration['planner'], '--device', protocol['device'], '--ocr', protocol['ocr'],
                       '--top-k', str(protocol['top_k'])]
            if configuration['retrieval'] in ('dense', 'hybrid'):
                command += ['--base-model-dir', protocol['base_model_dir']]
            if configuration['planner'] == 'local':
                command += ['--planner-model-dir', protocol['planner_model_dir']]
            print('Study:', source.stem, mode, flush=True)
            started = time.monotonic()
            error, returncode = None, None
            try:
                with (folder.parent / (source.stem + '.stdout.log')).open('x', encoding='utf-8') as stdout, \
                        (folder.parent / (source.stem + '.stderr.log')).open('x', encoding='utf-8') as stderr:
                    # Runtime already owns each worker timeout and diagnostics.
                    # Killing only this wrapper would risk orphaning a CUDA worker.
                    process = subprocess.run(command, cwd=ROOT, env=offline_environment(), stdout=stdout, stderr=stderr,
                                             check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    returncode = process.returncode
            except OSError as failure:
                error = type(failure).__name__ + ': ' + str(failure)
            receipt['attempts'].append({'document_id': source.stem, 'source_sha256': sha(source), 'mode': mode,
                'returncode': returncode, 'launch_error': error, 'elapsed_wall_seconds': time.monotonic() - started,
                'output': str(folder), 'report_exists': (folder / 'report.html').is_file(),
                'diagnostic_directories': [str(path) for path in folder.parent.glob(source.stem + '.failed_*')]})
            (output / 'attempts.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt['status'] = 'completed' if all(item['returncode'] == 0 and item['report_exists'] for item in receipt['attempts']) else 'completed_with_failures'
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    receipt['production_weights_unchanged'] = all(sha(ROOT / path) == expected for path, expected in protocol['production_model_files_sha256'].items())
    receipt['external_weights_unchanged'] = all(sha(path) == expected for path, expected in protocol['external_model_files_sha256'].items())
    (output / 'attempts.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freezing = commands.add_parser('freeze')
    freezing.add_argument('--source', type=Path, action='append', required=True)
    freezing.add_argument('--reference', type=Path, action='append', required=True)
    freezing.add_argument('--output', type=Path, required=True)
    running = commands.add_parser('run')
    running.add_argument('--protocol', type=Path, required=True)
    running.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(args.source, args.reference, args.output) if args.command == 'freeze' else run(args.protocol, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
