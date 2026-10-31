"""Verify the fast CLI and extension API in an empty, source-only Python venv.

No pip packages, model files, private documents, caches or existing reports are
copied into the temporary checkout. This checks that a reviewer can run the
authored Markdown demo after cloning; it does not measure learned-model quality.
Evidence is saved in a new output directory, never over an earlier run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import venv


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'extraction.py', 'planning.py', 'psai.py', 'psai_reports.py',
    'tools/demo_project.py', 'tools/verify_extension_demo.py',
    'examples/proposal.md',
)
SOURCE_PACKAGES = (
    'psai_project', 'psai_extension', 'psai_retrieval', 'psai_rules', 'psai_rag',
)


def _require(condition: bool, message: str) -> None:
    # Deliberately not Python assert: verification also works with python -O.
    if not condition:
        raise ValueError(message)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _copy_sources(destination: Path) -> dict:
    paths = [ROOT / relative for relative in SOURCE_FILES]
    for name in SOURCE_PACKAGES:
        directory = ROOT / name
        _require((directory / '__init__.py').is_file(), f'Missing source package: {name}')
        paths.extend(path for path in sorted(directory.rglob('*.py'))
                     if '__pycache__' not in path.parts)
    hashes = {}
    for source in paths:
        _require(source.is_file(), f'Missing runnable source file: {source}')
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = _hash(target.read_bytes())
    return hashes


def _run(command: list[str], *, cwd: Path, env: dict, log: Path) -> None:
    print('Running:', ' '.join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=180)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.returncode:
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f'Command exited {result.returncode}; inspect {log}')


def _check_analysis(directory: Path, raw_input: bytes, source_text: str) -> dict:
    analysis = json.loads((directory / 'analysis.json').read_text('utf-8'))
    metadata = analysis['metadata']
    source_hash = _hash(source_text.encode('utf-8'))
    _require(metadata['input_sha256'] == _hash(raw_input), 'Original input hash differs')
    _require(metadata['source_text_sha256'] == source_hash, 'Normalized source hash differs')
    _require((directory / 'source.txt').read_bytes() == source_text.encode('utf-8'),
             'Exported source snapshot differs')
    for name, expected in (('classifier', 'rules'), ('retrieval_mode', 'bm25'),
                           ('planner_mode', 'evidence')):
        _require(metadata[name] == expected, f'Unexpected {name}: {metadata[name]}')
    _require(metadata['input_unchanged'] is True, 'Input changed during analysis')
    _require(metadata['source_snapshot_unchanged'] is True, 'Source snapshot changed')
    _require([stage['stage'] for stage in metadata['stages']] == ['extract', 'retrieve', 'plan'],
             'The three actual worker stages did not complete in order')
    _require(metadata['human_content_review_complete'] is False,
             'Authored smoke input must not be reported as independent quality review')
    quote_count = 0
    for items in analysis['extraction']['fields'].values():
        for item in items:
            _require(source_text[item['char_start']:item['char_end']] == item['quote'],
                     'Extracted evidence quote differs from exact source span')
            quote_count += 1
    _require(quote_count > 0, 'The demo produced no extracted evidence')
    hits = analysis['retrieval']['hits']
    _require(bool(hits), 'The demo produced no BM25 hits')
    for hit in hits:
        _require(hit['verified'] is True, 'Retrieved evidence is not verified')
        _require(source_text[hit['char_start']:hit['char_end']] == hit['quote'],
                 'Retrieved evidence quote differs from exact source span')
    _require(bool(analysis['plan']['steps']), 'The demo produced no evidence-based plan')
    for name in ('report.html', 'workspace.html'):
        _require((directory / name).stat().st_size > 0, f'Missing HTML export: {name}')
    for name in ('extract', 'retrieve', 'plan'):
        response = json.loads((directory / 'jobs' / f'{name}.response.json').read_text('utf-8'))
        _require(response['status'] == 'completed', f'Actual {name} worker failed')
        _require(response['source_sha256'] == source_hash, f'{name} worker source hash differs')
        _require(response['training_performed'] is False, 'Smoke verification started training')
        _require(all(value is None for value in response['package_versions'].values()),
                 'Smoke verification used a venv containing ML packages')
    return {'input_sha256': metadata['input_sha256'], 'source_sha256': source_hash,
            'exact_extracted_quotes': quote_count, 'verified_bm25_hits': len(hits),
            'plan_steps': len(analysis['plan']['steps']), 'actual_workers_completed': 3}


def verify(output: Path) -> Path:
    output = output.resolve()
    _require(not output.exists(), 'Choose a new output directory; earlier evidence is preserved')
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix='psai-clean-clone-') as scratch:
        temporary = Path(scratch)
        source = temporary / 'source'
        hashes = _copy_sources(source)
        _write_json(output / 'source_files.json', hashes)
        environment_dir = temporary / 'environment'
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment_dir)
        python = environment_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        environment = dict(os.environ)
        for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PSAI_BGE_MODEL_DIR',
                     'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):
            environment.pop(name, None)
        environment.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONNOUSERSITE='1',
                           HF_HOME=str(temporary / 'empty-model-cache'), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        probe = (
            'import importlib.util,json,sys; '
            'names=["numpy","torch","transformers","peft"]; '
            'assert all(importlib.util.find_spec(n) is None for n in names); '
            'import psai_extension.server,psai_project.runtime,psai_retrieval,psai_rules; '
            'assert not any(n in sys.modules for n in names); '
            'print(json.dumps({"python":sys.version,"stdlib_only":True}))'
        )
        _run([str(python), '-c', probe], cwd=source, env=environment,
             log=output / 'environment.log')
        demo_output = output / 'cli'
        _run([str(python), 'tools/demo_project.py', '--fast', '--device', 'cpu',
              '--output', str(demo_output)], cwd=source, env=environment,
             log=output / 'cli.log')
        input_path = source / 'examples/proposal.md'
        raw_input = input_path.read_bytes()
        source_text = input_path.read_text('utf-8-sig')
        cli = _check_analysis(demo_output, raw_input, source_text)
        api_output = output / 'extension_api'
        _run([str(python), 'tools/verify_extension_demo.py', '--input', str(input_path),
              '--output', str(api_output), '--python', str(python), '--timeout', '120'],
             cwd=source, env=environment, log=output / 'extension_api.log')
        api_acceptance = json.loads((api_output / 'acceptance.json').read_text('utf-8'))
        _require(api_acceptance['status'] == 'passed', 'Extension API acceptance failed')
        api = _check_analysis(api_output / 'jobs' / api_acceptance['job_id'] / 'result',
                              raw_input, source_text)
        acceptance = {
            'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'source_files_copied': len(hashes),
            'stdlib_only_environment': True, 'model_files_copied': False,
            'existing_reports_copied': False, 'private_documents_copied': False,
            'cli': cli, 'extension_api': api,
            'api_auth_checked': api_acceptance['api_auth_checked'],
            'result_capability_checked': api_acceptance['result_capability_checked'],
            'scope': 'Actual rules/BM25/evidence demo; no learned-model accuracy claim',
        }
        target = output / 'acceptance.json'
        _write_json(target, acceptance)
    print('Clean-clone verification passed:', target, flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory for verification evidence')
    args = parser.parse_args()
    output = args.output or ROOT / 'outputs/clean_clone' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_') + uuid.uuid4().hex[:8])
    verify(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
