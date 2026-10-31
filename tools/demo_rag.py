"""Run a source-cited RAG demo; no training, download, or production replacement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def _verified_adapter() -> Path:
    portable = ROOT / 'models/rag/release_v1/retriever_adapter'
    release_path = ROOT / 'reports/rag_release_v1/RELEASE_MANIFEST.json'
    if portable.is_dir() and release_path.is_file():
        manifest = json.loads(release_path.read_text('utf-8'))
        for name in ['adapter_model.safetensors', 'adapter_config.json', 'base_provenance.json']:
            path = portable / name
            expected = manifest['files_sha256'].get(path.relative_to(ROOT).as_posix())
            if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Portable adapter integrity check failed: ' + name)
        return portable
    folder = ROOT / 'models/rag/retrieval_genq_lora_v1'
    receipt = json.loads((folder / 'training_receipt.json').read_text('utf-8'))
    if receipt.get('status') != 'completed':
        raise ValueError('The pilot adapter has not finished training')
    path = Path(receipt['adapter_dir']).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('Adapter receipt refers outside the retrieval pilot directory')
    commit = json.loads((path / 'commit.json').read_text('utf-8'))
    for name, expected in commit['files_sha256'].items():
        file = (path / name).resolve()
        if file.parent != path or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Pilot checkpoint integrity check failed: ' + name)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/proposal.md')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['bm25', 'dense', 'hybrid'], default='bm25')
    parser.add_argument('--query', action='append')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--resource-dir', type=Path)
    parser.add_argument('--base-model-dir', type=Path)
    parser.add_argument('--use-adapter', action='store_true', help='Explicitly use the experimental retrieval LoRA pilot')
    parser.add_argument('--rerank', action='store_true', help='Use the pinned local English cross-encoder')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if args.use_adapter and args.mode == 'bm25':
        parser.error('--use-adapter requires --mode dense or hybrid')
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runtime = ROOT / '.venv/psai_runtime/Scripts/python.exe'
    python = str(runtime if runtime.is_file() else Path(sys.executable))
    output = args.output or ROOT / 'outputs/rag_ready' / (
        'demo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    command = [python, '-u', str(ROOT / 'psai.py'), 'analyze', '--input', str(args.input.resolve()),
               '--output', str(output.resolve()), '--classifier', 'rules', '--retrieval', args.mode,
               '--top-k', str(args.top_k), '--device', args.device]
    for query in args.query or []:
        command += ['--query', query]
    for flag, value in [('--resource-dir', args.resource_dir), ('--base-model-dir', args.base_model_dir)]:
        if value is not None:
            command += [flag, str(value.resolve())]
    try:
        if args.use_adapter:
            command += ['--adapter-dir', str(_verified_adapter())]
        if args.rerank:
            manifest = json.loads((ROOT / 'reports/rag_studies/retrieval_genq_lora_v1/auxiliary_models.json').read_text('utf-8'))
            teacher = next(row for row in manifest['models'] if row['role'] == 'pseudo_label_teacher')
            command += ['--rerank-model-dir', teacher['local_path']]
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(2, f'RAG configuration: {error}\n')
    print(f'RAG: {args.mode}; input: {args.input.resolve()}', flush=True)
    print('Classifier: rules for this functional demo. No training is started.', flush=True)
    if args.use_adapter:
        print('Experimental LoRA selected explicitly; human relevance validation is pending.', flush=True)
    print('Reading source, retrieving evidence, and checking citations...', flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        return result.returncode
    document = json.loads((output / 'analysis.json').read_text('utf-8'))
    hits = document['retrieval']['hits']
    print(f'Completed: {len(hits)} retrieved passages with source spans verified.', flush=True)
    print('Open HTML: ' + str((output / 'report.html').resolve()), flush=True)
    print('Structured data: ' + str((output / 'analysis.json').resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
