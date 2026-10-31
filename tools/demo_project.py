"""One-command local PSAI demo, with isolated model stages and source reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/proposal.md')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fast', action='store_true', help='Explicit rules/BM25/source-plan mode without ML inference')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--ocr', choices=['auto', 'none', 'always'], default='auto')
    parser.add_argument('--resource-dir', type=Path)
    parser.add_argument('--base-model-dir', type=Path)
    parser.add_argument('--planner-model-dir', type=Path)
    parser.add_argument('--query', action='append')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runtime = ROOT / '.venv/psai_runtime/Scripts/python.exe'
    executable = str(runtime if runtime.is_file() else Path(sys.executable))
    output = args.output or ROOT / 'outputs/project_completion' / (
        'demo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    command = [executable, '-u', '-m', 'psai_project', '--input', str(args.input.resolve()),
               '--output', str(output.resolve()), '--device', args.device, '--ocr', args.ocr]
    if args.fast:
        command += ['--classifier', 'rules', '--retrieval', 'bm25', '--planner', 'evidence']
        print('FAST MODE: rules + BM25 + source-ordered plan; no learned model is used.', flush=True)
    else:
        command += ['--classifier', 'ensemble', '--retrieval', 'hybrid', '--planner', 'local']
        print('FULL LOCAL MODE: production ensemble -> hybrid RAG -> local LLM.', flush=True)
        print('Models run one stage at a time. No new training or API calls.', flush=True)
    for flag, value in [('--resource-dir', args.resource_dir), ('--base-model-dir', args.base_model_dir),
                        ('--planner-model-dir', args.planner_model_dir)]:
        if value is not None:
            command += [flag, str(value.resolve())]
    for query in args.query or []:
        command += ['--query', query]
    print('Input:', args.input.resolve(), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    print('Report:', (output / 'report.html').resolve(), flush=True)
    if (output / 'workspace.html').is_file():
        print('Workspace:', (output / 'workspace.html').resolve(), flush=True)
    print('Structured evidence and plan:', (output / 'analysis.json').resolve(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
