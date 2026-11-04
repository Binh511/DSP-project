"""One-command local PSAI demo, with isolated model stages and source reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psai_project.execution import project_command, runtime_python


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
    executable = runtime_python(ROOT)
    output = args.output or ROOT / 'outputs/project_completion' / (
        'demo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    command = project_command(executable, args.input, output,
        mode='fast' if args.fast else 'full', device=args.device, ocr=args.ocr,
        path_options={'resource-dir': args.resource_dir, 'base-model-dir': args.base_model_dir,
                      'planner-model-dir': args.planner_model_dir}, queries=args.query)
    if args.fast:
        print('FAST MODE: rules + BM25 + source-ordered plan; no learned model is used.', flush=True)
    else:
        print('FULL LOCAL MODE: production ensemble -> hybrid RAG -> local LLM.', flush=True)
        print('Models run one stage at a time. No new training or API calls.', flush=True)
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
