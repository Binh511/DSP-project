"""Bundle curated classifier/RAG evidence plus the complete local pipeline.

Previous handoff archives supply allowlists; files are copied from the current
workspace and inventoried. Full model weights/environments/secrets are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
REPORTS = ROOT / 'reports/project_completion'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def archive_allowlist(path):
    result = set()
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None: raise ValueError('Input archive CRC is invalid')
        for name in archive.namelist():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe source archive member')
            file = ROOT / name
            if file.is_file(): result.add(file)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORTS / 'PSAI_PROJECT_CODE_VA_BANG_CHUNG.zip')
    args = parser.parse_args()
    from tools.finalize_project_release import verify
    release = verify()
    output = args.output.resolve()
    if output.exists(): raise FileExistsError('Existing archive is preserved; choose a new output')
    paths = {ROOT / name for name in release['files_sha256']}
    for source in [ROOT / 'reports/classifier/release_source_clean_v2/PSAI_CLASSIFIER_V2_CODE_VA_BANG_CHUNG.zip',
                   ROOT / 'reports/rag_release_v1/PSAI_RAG_BAN_GIAO_v1_1.zip']:
        paths |= archive_allowlist(source)
    for package in ['psai_project', 'psai_planner', 'psai_ocr', 'psai_rules']:
        paths |= {file for file in (ROOT / package).iterdir() if file.is_file() and file.suffix in ['.py', '.md', '.txt', '.ps1']}
    for name in ['README.md', 'README_PROJECT.md', 'README_RAG.md', 'README_CLASSIFIER.md',
                 'requirements-project.txt', 'tools/setup_project_planner_model.py',
                 'tools/build_complete_project_report.py', 'tools/build_rag_completion_report.py',
                 'tools/package_complete_project.py', 'tools/verify_rag_pdf_demo.py',
                 'tests/__init__.py', 'test_psai_reports.py']:
        paths.add(ROOT / name)
    receipt = json.loads((REPORTS / 'latest_acceptance.json').read_text('utf-8'))
    fixture_root = (ROOT / receipt['fixture']).parent
    paths |= {file for file in fixture_root.rglob('*') if file.is_file()}
    paths |= {file for file in REPORTS.iterdir() if file.is_file()
              and file.suffix in ['.json', '.md', '.html', '.log']
              and file.name not in ['package_receipt.json', 'handoff_verification.json']}
    for file in ROOT.glob('tests/test_project_*.py'): paths.add(file)
    files = []
    for path in sorted(paths):
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Required handoff file is missing/outside root: ' + str(path))
        name = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in ['.zip', '.pyc', '.pt', '.bin'] or '.venv' in path.parts or path.name.startswith('.env'):
            continue
        if path.suffix.lower() == '.safetensors' and path.stat().st_size > 8 * 1024 * 1024:
            continue
        files.append(path)
    inventory = {'created_utc': datetime.now(timezone.utc).isoformat(),
                 'files': {},
                 'weights_policy': 'Only small existing retrieval adapter included; base/LLM/classifier weights separate.',
                 'quality_status': 'Functional acceptance passed; independent human quality evaluation incomplete.'}
    def member_name(path):
        if path.is_relative_to(fixture_root / 'project'):
            return 'DEMO/' + path.relative_to(fixture_root / 'project').as_posix()
        if path.is_relative_to(fixture_root):
            return 'DEMO_INPUT/' + path.relative_to(fixture_root).as_posix()
        return path.relative_to(ROOT).as_posix()
    members = {member_name(path): path for path in files}
    if len(members) != len(files): raise ValueError('Two source files map to the same archive member')
    inventory['files'] = {name: sha(path) for name, path in members.items()}
    inventory['relocated_demo_paths'] = {path.relative_to(ROOT).as_posix(): name for name, path in members.items()
                                         if name.startswith(('DEMO/', 'DEMO_INPUT/'))}
    note = '''PSAI COMPLETE LOCAL PIPELINE — CODE AND EVIDENCE

Open README_PROJECT.md, then reports/project_completion/BAO_CAO_HOAN_THIEN_PROJECT.html.
Open DEMO/report.html and DEMO/workspace.html for the actual full model output.
DEMO_INPUT contains the authored mixed PDF. The original acceptance receipt
retains original-machine paths; COMPLETE_HANDOFF_FILES.json maps those paths
to the shorter DEMO folders used here for Windows path-length portability.

Quick code-only demo: python tools/demo_project.py --fast
Full local demo: python tools/demo_project.py
Verify source bundle: python tools/finalize_project_release.py --verify-only

Full inference requires separately supplied model assets:
  models/final_s11, models/final_s22, models/final_s33 (complete trained exports)
  complete BAAI/bge-m3 snapshot (see PROJECT_RELEASE_MANIFEST.json)
  complete Qwen/Qwen2.5-1.5B-Instruct snapshot revision recorded in the release.
Provide --base-model-dir / PSAI_BGE_MODEL_DIR and --planner-model-dir as needed.
Use an appropriate CPU/CUDA Torch environment; the measured host is Windows11,
Python3.13 and RTX4050 6GB. Native OCR uses installed Windows en-US capability.
Helper execution policy is process-scoped only; it does not change stored policy.
No model download or external API call occurs during inference.

The HTML report embeds source text/OCR images. Keep workspace.html beside
report.html; export progress JSON before closing the browser. Task progress is
not a model-quality score. Suggestions still require human review.

Research data/reports preserve historical splits. Never train/tune on held-out
test sources. Synthetic labels are not independent human ground truth.
The classifier research release and production ensemble are distinct; do not
attribute the research validation score to the production ensemble.
Historical classifier packaging receipts refer to the original classifier ZIP;
this combined archive has its own COMPLETE_HANDOFF_FILES.json inventory.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in members.items(): archive.write(path, name)
        archive.writestr('START_HERE_PROJECT.txt', note)
        archive.writestr('COMPLETE_HANDOFF_FILES.json', json.dumps(inventory, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None: raise ValueError('Combined archive CRC check failed')
    summary = {'path': str(output), 'files': len(files) + 2, 'bytes': output.stat().st_size,
               'sha256': sha(output), 'crc_verified': True, 'full_model_weights_included': False}
    (REPORTS / 'package_receipt.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
