"""Functional acceptance of PDF/OCR -> ensemble -> hybrid -> local LLM -> workspace.

Uses an authored fixture only. This is not an independent quality evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/project_completion/acceptance')
    args = parser.parse_args()
    directory = args.output_root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader, PdfWriter
    from tools.verify_rag_pdf_demo import write_authored_pdf
    from psai_project import run_project
    from psai_project.workspace import initial_progress, validate_progress
    from tools.verify_project_ensemble_rag import verify_export

    native_pdf = directory / 'authored_native.pdf'
    write_authored_pdf(native_pdf)
    lines = ['PROJECT ACTIVITIES', 'We will implement a Python collector.',
             'We will test the collector with local sample data.',
             'We will document deployment and source citations.']
    image = Image.new('RGB', (1800, 1100), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 42)
    for index, line in enumerate(lines):
        draw.text((80, 100 + 180 * index), line, fill='black', font=font)
    image.save(directory / 'authored_scan.png')
    image.save(directory / 'authored_scan.pdf', 'PDF', resolution=144)
    image.close()
    writer = PdfWriter()
    for path in [native_pdf, directory / 'authored_scan.pdf']:
        for page in PdfReader(str(path)).pages:
            writer.add_page(page)
    fixture = directory / 'complete_project_fixture.pdf'
    with fixture.open('wb') as stream:
        writer.write(stream)
    before = {str(path.relative_to(ROOT)): sha(path) for path in
        [ROOT / 'models' / name / 'model.safetensors' for name in ['final_s11', 'final_s22', 'final_s33']]}
    print('Starting full local project acceptance on authored 3-page mixed PDF...', flush=True)
    report = run_project(fixture, directory / 'project', classifier='ensemble', retrieval='hybrid',
                         planner='local', ocr='auto', device='cuda', resource_dir=ROOT / 'examples/rag_resources')
    folder = report.parent
    analysis = json.loads((folder / 'analysis.json').read_text('utf-8'))
    checks = verify_export(folder)
    assert analysis['source']['ocr_pages'] >= 1
    assert analysis['plan']['planning_method'] == 'local_llm_evidence_catalog_v1'
    assert analysis['plan']['provenance']['model_loaded'] is True
    assert analysis['plan']['steps'] and (folder / 'workspace.html').is_file()
    assert analysis['rule_facts']['fields']['budgets']
    source = (folder / 'source.txt').read_bytes()
    assert hashlib.sha256(source).hexdigest() == analysis['metadata']['source_text_sha256']
    for step in analysis['plan']['steps']:
        assert step['needs_review'] is True
    initial = initial_progress(analysis)
    assert validate_progress(initial, analysis) == initial
    after = {name: sha(ROOT / name) for name in before}
    assert before == after
    class Anchors(HTMLParser):
        def __init__(self): super().__init__(); self.ids = set(); self.links = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs: self.ids.add(attrs['id'])
            if tag == 'a' and attrs.get('href', '').startswith('#'): self.links.append(attrs['href'][1:])
    anchors = Anchors(); anchors.feed(report.read_text('utf-8'))
    assert all(anchor in anchors.ids for anchor in anchors.links)
    responses = {name: json.loads((folder / f'jobs/{name}.response.json').read_text('utf-8'))
                 for name in ['extract', 'retrieve', 'plan']}
    receipt = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'fixture': str(fixture.relative_to(ROOT)), 'report': str(report.relative_to(ROOT)),
        'workspace': str((folder / 'workspace.html').relative_to(ROOT)), 'source_checks': checks,
        'ocr_pages': analysis['source']['ocr_pages'], 'llm_tasks': len(analysis['plan']['steps']),
        'llm_model_loaded': True, 'source_byte_hash_verified': True,
        'all_report_anchors_resolve': True, 'checkpoint_hashes_unchanged': True,
        'classifier_checkpoints_sha256': after,
        'gpu_peaks': {name: value.get('cuda') for name, value in responses.items()},
        'stages': analysis['metadata']['stages'],
        'quality_scope': 'Authored functional fixture only; no final-test score or independent human quality claim'}
    (directory / 'acceptance_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    pointer = ROOT / 'reports/project_completion/latest_acceptance.json'
    pointer.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
