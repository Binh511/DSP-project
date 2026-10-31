"""Package the RAG code, portable adapter, data and evidence using an allowlist.

No API keys, environments, full base weights, classifier checkpoints, or old
production index are included. Original study files are copied unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/rag_release_v1/PSAI_RAG_BAN_GIAO.zip')
    args = parser.parse_args()
    from tools.finalize_rag_release import verify
    release = verify()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Existing handoff archive is preserved; choose a new output')
    paths = {ROOT / name for name in release['files_sha256']}
    paths |= {ROOT / name for name in ['README_RAG.md', 'psai_retrieval/README.md',
        'test_psai_reports.py', 'tests/__init__.py', 'tests/test_psai_completion.py',
        'tests/test_rag_review_regressions.py', 'tools/build_rag_completion_report.py',
        'tools/package_rag_release.py', 'examples/proposal.md', 'examples/rag_resources/local_collector_guide.md']}
    study = ROOT / 'reports/rag_studies/retrieval_genq_lora_v1'
    for name in ['corpus.jsonl', 'candidate_passages.jsonl', 'source_manifest.json', 'split_manifest.json',
        'train_pairs.jsonl', 'validation_pairs.jsonl', 'generation_report.json', 'query_generation_manifest.json',
        'base_metrics.json', 'adapter_metrics.json', 'base_predictions.jsonl', 'adapter_predictions.jsonl',
        'selection_decision.json', 'KET_QUA_RAG_PILOT.md', 'DATA_SUPERVISION_AUDIT.md',
        'QUALITY_AND_ERROR_ANALYSIS.md', 'KE_HOACH_RAG.md', 'HUMAN_FINAL_TEST_TEMPLATE.csv']:
        paths.add(study / name)
    for directory in ['psai_rag_quality/artifacts/retrieval_genq_lora_v1', 'outputs/rag_ready/release_demo_v1_1']:
        paths.update(path for path in (ROOT / directory).rglob('*') if path.is_file())
    reports = ROOT / 'reports/rag_release_v1'
    for name in ['RELEASE_MANIFEST.json', 'BAO_CAO_RAG.md', 'BAO_CAO_RAG.html',
                 'verification.json', 'environment.json', 'unit_tests.log', 'model_runtime_smoke.json']:
        paths.add(reports / name)
    verification = json.loads((reports / 'verification.json').read_text('utf-8'))
    pdf_receipt = ROOT / verification['pdf_functional_smoke']
    paths.update(path for path in pdf_receipt.parent.rglob('*') if path.is_file())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError('Required handoff files are missing: ' + ', '.join(missing))
    for path in paths:
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Allowlisted handoff path escapes the workspace')
        if path.suffix.lower() == '.zip' or '.venv' in path.parts or path.name.startswith('.env'):
            raise ValueError('Unexpected archive, environment or secret file in allowlist')
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'files': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)},
               'base_weights_included': False, 'classifier_models_included': False,
               'study_contains_held_out_test_source_chunks': True,
               'instruction': 'Never use held-out test sources for training/tuning; no human final-test judgments are included.'}
    note = '''PSAI RAG handoff

Open README_RAG.md and reports/rag_release_v1/BAO_CAO_RAG.html first.
Quick demo: python tools/demo_rag.py
Verify package runtime: python tools/finalize_rag_release.py --verify-only
RAG tests: python -m unittest psai_retrieval.test_runtime psai_rag_quality.test_quality tests.test_psai_completion tests.test_rag_review_regressions test_psai_reports psai_rag.test_data psai_rag.test_train psai_rag.test_evaluate psai_rag.test_generate_queries

The portable adapter is included; BGE-M3 base weights are NOT included.
Provide the complete pinned base snapshot/tokenizer/pooling configs; set
PSAI_BGE_MODEL_DIR or pass --base-model-dir when using dense/hybrid.
requirements-rag.txt records runtime versions. Training also needs the
sentence-transformers version recorded in environment.json plus its dependencies.
The full-project 193-test result was obtained on the original machine; this
RAG-only handoff contains a smaller focused suite and no classifier checkpoints.

Study JSONL files are unchanged. Preserve the source splits and hashes.
The corpus contains held-out test-source passages; they must NEVER be used for
training, mining train negatives or parameter selection. Do not present
synthetic query/passage labels as independent human ground truth.
Original PDFs/cleaned Markdown sources are not included: their paths/hashes
are provenance references. Supply original sources separately for source audit.
Review packets embed candidate text. Send only the review form/CSV to raters;
keep coordinator_key.json away from independent raters.

The historical training driver keeps original machine paths. Its explicit
train_adapter API accepts a local base path and a new study directory;
new experiments must not overwrite the frozen study or copied results.
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('START_HERE.txt', note)
        archive.writestr('HANDOFF_FILES.json', json.dumps(receipt, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
    result = {'path': str(output), 'files': len(paths) + 2, 'bytes': output.stat().st_size,
              'sha256': sha(output), 'crc_verified': True}
    (reports / 'handoff_receipt.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
