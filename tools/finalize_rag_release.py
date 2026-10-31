"""Export a portable, hash-bound RAG adapter and verify the runtime release.

Original study outputs, classifier releases, and production indexes are read only.
The portable export changes only the adapter's base-location configuration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RELEASE = ROOT / 'reports/rag_release_v1'
EXPORT = ROOT / 'models/rag/release_v1/retriever_adapter'


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify():
    manifest = read(RELEASE / 'RELEASE_MANIFEST.json')
    for name, expected in manifest['files_sha256'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise ValueError('Release integrity mismatch: ' + name)
    checkpoint = ROOT / manifest['study_checkpoint']
    exported = sha(EXPORT / 'adapter_model.safetensors')
    if exported != manifest['adapter_weights_sha256']:
        raise ValueError('Portable adapter differs from its recorded release weight hash')
    original_available = (checkpoint / 'adapter_model.safetensors').is_file()
    if original_available and exported != sha(checkpoint / 'adapter_model.safetensors'):
        raise ValueError('Portable adapter weights differ from the completed study checkpoint')
    print(json.dumps({'status': 'verified', 'files': len(manifest['files_sha256']),
                      'release': str(RELEASE), 'adapter_export': str(EXPORT),
                      'source_study_checkpoint_available': original_available,
                      'production_changed': False}, indent=2), flush=True)
    return manifest


def build():
    if (RELEASE / 'RELEASE_MANIFEST.json').exists():
        return verify()
    from psai_retrieval.runtime import _model_files, DEFAULT_BASE
    receipt = read(ROOT / 'models/rag/retrieval_genq_lora_v1/training_receipt.json')
    if receipt.get('status') != 'completed' or receipt['global_step'] != receipt['planned_steps']:
        raise ValueError('Cannot export an incomplete training run')
    checkpoint = Path(receipt['adapter_dir']).resolve()
    if not checkpoint.is_relative_to(ROOT / 'models/rag/retrieval_genq_lora_v1/checkpoints'):
        raise ValueError('Unexpected study checkpoint path')
    commit = read(checkpoint / 'commit.json')
    for name, expected in commit['files_sha256'].items():
        target = (checkpoint / name).resolve()
        if target.parent != checkpoint or sha(target) != expected:
            raise ValueError('Study checkpoint integrity mismatch: ' + name)
    for name, expected in receipt['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen study code changed: ' + name)
    base = _model_files(DEFAULT_BASE)
    for group in ['weights_sha256', 'tokenizer_sha256', 'configuration_sha256']:
        for name, expected in receipt['base'][group].items():
            if base['files_sha256'].get(name) != expected:
                raise ValueError('Local base differs from the training receipt: ' + name)
    if EXPORT.exists():
        raise FileExistsError('Adapter export already exists without a release manifest; inspect it before proceeding')
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.rag-export-', dir=EXPORT.parent))
    try:
        shutil.copyfile(checkpoint / 'adapter_model.safetensors', stage / 'adapter_model.safetensors')
        config = read(checkpoint / 'adapter_config.json')
        config['base_model_name_or_path'] = 'BAAI/bge-m3'
        config['revision'] = base['revision']
        (stage / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        (stage / 'base_provenance.json').write_text(json.dumps({
            'model_id': 'BAAI/bge-m3', 'revision': base['revision'],
            'files_sha256': base['files_sha256'],
        }, indent=2) + '\n', encoding='utf-8')
        stage.rename(EXPORT)
    finally:
        if stage.exists():
            if stage.resolve().parent != EXPORT.parent or not stage.name.startswith('.rag-export-'):
                raise RuntimeError('Unsafe export staging cleanup target')
            shutil.rmtree(stage)
    sources = [ROOT / name for name in ['psai.py', 'psai_reports.py', 'extraction.py', 'planning.py',
        'pdf_preprocessing.py', 'ensemble_classifier.py', 'requirements-rag.txt',
        'tools/demo_rag.py', 'tools/verify_rag_pdf_demo.py', 'tools/finalize_rag_release.py']]
    for directory in ['psai_retrieval', 'psai_rag_quality', 'psai_rag']:
        sources += list((ROOT / directory).glob('*.py'))
    sources += list(EXPORT.glob('*'))
    manifest = {
        'release_version': 'rag-runtime-1.0', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'source_grounded_local_document_and_resource_retrieval',
        'default_demo_mode': 'bm25', 'default_dense_encoder': 'BAAI/bge-m3 frozen base',
        'experimental_adapter': str(EXPORT.relative_to(ROOT)),
        'study_checkpoint': str(checkpoint.relative_to(ROOT)),
        'adapter_weights_sha256': sha(EXPORT / 'adapter_model.safetensors'),
        'base_revision': base['revision'], 'base_files_sha256': base['files_sha256'],
        'portable_config_note': 'Base path changed to repo ID; weights identical. Relocation checks base file hashes.',
        'independent_human_validation_complete': False, 'final_test_evaluated': False,
        'production_replacement': False,
        'files_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(sources)) if path.is_file()},
    }
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / 'RELEASE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else build()


if __name__ == '__main__':
    main()
