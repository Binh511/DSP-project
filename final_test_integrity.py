"""Integrity helpers for final evaluation; never infer human approvals."""
from __future__ import annotations

from pathlib import Path
import math
from numbers import Integral

import pandas as pd

from study_data import ROOT, sha


def inference_manifest(model_dir: Path) -> dict:
    required = ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'inference_config.json', 'training_provenance.json']
    files = {name: sha(model_dir / name) for name in required}
    for name in ['special_tokens_map.json', 'spm.model', 'sentencepiece.bpe.model', 'added_tokens.json']:
        if (model_dir / name).is_file():
            files[name] = sha(model_dir / name)
    implementation = ['study_evaluation.py', 'study_inputs.py', 'train_human_cv.py',
                      'study_final_test.py', 'final_test_integrity.py', 'replay_validation.py']
    import torch
    runtime = dict(torch=torch.__version__, cuda=torch.version.cuda,
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                   cudnn=torch.backends.cudnn.version(),
                   cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32,
                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return dict(model_files=files, implementation={name: sha(ROOT / name) for name in implementation}, runtime=runtime)


def validate_validation_binding(result: dict, provenance: dict, config: dict) -> None:
    if result.get('evaluation_kind') != 'validation_selection':
        raise ValueError('Model selection requires validation selection metrics')
    for key in ['train_sha256', 'validation_sha256']:
        if not result.get(key) or result[key] != provenance.get(key):
            raise ValueError('Validation metrics and training provenance differ: ' + key)
    if result.get('config') != provenance.get('config'):
        raise ValueError('Validation and training configurations differ')
    for key, value in result['config'].items():
        if config.get(key) != value:
            raise ValueError('Inference configuration differs from evaluated configuration: ' + key)
    canonical = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']
    if (config.get('labels') != canonical or config.get('other_bias') != 0
            or config.get('ctx_version') != 3 or config.get('input_protocol') != 'study-v1'):
        raise ValueError('Selection requires the declared study-v1 labels/context and zero OTHER bias')
    if not math.isfinite(result.get('macro_f1', float('nan'))) or result.get('n', 0) < 1:
        raise ValueError('Invalid validation result')


def known_development_artifacts(root: Path = ROOT) -> dict:
    """Inventory labels, pseudo labels and review/tuning queues, including errors."""
    paths = {root / 'data_segment_cls/segment_classification.csv', root / 'extra_human.csv'}
    for pattern in ['extra*.csv', 'label*.csv', 'pseudo*.csv', 'check*.csv', 'review*.csv']:
        paths.update(root.glob(pattern))
    hashes, documents, errors = {}, {}, []
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root)).replace('\\', '/')
        hashes[relative] = sha(path)
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if 'document_id' in frame:
                for document_id, count in frame.document_id.value_counts().items():
                    documents.setdefault(document_id, []).append(dict(path=relative, rows=int(count)))
        except Exception as exc:
            errors.append(dict(path=relative, error=type(exc).__name__ + ': ' + str(exc)))
    for path in sorted((root / 'data_labeled').glob('PSAI*.md')):
        relative = str(path.relative_to(root)).replace('\\', '/')
        hashes[relative] = sha(path)
        documents.setdefault(path.stem, []).append(dict(path=relative, rows=None))
    return dict(hashes=hashes, documents=documents, read_errors=errors)


def verify_screening(screening: dict, rows_hash: str, inventory: dict, document_ids: set) -> None:
    if screening.get('candidate_rows_sha256') != rows_hash:
        raise ValueError('Screening does not match current candidate rows; screen again')
    if screening.get('development_artifact_hashes') != inventory['hashes']:
        raise ValueError('Development artifacts changed after screening')
    if inventory['read_errors']:
        raise ValueError('Unreadable development artifacts must be resolved before final evaluation')
    entries = screening.get('documents', [])
    if len(entries) != len(document_ids) or {e['document_id'] for e in entries} != document_ids:
        raise ValueError('Screening must cover every candidate document exactly once')
    if any(e.get('disposition') != 'APPROVED_INDEPENDENT' or not e.get('reviewed_by', '').strip() for e in entries):
        raise ValueError('Each candidate requires an explicit reviewed independence disposition')
    exposed = document_ids & set(inventory['documents'])
    if exposed:
        raise ValueError('Candidate documents already occur in development artifacts: ' + ', '.join(sorted(exposed)))


def verify_candidate_sources(gate: dict, rows: pd.DataFrame) -> None:
    paths, hashes = gate.get('document_paths', {}), gate.get('document_hashes', {})
    if not paths or set(paths) != set(hashes) or set(paths) != set(rows.document_id):
        raise ValueError('Candidate source paths/hashes must cover exactly the candidate document IDs')
    if not {'char_start', 'char_end', 'text'} <= set(rows):
        raise ValueError('Candidate source spans must be recorded')
    for document_id, value in paths.items():
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != hashes[document_id]:
            raise ValueError('Candidate source changed or is missing: ' + document_id)
        source = path.read_text(encoding='utf-8-sig')
        for row in rows.loc[rows.document_id.eq(document_id)].itertuples():
            start, end = row.char_start, row.char_end
            if (not isinstance(start, Integral) or not isinstance(end, Integral)
                    or not 0 <= start < end <= len(source) or source[start:end] != row.text):
                raise ValueError('Candidate text does not match its source span: ' + document_id)
