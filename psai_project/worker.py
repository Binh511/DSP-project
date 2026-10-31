"""One isolated model stage. No model is imported until a job is executed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def _hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text('utf-8'))


def _write(path, payload):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _extract(job, text):
    from extraction import extract_document
    mode = job.get('classifier', 'ensemble')
    if mode not in ('ensemble', 'rules'):
        raise ValueError('Unknown classifier stage mode')
    predictions = {'mode': mode, 'labels': [], 'items': []}
    models = {}
    requested_device = job.get('device', 'auto')
    if requested_device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Classifier device must be auto, cpu or cuda')
    if mode == 'rules':
        record = extract_document(job['document_id'], text, use_model=False)
    else:
        import ensemble_classifier as ensemble
        previous_device = ensemble._device
        target_device = previous_device if requested_device == 'auto' else requested_device
        if requested_device == 'cuda' and not ensemble.torch.cuda.is_available():
            raise ValueError('Classifier CUDA requested but unavailable')
        if ensemble._members is not None and target_device != previous_device:
            raise ValueError('Use a fresh worker when changing the classifier device')
        ensemble._device = target_device
        original = ensemble.classify_probs

        def observe(items, batch_size=32):
            values = original(items, batch_size)
            if values.shape != (len(items), 5):
                raise ValueError('Classifier probability shape is invalid')
            for item, row in zip(items, values):
                probabilities = [float(value) for value in row]
                if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
                        or not math.isclose(sum(probabilities), 1., abs_tol=1e-5)):
                    raise ValueError('Classifier returned an invalid probability distribution')
                predictions['items'].append({'input': item, 'probabilities': probabilities,
                    'label': ensemble.LABELS[int(row.argmax())]})
            return values

        ensemble.classify_probs = observe
        try:
            record = extract_document(job['document_id'], text, use_model=True)
        finally:
            ensemble.classify_probs = original
            ensemble._device = previous_device
        predictions['labels'] = list(ensemble.LABELS)
        for name in ensemble.MODEL_NAMES:
            folder = ensemble.MODELS_DIR / name
            models[name] = {file: _hash_file(folder / file) for file in
                           ['model.safetensors', 'config.json', 'tokenizer.json', 'inference_config.json']}
    return {'record': record, 'classifier_predictions': predictions,
            'provenance': {'method': record['extraction_method'], 'model_files_sha256': models,
                           'device': target_device if mode == 'ensemble' else None,
                           'probabilities_observed_without_modification': mode == 'ensemble'}}


def _retrieve(job, text):
    from psai_retrieval import retrieve_document
    from psai import _verified_retrieval, DEFAULT_RETRIEVAL_QUERIES
    mode = job.get('retrieval', 'hybrid')
    if mode == 'none':
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        payload = {'mode': 'none', 'queries': [], 'hits': [], 'provenance': {'method': 'disabled'}}
    else:
        options = {key: job.get(key) for key in ['base_model_dir', 'adapter_dir', 'rerank_model_dir']}
        resource = Path(job['resource_dir']).resolve() if job.get('resource_dir') else None
        payload = retrieve_document(text, job['document_id'], queries=job.get('queries') or DEFAULT_RETRIEVAL_QUERIES,
            top_k=job.get('top_k', 5), mode=mode, resource_dir=resource,
            device=job.get('device', 'auto'), **options)
    resource = Path(job['resource_dir']).resolve() if job.get('resource_dir') else None
    payload, snapshots = _verified_retrieval(payload, text, Path(job['input_path']).resolve(), resource)
    return {'retrieval': payload, 'source_texts': snapshots,
            'provenance': payload.get('provenance', {})}


def _plan(job, text):
    from planning import generate_plan
    record = _read(job['record_path'])['record']
    retrieval_result = _read(job['retrieval_path'])
    retrieved = retrieval_result['retrieval']
    record['related_context'] = retrieved['hits']
    source_plan = generate_plan(record, source_text=text)
    mode = job.get('planner', 'local')
    if mode == 'evidence':
        plan = source_plan
    elif mode == 'local':
        from psai_planner import plan_project
        model_dir = job.get('planner_model_dir')
        config = {'device': job.get('device', 'auto'), **job.get('planner_config', {})}
        if model_dir is None:
            manifest = _read(ROOT / 'reports/project_completion/local_planner_model.json')
            model_dir = manifest['local_path']
            config['model_revision'] = manifest['revision']
        elif len(Path(model_dir).name) == 40 and all(char in '0123456789abcdef' for char in Path(model_dir).name):
            config['model_revision'] = Path(model_dir).name
        plan = plan_project(record, text,
            retrieval={**retrieved, 'source_texts': retrieval_result['source_texts']},
            model_dir=model_dir, config=config)
        plan['related_context'] = retrieved['hits']
    else:
        raise ValueError('Planner must be local or evidence')
    return {'plan': plan, 'source_plan': source_plan,
            'provenance': plan.get('provenance', {'method': plan['planning_method'], 'model_loaded': False})}


def execute_job(job):
    if not isinstance(job, dict) or job.get('stage') not in ('extract', 'retrieve', 'plan'):
        raise ValueError('Worker needs a known stage')
    raw = Path(job['source_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != job.get('source_sha256'):
        raise ValueError('Source changed between project stages')
    text = raw.decode('utf-8')
    if not text.strip():
        raise ValueError('Source text is empty')
    if Path(job['input_path']).stem != job['document_id']:
        raise ValueError('Document ID differs from the supplied original input')
    operation = {'extract': _extract, 'retrieve': _retrieve, 'plan': _plan}[job['stage']]
    started = time.perf_counter()
    response = operation(job, text)
    if hashlib.sha256(Path(job['source_path']).read_bytes()).hexdigest() != job['source_sha256']:
        raise ValueError('Source changed during a model stage')
    packages = {}
    for name in ['numpy', 'torch', 'transformers', 'peft']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    response.update({'status': 'completed', 'stage': job['stage'],
        'source_sha256': job['source_sha256'], 'elapsed_seconds': time.perf_counter() - started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'package_versions': packages,
        'worker_sha256': _hash_file(Path(__file__)), 'training_performed': False})
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        response['cuda'] = {'device': torch.cuda.get_device_name(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--response', required=True, type=Path)
    args = parser.parse_args()
    try:
        job = _read(args.request)
        print('Project stage:', job.get('stage'), flush=True)
        response = execute_job(job)
        _write(args.response, response)
        return 0
    except Exception as error:
        response = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                    'traceback': traceback.format_exc()}
        for name in ['raw_response', 'messages', 'provenance']:
            if hasattr(error, name): response[name] = getattr(error, name)
        _write(args.response, response)
        print('Project stage failed:', error, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
