"""Verify real production-ensemble -> retrieval -> cited-report wiring on an authored fixture.

This is a functional smoke check, never a quality benchmark or final-test score.
It performs local inference only and does not train or replace model checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_probabilities(rows, labels, expected_rows: int) -> list[list[float]]:
    values = [[float(value) for value in row] for row in rows]
    if len(values) != expected_rows:
        raise ValueError('Prediction count differs from source sentences')
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError('Expected five distinct classifier labels')
    for row in values:
        if (len(row) != len(labels) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row)
                or not math.isclose(sum(row), 1.0, abs_tol=1e-5)):
            raise ValueError('Invalid five-class probability distribution')
    return values


def verify_export(folder: Path) -> dict:
    """Check exported citations and that actual ensemble extraction was selected."""
    folder = Path(folder).resolve()
    payload = json.loads((folder / 'analysis.json').read_text('utf-8'))
    text = (folder / 'source.txt').read_text('utf-8')
    if payload['metadata']['classifier'] != 'ensemble':
        raise ValueError('Export does not use the production ensemble')
    if payload['extraction']['extraction_method'] != 'deberta_classifier_plus_cue_rules':
        raise ValueError('Export does not contain model-based extraction')
    if payload['metadata'].get('production_model_changed') is not False:
        raise ValueError('Export must explicitly preserve production checkpoints')
    extraction_count, model_evidence_count = 0, 0
    for entries in payload['extraction']['fields'].values():
        for item in entries:
            start, end = item['char_start'], item['char_end']
            if (type(start) is not int or type(end) is not int or item.get('verified') is not True
                    or not 0 <= start < end <= len(text) or text[start:end] != item['quote']):
                raise ValueError('Extraction citation differs from exported source')
            extraction_count += 1
            model_evidence_count += item.get('evidence_method') in ('model_label', 'model_label_with_cue_rule')
    retrieval = payload.get('retrieval')
    if not isinstance(retrieval, dict) or not retrieval.get('hits'):
        raise ValueError('Functional check requires at least one retrieved passage')
    for hit in retrieval['hits']:
        snapshot = (folder / hit['citation_source']).resolve()
        if not snapshot.is_relative_to(folder):
            raise ValueError('Retrieval citation escapes exported report')
        source = snapshot.read_text('utf-8')
        start, end = hit['char_start'], hit['char_end']
        if (type(start) is not int or type(end) is not int or hit.get('verified') is not True
                or not 0 <= start < end <= len(source) or source[start:end] != hit['quote']
                or hashlib.sha256(source.encode('utf-8')).hexdigest() != hit['source_sha256']):
            raise ValueError('Retrieval citation/hash differs from exported source')
    if payload['plan'].get('related_context') != retrieval['hits']:
        raise ValueError('Retrieved context was not connected to the plan')
    by_id = {entry['evidence_id']: entry for entries in payload['extraction']['fields'].values() for entry in entries}
    for step in payload['plan']['steps']:
        if not any(evidence_id in by_id and step['description'] == by_id[evidence_id]['value'].strip()
                   for evidence_id in step['evidence_ids']):
            raise ValueError('Plan step is not grounded in extracted evidence')
    return {'extracted_evidence_count': extraction_count, 'model_label_evidence_count': model_evidence_count,
            'retrieved_evidence_count': len(retrieval['hits']), 'plan_step_count': len(payload['plan']['steps']),
            'all_exported_source_spans_verified': True, 'retrieval_context_connected_to_plan': True,
            'research_accuracy_measured': False, 'human_label_quality_verified': False}


def run_check(input_path: Path, output_dir: Path, *, mode: str = 'bm25',
              base_model_dir: Path | None = None, adapter_dir: Path | None = None,
              rerank_model_dir: Path | None = None, device: str = 'auto') -> dict:
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Choose a new smoke-check output directory')
    if input_path.suffix.lower() not in ('.md', '.txt') or not input_path.is_file():
        raise ValueError('Use an authored Markdown/text fixture; this command does not evaluate datasets')
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    receipt = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'starting', 'purpose': 'authored_fixture_functional_smoke_not_quality_benchmark',
               'input': str(input_path), 'output_dir': str(output_dir), 'retrieval_mode': mode,
               'training_started': False, 'final_test_evaluated': False, 'production_replacement': False}

    def record(stage: str):
        receipt['status'] = stage
        receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output_dir / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding='utf-8')
        print(stage, flush=True)

    try:
        from extraction import sentences_with_context
        import ensemble_classifier as ensemble
        import psai
        text = input_path.read_text('utf-8-sig')
        sentences = list(sentences_with_context(text))
        if not sentences:
            raise ValueError('Authored fixture has no classifiable sentences')
        items = [{'text': row['text'], 'section': row['section'], 'prev_text': row['prev'],
                  'pos': row['pos'], 'next_text': row['next_text']} for row in sentences]
        receipt['input_sha256_before'] = sha256_file(input_path)
        receipt['models_before'] = {
            name: {filename: sha256_file(ROOT / 'models' / name / filename)
                   for filename in ('model.safetensors', 'config.json', 'inference_config.json', 'tokenizer.json')}
            for name in ensemble.MODEL_NAMES}
        record('classifying_with_real_three_member_ensemble')
        probabilities = validate_probabilities(ensemble.classify_probs(items, batch_size=4),
                                               ensemble.LABELS, len(items))
        receipt['loaded_members'] = [
            {'name': name, 'parameters': sum(parameter.numel() for parameter in model.parameters()),
             'device': str(next(model.parameters()).device), 'dtype': str(next(model.parameters()).dtype),
             'context_version': configuration.get('ctx_version')}
            for name, (configuration, _, model) in zip(ensemble.MODEL_NAMES, ensemble._members)]
        predictions = []
        for sentence, item, row in zip(sentences, items, probabilities):
            winner = max(range(len(row)), key=row.__getitem__)
            predictions.append({**item, 'char_start': sentence['start'], 'char_end': sentence['end'],
                                'probabilities': dict(zip(ensemble.LABELS, row)),
                                'prediction': ensemble.LABELS[winner], 'score': row[winner]})
        (output_dir / 'classifier_predictions.json').write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        receipt['sentence_count'] = len(predictions)
        record('running_ensemble_retrieval_report')
        report = psai.analyze(input_path, output_dir / 'report', classifier='ensemble', retrieval=mode,
                              queries=['miner performance project objective', 'Python collector API credentials',
                                       'monitoring dashboard user guide'], top_k=5,
                              base_model_dir=base_model_dir, adapter_dir=adapter_dir,
                              rerank_model_dir=rerank_model_dir, device=device)
        receipt['checks'] = verify_export(report.parent)
        if receipt['checks']['model_label_evidence_count'] < 1:
            raise ValueError('Fixture produced no admitted model-label evidence; inspect predictions manually')
        receipt['models_after'] = {
            name: {filename: sha256_file(ROOT / 'models' / name / filename) for filename in hashes}
            for name, hashes in receipt['models_before'].items()}
        receipt['input_sha256_after'] = sha256_file(input_path)
        receipt['production_weights_unchanged'] = receipt['models_before'] == receipt['models_after']
        receipt['source_unchanged'] = receipt['input_sha256_before'] == receipt['input_sha256_after']
        if not receipt['production_weights_unchanged'] or not receipt['source_unchanged']:
            raise ValueError('Source or model file hashes changed during functional check')
        receipt['report'] = str(report.parent / 'report.html')
        receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
        record('completed')
        return receipt
    except Exception as error:
        receipt['error_type'], receipt['error'] = type(error).__name__, str(error)
        record('failed')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/proposal.md')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('bm25', 'hybrid'), default='bm25')
    parser.add_argument('--base-model-dir', type=Path)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--rerank-model-dir', type=Path)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = parser.parse_args(argv)
    if args.mode == 'bm25' and any((args.base_model_dir, args.adapter_dir, args.rerank_model_dir)):
        parser.error('Embedding/adapter/reranker options require --mode hybrid')
    output = args.output or ROOT / 'outputs/project_completion/ensemble_smoke' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + args.mode + '_' + uuid.uuid4().hex[:8])
    result = run_check(args.input, output, mode=args.mode, base_model_dir=args.base_model_dir,
                       adapter_dir=args.adapter_dir, rerank_model_dir=args.rerank_model_dir, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
