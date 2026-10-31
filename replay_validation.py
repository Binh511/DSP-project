"""Replay only a frozen validation split and bind a complete inference manifest.

This is never called automatically and never evaluates test/final test.
Existing historical metrics are preserved; output must be a new directory.
"""
import argparse
from pathlib import Path

import pandas as pd

from final_test_integrity import inference_manifest, validate_validation_binding
from study_data import read_json, sha
from study_evaluation import fingerprint, predict_local, save_evaluation


def replay(experiment_dir: Path, model_dir: Path, output_dir: Path):
    experiment_dir, model_dir, output_dir = map(Path, (experiment_dir, model_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('Replay output must be a new directory')
    result = read_json(experiment_dir / 'validation_metrics.json')
    provenance = read_json(model_dir / 'training_provenance.json')
    config = read_json(model_dir / 'inference_config.json')
    validate_validation_binding(result, provenance, config)
    if provenance.get('max_steps') is not None or (model_dir / 'invalid_checkpoint.json').exists():
        raise ValueError('Cannot replay a smoke or invalid model for selection')
    if sha(model_dir / 'model.safetensors') != result['model_weight_sha256']:
        raise ValueError('Model differs from the historical validation result')
    rows = pd.read_csv(experiment_dir / 'validation_rows.csv', keep_default_na=False, float_precision='round_trip')
    if fingerprint(rows) != result['validation_sha256']:
        raise ValueError('Frozen validation rows changed')
    if (set(rows.document_id) != set(provenance['validation_document_ids'])
            or set(rows.document_id) & set(provenance['train_document_ids'])):
        raise ValueError('Rows are not exactly the recorded independent validation documents')
    manifest = inference_manifest(model_dir)
    probabilities, benchmark = predict_local(model_dir, rows)
    if inference_manifest(model_dir) != manifest:
        raise ValueError('Inference implementation/artifacts/runtime changed during replay')
    return save_evaluation(output_dir, 'validation', rows, probabilities, dict(
        evaluation_kind='validation_selection', config=result['config'], benchmark=benchmark,
        train_sha256=result['train_sha256'], validation_sha256=result['validation_sha256'],
        model_weight_sha256=result['model_weight_sha256'], validation_inference_manifest=manifest,
        replay_of_validation_metrics_sha256=sha(experiment_dir / 'validation_metrics.json'),
        historical_metrics_modified=False, test_or_final_accessed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(replay(args.experiment_dir, args.model_dir, args.output))
