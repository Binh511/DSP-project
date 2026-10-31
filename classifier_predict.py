"""Run an explicitly selected local study-v1 classifier on JSON/JSONL items.

No checkpoint is selected by default and no production model is changed.
Input JSON is an array; JSONL contains one item object per nonblank line.
Output is a JSON array with canonical-label probabilities and input IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from study_inputs import MODES, encode_items

LABELS = ('OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER')


def validate_records(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise ValueError('Input must be a list of item objects')
    for index, item in enumerate(records):
        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
            raise ValueError(f'Item {index} needs nonempty text')
        if any(not isinstance(item.get(field, ''), str) for field in ('section', 'prev_text', 'next_text')):
            raise ValueError(f'Item {index} context fields must be strings')
        pos = item.get('pos', 0)
        if isinstance(pos, bool) or not isinstance(pos, (int, float)) or not math.isfinite(pos) or not 0 <= pos <= 1:
            raise ValueError(f'Item {index} pos must be finite in [0, 1]')
        for field in ('id', 'item_id'):
            if field in item and (isinstance(item[field], bool) or not isinstance(item[field], (str, int))):
                raise ValueError(f'Item {index} {field} must be a string or integer')


def _checkpoint_config(model_dir: Path) -> dict:
    if not model_dir.is_dir():
        raise ValueError('model_dir must be an existing local checkpoint directory')
    if (model_dir / 'invalid_checkpoint.json').exists():
        raise ValueError('Checkpoint is marked invalid; inspect invalid_checkpoint.json')
    config = json.loads((model_dir / 'inference_config.json').read_text(encoding='utf-8'))
    model_config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    labels = config.get('labels')
    if (not isinstance(labels, list) or len(labels) != 5 or any(not isinstance(v, str) for v in labels)
            or set(labels) != set(LABELS)):
        raise ValueError('Saved inference labels must map exactly the five PSAI labels')
    if config.get('input_protocol') != 'study-v1' or config.get('context_mode', 'full') not in MODES:
        raise ValueError('Only saved study-v1 checkpoints with a known context mode are supported')
    for field, default in [('max_len', 256), ('text_max_chars', 600)]:
        value = config.get(field, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{field} must be a positive integer')
    bias = config.get('other_bias', 0.)
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
        raise ValueError('other_bias must be finite')
    expected_ids = {str(index): label for index, label in enumerate(labels)}
    expected_labels = {label: index for index, label in enumerate(labels)}
    if (model_config.get('id2label') != expected_ids or model_config.get('label2id') != expected_labels
            or model_config.get('num_labels', 5) != 5):
        raise ValueError('Model label mapping disagrees with saved inference label order')
    return config


def _load_checkpoint(model_dir: Path, config: dict):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
    model = model.to(device=device, dtype=torch.float32).eval()
    if (model.config.num_labels != 5
            or {str(k): v for k, v in model.config.id2label.items()} != {str(i): l for i, l in enumerate(config['labels'])}
            or model.config.label2id != {l: i for i, l in enumerate(config['labels'])}):
        raise ValueError('Loaded model label mapping disagrees with saved inference config')
    return tokenizer, model, torch, device


def predict(records: list[dict], model_dir: str | Path, batch_size: int = 8) -> list[dict]:
    """Predict using the explicitly supplied saved study checkpoint, without scoring."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    validate_records(records)
    if not records:
        return []
    model_dir = Path(model_dir).resolve()
    config = _checkpoint_config(model_dir)
    tokenizer, model, torch, device = _load_checkpoint(model_dir, config)
    results = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        encoding = encode_items(tokenizer, batch, config, padding=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            logits = model(**encoding).logits.float()
            if tuple(logits.shape) != (len(batch), 5) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Classifier returned invalid or nonfinite logits')
            logits[:, config['labels'].index('OTHER')] += config.get('other_bias', 0.)
            probabilities = logits.softmax(-1)
            if not bool(torch.isfinite(probabilities).all()):
                raise ValueError('Classifier returned nonfinite probabilities')
            probabilities = probabilities.cpu().tolist()
        for local_index, (item, values) in enumerate(zip(batch, probabilities)):
            canonical = {label: float(values[config['labels'].index(label)]) for label in LABELS}
            label = max(canonical, key=canonical.get)
            result = dict(index=offset + local_index, label=label,
                          confidence=canonical[label], probabilities=canonical)
            result.update({field: item[field] for field in ('id', 'item_id') if field in item})
            results.append(result)
    return results


classify = predict


def read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if not text.strip():
        return []
    if path.suffix.lower() == '.jsonl':
        try:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONL input: {exc}') from exc
    else:
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError('JSON input must be an array; use .jsonl for one object per line') from exc
    validate_records(records)
    return records


def write_predictions(path: str | Path, predictions: list[dict]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('Prediction output already exists; choose a new file')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix=path.name + '.',
                                         suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(predictions, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link publishes atomically without clobbering a
        # competing writer that creates the destination after our first check.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError('Prediction output already exists; choose a new file')
        predictions = predict(read_records(args.input), args.model_dir, args.batch_size)
        write_predictions(args.output, predictions)
    except (OSError, ValueError, TypeError) as exc:
        print(f'Classifier inference failed: {exc}', file=sys.stderr)
        return 1
    print(f'Saved {len(predictions)} predictions to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
