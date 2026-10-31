"""Sequential local evaluation with saved probabilities, provenance and timing."""
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from study_data import OUT, sha, read_json, write_json
from study_inputs import encode_items
from train_human_cv import ROOT, LABELS, read_original, read_human

PROTOCOL = 'psai-local-evaluation-v1; text600; legacy_section_last; saved_bias; float32'


def fingerprint(rows):
    return hashlib.sha256(rows.fillna('').to_csv(index=False).encode()).hexdigest()


def legacy_context(item, cfg):
    section, prev = str(item.get('section', '') or ''), str(item.get('prev_text', '') or '')
    text = str(item['text'])
    if cfg.get('ctx_version') == 3:
        from train_human_cv import context
        return context(section, prev, item.get('pos', 0), text, item.get('next_text', ''))
    if cfg.get('ctx_version') == 2:
        import re
        bullet = int(bool(re.match(r'^\s*(?:[-*●•]|\d+[.)])\s', text)))
        return f'Section: {section.strip()[-120:]} | Pos: {int(float(item.get("pos", 0)) * 10)} | Bullet: {bullet} | Previous: {prev.strip()[-200:]}'
    return f'Section: {section.strip().split(" > ")[-1][:120]} | Previous: {prev.strip()[-200:]}'


def predict_local(model_dir, rows, batch_size=8):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if batch_size < 1 or rows.empty:
        raise ValueError('Require nonempty rows and positive batch size')
    if (model_dir / 'invalid_checkpoint.json').exists():
        raise ValueError('This checkpoint is marked invalid; inspect failure metadata')
    config = read_json(model_dir / 'inference_config.json')
    if set(config['labels']) != set(LABELS) or len(config['labels']) != 5:
        raise ValueError('Invalid model label mapping')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).to(device).eval()
    sync()
    load_seconds = time.perf_counter() - start
    items = rows.fillna('').to_dict('records')

    def encode(batch):
        if config.get('input_protocol') == 'study-v1':
            return encode_items(tokenizer, batch, config, padding=True, return_tensors='pt').to(device)
        return tokenizer([str(i['text'])[:config.get('text_max_chars', 600)] for i in batch],
                         [legacy_context(i, config) for i in batch],
                         truncation='longest_first', max_length=config.get('max_len', 192),
                         padding=True, return_tensors='pt').to(device)

    # Warm up once; separate model loading, model execution and tokenization.
    with torch.inference_mode():
        model(**encode(items[:min(batch_size, len(items))]))
    sync()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    probabilities, batch_times, forward_times = [], [], []
    for offset in range(0, len(items), batch_size):
        sync()
        start = time.perf_counter()
        encoding = encode(items[offset:offset + batch_size])
        sync()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            logits = model(**encoding).logits.float()
            logits[:, config['labels'].index('OTHER')] += config.get('other_bias', 0.)
            probs = logits.softmax(-1).cpu().numpy()
        sync()
        end = time.perf_counter()
        forward_times.append(end - forward_start)
        batch_times.append(end - start)
        probabilities.append(probs[:, [config['labels'].index(l) for l in LABELS]])
    peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    benchmark = dict(device=device, batch_size=batch_size, n=len(rows), load_seconds=load_seconds,
                     warmup_batches=1, inference_seconds=sum(batch_times),
                     forward_seconds=sum(forward_times), sentences_per_second=len(rows) / sum(batch_times),
                     mean_ms_per_sentence=1000 * sum(batch_times) / len(rows),
                     median_batch_ms=1000 * float(np.median(batch_times)),
                     p95_batch_ms=1000 * float(np.percentile(batch_times, 95)),
                     peak_allocated_vram_bytes=peak,
                     note='Includes tokenization in inference_seconds; excludes load/warmup. No PDF/OCR timing. Small sample; batch latency is not single-sentence latency.')
    del model, tokenizer, encoding, logits
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.vstack(probabilities), benchmark


def save_evaluation(folder, name, rows, probabilities, metadata):
    if probabilities.shape != (len(rows), 5) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid probability matrix')
    np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-5)
    pred_ids = probabilities.argmax(1)
    pred = np.array(LABELS)[pred_ids]
    result = classification_report(rows.label, pred, labels=LABELS, output_dict=True, zero_division=0)
    folder.mkdir(parents=True, exist_ok=True)
    output = rows.copy()
    output['true_label'], output['predicted_label'] = rows.label, pred
    output['confidence'] = probabilities.max(1)
    for i, label in enumerate(LABELS):
        output['prob_' + label] = probabilities[:, i]
    output.to_csv(folder / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
    output.loc[output.true_label.ne(output.predicted_label)].to_csv(folder / f'{name}_errors.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(confusion_matrix(rows.label, pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(folder / f'{name}_confusion.csv', encoding='utf-8-sig')
    report = dict(model=name, n=len(rows), accuracy=float(accuracy_score(rows.label, pred)),
                  macro_f1=result['macro avg']['f1-score'], per_class={l: result[l] for l in LABELS}, **metadata)
    write_json(folder / f'{name}_metrics.json', report)
    return report


def evaluate_models(names=None, dataset='human_holdout_diagnostic', batch_size=8):
    original = read_original(ROOT / 'data_segment_cls/segment_classification.csv')
    _, holdout, _ = read_human(ROOT / 'extra_human.csv', original)
    rows = holdout.reset_index(drop=True) if dataset == 'human_holdout_diagnostic' else original[original.fold.eq(0)].reset_index(drop=True)
    if dataset not in ('human_holdout_diagnostic', 'historical_fold0_diagnostic'):
        raise ValueError('This command only permits explicitly diagnostic evaluation')
    folder = OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    models = names or [p.name for p in sorted((ROOT / 'models').iterdir()) if (p / 'inference_config.json').exists()]
    registry_path = OUT / 'model_registry.json'
    registry = {r['model']: r for r in read_json(registry_path)} if registry_path.exists() else {}
    results, probabilities = [], {}
    for name in models:
        if Path(name).name != name:
            raise ValueError('Model must be a directory name under models')
        model_dir = ROOT / 'models' / name
        signature = dict(protocol=PROTOCOL, dataset_sha256=fingerprint(rows),
                         model_weight_sha256={p.name: sha(p) for p in sorted(model_dir.glob('*.safetensors'))},
                         inference_config_sha256=sha(model_dir / 'inference_config.json'))
        metric_path = folder / f'{name}_metrics.json'
        if metric_path.exists():
            saved = read_json(metric_path)
            if any(saved.get(k) != v for k, v in signature.items()):
                raise ValueError(f'Stale evaluation cache for {name}; use a new versioned output folder')
            probs = pd.read_csv(folder / f'{name}_predictions.csv')[[f'prob_{l}' for l in LABELS]].to_numpy()
            results.append(saved)
        else:
            print('EVALUATE', name, dataset, flush=True)
            probs, benchmark = predict_local(model_dir, rows, batch_size)
            report = save_evaluation(folder, name, rows, probs, dict(**signature, benchmark=benchmark,
                evaluation_kind='diagnostic_not_independent; do_not_select_final_model',
                provenance=registry.get(name, {}).get('training_provenance', 'unknown')))
            results.append(report)
        probabilities[name] = probs
    member_names = ['final_s11', 'final_s22', 'final_s33']
    if all(n in probabilities for n in member_names):
        start = time.perf_counter()
        averaged = np.mean([probabilities[n] for n in member_names], axis=0)
        average_seconds = time.perf_counter() - start
        member_times = [r['benchmark']['inference_seconds'] for r in results if r['model'] in member_names]
        ensemble = save_evaluation(folder, 'ensemble_final_s11_s22_s33', rows, averaged,
                 dict(protocol=PROTOCOL, dataset_sha256=fingerprint(rows), members=member_names,
                      evaluation_kind='diagnostic_not_independent; do_not_select_final_model',
                      benchmark=dict(serial_member_inference_seconds=sum(member_times),
                                     average_probabilities_seconds=average_seconds,
                                     note='Serial ensemble using cached per-member probability outputs. Sum of separately measured member times; not a simultaneous-resident ensemble benchmark.')))
        results.append(ensemble)
        from itertools import combinations
        errors = {n: np.array(LABELS)[probabilities[n].argmax(1)] != rows.label.to_numpy() for n in member_names}
        overlap = []
        for a, b in combinations(member_names, 2):
            union = int((errors[a] | errors[b]).sum())
            overlap.append(dict(a=a, b=b, both_wrong=int((errors[a] & errors[b]).sum()),
                                either_wrong=union, error_jaccard=float((errors[a] & errors[b]).sum() / union) if union else None))
        write_json(folder / 'ensemble_error_overlap.json', overlap)
    pd.DataFrame([{k: r[k] for k in ('model', 'n', 'accuracy', 'macro_f1', 'evaluation_kind')} for r in results]).to_csv(folder / 'comparison.csv', index=False, encoding='utf-8-sig')
    write_json(folder / 'environment.json', {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'pandas', 'scikit-learn')})
    return results


def benchmark_deployed(repeats=3):
    """Measure active members and the actual simultaneously loaded ensemble."""
    import torch
    import ensemble_classifier as ensemble
    original = read_original(ROOT / 'data_segment_cls/segment_classification.csv')
    _, rows, _ = read_human(ROOT / 'extra_human.csv', original)
    results = []
    for name in ensemble.MODEL_NAMES:
        runs = [predict_local(ROOT / 'models' / name, rows, batch_size=8)[1] for _ in range(repeats)]
        results.append(dict(model=name, runs=runs,
                            mean_seconds=float(np.mean([x['inference_seconds'] for x in runs])),
                            std_seconds=float(np.std([x['inference_seconds'] for x in runs], ddof=1)),
                            peak_allocated_vram_bytes=max(x['peak_allocated_vram_bytes'] or 0 for x in runs)))
    inputs = rows.fillna('').to_dict('records')
    ensemble.classify_probs(inputs[:8], batch_size=8)
    timings = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        start = time.perf_counter()
        ensemble.classify_probs(inputs, batch_size=8)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    results.append(dict(model='active_ensemble_simultaneous', runs_seconds=timings,
                        mean_seconds=float(np.mean(timings)), std_seconds=float(np.std(timings, ddof=1)),
                        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))
    ensemble._members = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(OUT / 'benchmark_deployed.json', dict(n=len(rows), repeats=repeats, batch_size=8,
               protocol='All rows including tokenization; exclude model load and first warmup; synchronise CUDA; no PDF/OCR time',
               dataset_sha256=fingerprint(rows), results=results))
    return results
