"""Predeclared controlled experiments; select on validation, never auto-open test."""
import gc
import hashlib
import importlib.metadata
import itertools
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fair_fold0 import partitions
from study_data import OUT, sha, read_json, write_json
from study_evaluation import fingerprint, save_evaluation, predict_local
from study_inputs import encode_items
from train_human_cv import ROOT, COLS, LABELS, L2I, BASE_MODEL, text_key


def split_data(include_human=True, overlay_path=None):
    original, train, test, validation, _, human, holdout, _ = partitions()
    if overlay_path is not None:
        train = apply_training_overlay(original, train, overlay_path)
    held = set(pd.concat([test, validation, holdout]).text.map(text_key))
    def purge(rows):
        mask = rows.text.map(text_key).isin(held)
        for col in ('prev_text', 'next_text'):
            mask |= rows[col].fillna('').map(text_key).isin(held)
        return rows.loc[~mask, COLS].copy()
    base, extra = purge(train), purge(human)
    training = pd.concat([base, extra], ignore_index=True) if include_human else base.reset_index(drop=True)
    assert not set(training.document_id) & set(pd.concat([test, validation, holdout]).document_id)
    return training, validation[COLS].reset_index(drop=True)


def apply_training_overlay(original, training, overlay_path):
    overlay = pd.read_csv(overlay_path, dtype={'document_id': str}, keep_default_na=False)
    required = {'document_id', 'source_line', 'text', 'annotation'}
    if not required <= set(overlay) or not overlay.annotation.isin(LABELS).all():
        raise ValueError('Invalid training-only label overlay')
    overlay['source_line'] = pd.to_numeric(overlay.source_line, errors='raise').astype(int)
    keys = ['document_id', 'source_line', 'text']
    if overlay.duplicated(keys).any() or original.duplicated(keys).any():
        raise ValueError('Nonunique overlay/source row key')
    matched = overlay.merge(original[keys + ['fold']], on=keys, how='left', validate='one_to_one')
    if matched.fold.isna().any() or not matched.fold.isin([2, 3, 4]).all():
        raise ValueError('Overlay may only change original training folds 2/3/4')
    result = training.merge(overlay[keys + ['annotation']], on=keys, how='left', validate='one_to_one')
    result['label'] = result.annotation.fillna(result.label)
    return result.drop(columns='annotation')


def create_plan(study_dir=None, overlay_path=None):
    study_dir = Path(study_dir) if study_dir is not None else OUT
    baseline = dict(context_mode='full', class_weight=True, label_smoothing=0.1,
                    include_human=True, seed=42, learning_rate=3e-5, epochs=5,
                    max_len=256, text_max_chars=600, input_protocol='study-v1',
                    warmup_ratio=0.1, split_seed=42, context_source='fixed_dataset_v3')
    experiments = []
    def add(name, group, **changes):
        experiments.append(dict(name=name, group=group, config={**baseline, **changes}, status='not_run'))
    add('baseline', 'baseline')
    if overlay_path is not None:
        for experiment in experiments:
            experiment['config']['use_label_overlay'] = True
        baseline['use_label_overlay'] = True
        add('baseline_original_labels', 'label_revision_ablation', use_label_overlay=False)
    for name, mode in [('A0_text', 'text'), ('A1_section', 'section'), ('A2_section_position', 'section_position'), ('A3_neighbors', 'neighbors')]:
        add(name, 'context_ablation', context_mode=mode)
    add('no_class_weights', 'loss_ablation', class_weight=False)
    add('no_label_smoothing', 'loss_ablation', label_smoothing=0.)
    add('no_human', 'data_ablation', include_human=False)
    for seed in (123, 2026):
        add('seed_' + str(seed), 'seed_stability', seed=seed)
    for lr in (1e-5, 2e-5):
        add('lr_' + str(lr), 'learning_rate_search', learning_rate=lr)
    for length in (128, 384):
        add('maxlen_' + str(length), 'length_search', max_len=length)
    for epochs in (3, 7):
        add('epochs_' + str(epochs), 'epoch_search', epochs=epochs)
    path = study_dir / 'experiment_plan.json'
    if path.exists():
        if read_json(path)['experiments'] != experiments:
            raise ValueError('Plan already exists with different content')
        return
    train, val = split_data(overlay_path=overlay_path)
    write_json(path, dict(selection='validation macro_f1 only', test_policy='No test access in train command',
                         baseline_training_sha256=fingerprint(train), validation_sha256=fingerprint(val),
                         estimated_full_runs=len(experiments),
                         overlay_sha256=sha(overlay_path) if overlay_path else None,
                         warning='Plan is not a result. Run baseline first; do not claim ablation/mean/std before completed runs. Fixed split across seeds.',
                         experiments=experiments))


def run_experiment(name, max_steps=None, study_dir=None, overlay_path=None):
    study_dir = Path(study_dir) if study_dir is not None else OUT
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, set_seed
    from classifier_training import WeightedClassificationTrainer
    plan = read_json(study_dir / 'experiment_plan.json')
    if any(sha(ROOT / filename) != digest for filename, digest in plan.get('code_sha256', {}).items()):
        raise ValueError('Study source code changed after plan freeze')
    if plan.get('overlay_sha256') != (sha(overlay_path) if overlay_path else None):
        raise ValueError('Overlay does not match frozen experiment plan')
    experiment = next((x for x in plan['experiments'] if x['name'] == name), None)
    if not experiment:
        raise ValueError('Unknown predeclared experiment')
    if not torch.cuda.is_available():
        raise RuntimeError('Full experiments require CUDA; use CPU unit tests for correctness')
    cfg = experiment['config']
    run_name = name if max_steps is None else f'{name}_smoke_{max_steps}'
    destination = ROOT / 'models' / 'controlled_study' / run_name
    if study_dir != OUT:
        destination = ROOT / 'models' / 'controlled_study' / study_dir.name / run_name
    run_folder = study_dir / 'experiments' / run_name
    if destination.exists() or run_folder.exists():
        raise FileExistsError('Never overwrite an experiment; use a new study version')
    train, val = split_data(cfg['include_human'], overlay_path if cfg.get('use_label_overlay', True) else None)
    full_train, _ = split_data(True, overlay_path)
    if fingerprint(full_train) != plan['baseline_training_sha256'] or fingerprint(val) != plan['validation_sha256']:
        raise ValueError('Dataset changed since plan was frozen')
    run_folder.mkdir(parents=True)
    source_files = ['study_inputs.py', 'study_experiments.py', 'classifier_training.py', 'fair_fold0.py', 'train_human_cv.py']
    provenance = dict(config=cfg, train_sha256=fingerprint(train), validation_sha256=fingerprint(val),
                      train_document_ids=sorted(train.document_id.unique()),
                      validation_document_ids=sorted(val.document_id.unique()),
                      train_count=len(train), validation_count=len(val), max_steps=max_steps,
                      code_sha256={f: sha(ROOT / f) for f in source_files},
                      data_sha256={f: sha(ROOT / f) for f in ['data_segment_cls/segment_classification.csv', 'extra_human.csv']},
                      library_versions={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'datasets', 'accelerate']})
    if overlay_path is not None:
        provenance['label_overlay_sha256'] = sha(overlay_path)
        provenance['label_origin'] = 'user_supplied_labels; independent_annotation_not_verified'
    write_json(run_folder / 'provenance.json', provenance)  # before any training
    (run_folder / 'source_snapshot').mkdir()
    for filename in source_files:
        shutil.copy2(ROOT / filename, run_folder / 'source_snapshot' / filename)
    if overlay_path is not None:
        shutil.copy2(overlay_path, run_folder / 'label_overlay.csv')
    train.to_csv(run_folder / 'training_rows.csv', index=False, encoding='utf-8-sig')
    val.to_csv(run_folder / 'validation_rows.csv', index=False, encoding='utf-8-sig')
    set_seed(cfg['seed'])
    # Cached mDeBERTa uses a SentencePiece/Unigram tokenizer, not Mistral's
    # regex tokenizer. Preserve it instead of applying an unrelated regex fix.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, fix_mistral_regex=False)
    provenance['tokenizer_backend_sha256'] = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
    def dataset(rows):
        enc = encode_items(tok, rows.fillna('').to_dict('records'), cfg)
        return Dataset.from_dict({**enc, 'labels': rows.label.map(L2I).tolist()})
    train_ds, val_ds = dataset(train), dataset(val)
    counts = np.maximum(np.bincount(train.label.map(L2I), minlength=5), 1)
    weights = len(train) / (5 * counts) if cfg['class_weight'] else np.ones(5)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=5,
                id2label=dict(enumerate(LABELS)), label2id=L2I, local_files_only=True,
                dtype=torch.float32)
    provenance['base_model_revision'] = getattr(model.config, '_commit_hash', None)
    write_json(run_folder / 'provenance.json', provenance)
    total_updates = max_steps or math.ceil(len(train) / 16) * cfg['epochs']
    provenance['effective_training_arguments'] = dict(batch=1, accumulation=16, eval_batch=8,
                    total_updates=total_updates, warmup_steps=math.ceil(cfg['warmup_ratio'] * total_updates),
                    optimizer='adamw_torch', scheduler='linear', class_weights=weights.tolist(),
                    bf16=bool(torch.cuda.is_bf16_supported()), parameter_dtype='float32')
    write_json(run_folder / 'provenance.json', provenance)
    args = TrainingArguments(output_dir=str(run_folder / 'checkpoints'),
                num_train_epochs=cfg['epochs'], max_steps=max_steps if max_steps else -1,
                learning_rate=cfg['learning_rate'], per_device_train_batch_size=1,
                gradient_accumulation_steps=16, per_device_eval_batch_size=8,
                warmup_steps=math.ceil(cfg['warmup_ratio'] * total_updates), weight_decay=0.01, max_grad_norm=1.,
                optim='adamw_torch', lr_scheduler_type='linear',
                bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                gradient_checkpointing=True, eval_strategy='epoch', save_strategy='epoch',
                load_best_model_at_end=True, metric_for_best_model='macro_f1', greater_is_better=True,
                save_total_limit=2, report_to='none', dataloader_num_workers=0, seed=cfg['seed'])
    from sklearn.metrics import f1_score
    def validation_metrics(p):
        if not np.isfinite(p.predictions).all():
            raise FloatingPointError('Nonfinite validation logits; do not select or export this checkpoint')
        return {'macro_f1': f1_score(p.label_ids, p.predictions.argmax(1), labels=range(5), average='macro', zero_division=0)}
    trainer = WeightedClassificationTrainer(model=model, args=args, train_dataset=train_ds,
                eval_dataset=val_ds, processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                class_weights=weights, label_smoothing=cfg['label_smoothing'],
                compute_metrics=validation_metrics)
    trainer.train()
    destination.mkdir(parents=True)
    trainer.save_model(str(destination))
    tok.save_pretrained(destination)
    trainer.state.save_to_json(str(destination / 'training_state.json'))
    write_json(destination / 'inference_config.json', dict(**cfg, labels=LABELS, ctx_version=3,
               other_bias=0., loss_normalization='microbatch_mean_trainer_accumulation'))
    write_json(destination / 'training_provenance.json', provenance)
    del trainer, model, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    probabilities, benchmark = predict_local(destination, val)
    save_evaluation(run_folder, 'validation', val, probabilities,
                    dict(evaluation_kind='smoke_not_for_selection' if max_steps else 'validation_selection',
                         config=cfg, benchmark=benchmark, train_sha256=fingerprint(train),
                         validation_sha256=fingerprint(val), model_weight_sha256=sha(destination / 'model.safetensors')))


def aggregate(study_dir=None):
    study_dir = Path(study_dir) if study_dir is not None else OUT
    rows = []
    folder = study_dir / 'experiments'
    if folder.exists():
        for path in folder.glob('*/validation_metrics.json'):
            result = read_json(path)
            if result['evaluation_kind'] != 'validation_selection':
                continue
            rows.append(dict(run=path.parent.name, macro_f1=result['macro_f1'], accuracy=result['accuracy'],
                             **result['config'], train_sha256=result['train_sha256'],
                             validation_sha256=result['validation_sha256']))
    pd.DataFrame(rows, columns=list(rows[0]) if rows else ['run', 'macro_f1', 'accuracy', 'seed', 'status']).to_csv(study_dir / 'completed_experiments.csv', index=False, encoding='utf-8-sig')
    # Only group runs identical in EVERYTHING except the training seed.
    groups = {}
    import json
    for row in rows:
        key = json.dumps({k: v for k, v in row.items() if k not in ('run', 'seed', 'macro_f1', 'accuracy')}, sort_keys=True)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, items in groups.items():
        seeds = [r['seed'] for r in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed in otherwise identical experiment group')
        values = [r['macro_f1'] for r in items]
        summaries.append(dict(configuration=json.loads(key), seeds=seeds, n=len(items),
                              mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              stability_evidence='at_least_3_seeds' if len(values) >= 3 else 'insufficient_runs'))
    write_json(study_dir / 'seed_stability.json', summaries)
    return rows
