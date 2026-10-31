"""Controlled paired-seed study of prepared Section context; never reads test.

Preserves the released v1 model and original studies. Trainer checkpoint
selection follows the old BF16 validation procedure; exported comparison uses
the common saved FP32 protocol. No AI labels are applied.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from study_data import ROOT, OUT, read_json, write_json, sha
from study_evaluation import fingerprint, predict_local, save_evaluation
from study_inputs import encode_items
from train_human_cv import LABELS, L2I, BASE_MODEL, COLS
from final_test_integrity import inference_manifest
from build_classifier_report import table

STUDY = OUT / 'studies/classifier_source_clean_context_study_v1'
DATA = OUT / 'studies/classifier_source_clean_v1'
REFERENCE = OUT / 'studies/classifier_validation_replay_v1'
SEEDS = [123, 42, 2026]
CODE = ['classifier_context_experiment.py', 'classifier_training.py', 'study_inputs.py',
        'study_evaluation.py', 'study_data.py', 'train_human_cv.py', 'final_test_integrity.py']
VARIANTS = ['baseline', 'context']


def validated_rows(variant='context', data=DATA):
    if variant not in VARIANTS:
        raise ValueError('Unknown context variant')
    manifest = read_json(data / 'preparation_manifest.json')
    if manifest.get('train_validation_source_overlap_after_purge') != 0:
        raise ValueError('Prepared source groups are not disjoint')
    for name, digest in manifest['evidence_files'].items():
        if sha(data / name) != digest:
            raise ValueError('Prepared context evidence changed: ' + name)
    frames = []
    for split in ['training', 'validation']:
        before = pd.read_csv(data / f'baseline_{split}_rows.csv', keep_default_na=False, float_precision='round_trip')
        after = pd.read_csv(data / f'{variant}_{split}_rows.csv', keep_default_na=False, float_precision='round_trip')
        if list(after) != COLS or after.empty:
            raise ValueError('Invalid prepared dataset columns or empty rows')
        pd.testing.assert_frame_equal(before.drop(columns='section'), after.drop(columns='section'))
        if not after.label.isin(LABELS).all():
            raise ValueError('Invalid labels')
        frames.append(after)
    if set(frames[0].document_id) & set(frames[1].document_id):
        raise ValueError('Training and validation share documents')
    return frames


def prepare():
    pairs = {variant: validated_rows(variant) for variant in VARIANTS}
    reference = read_json(REFERENCE / 'seed_123/validation_metrics.json')
    cfg = {**reference['config']}
    experiments = []
    for seed in SEEDS:
        for variant in VARIANTS:
            train, validation = pairs[variant]
            experiments.append(dict(name=f'{variant}_seed_{seed}', variant=variant,
                train_sha256=fingerprint(train), validation_sha256=fingerprint(validation),
                config={**cfg, 'seed': seed,
                        'context_source': 'fixed_dataset_v3_source_disjoint' if variant == 'baseline' else 'classifier-context-v2.1-source-disjoint'}))
    plan = dict(purpose='Paired original-context vs revised-context on identical SOURCE-DISJOINT training; original labels',
                prepared_manifest_sha256=sha(DATA / 'preparation_manifest.json'),
                code_sha256={name: sha(ROOT / name) for name in CODE},
                experiments=experiments,
                reference_metrics_sha256={str(s): sha(REFERENCE / f'seed_{s}/validation_metrics.json') for s in SEEDS},
                checkpoint_selection='same as original: Trainer epoch validation macro-F1 with BF16 autocast',
                comparison='new paired baseline+context on same purged train and same validation labels/order; old-model scores are historical only; no final test',
                ai_labels_applied=False, production_replaced=False, release_v1_replaced=False)
    STUDY.mkdir(parents=True, exist_ok=True)
    target = STUDY / 'experiment_plan.json'
    if target.exists():
        if read_json(target) != plan:
            raise ValueError('Frozen study changed; create a new version')
    else:
        write_json(target, plan)
        snapshot = STUDY / 'source_snapshot'
        snapshot.mkdir()
        for name in CODE:
            shutil.copy2(ROOT / name, snapshot / name)
    return plan, pairs


def run(name):
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, set_seed)
    from sklearn.metrics import f1_score
    from classifier_training import WeightedClassificationTrainer
    plan, pairs = prepare()
    if not torch.cuda.is_available():
        raise RuntimeError('This controlled study requires the local CUDA GPU')
    experiment = next(e for e in plan['experiments'] if e['name'] == name)
    cfg, variant = experiment['config'], experiment['variant']
    seed = cfg['seed']
    train, validation = pairs[variant]
    output = STUDY / 'experiments' / name
    destination = ROOT / 'models/controlled_study' / STUDY.name / name
    if output.exists() or destination.exists():
        raise FileExistsError('Never overwrite a context experiment')
    output.mkdir(parents=True)
    provenance = dict(config=cfg, train_sha256=fingerprint(train), validation_sha256=fingerprint(validation),
                      train_document_ids=sorted(train.document_id.unique()),
                      validation_document_ids=sorted(validation.document_id.unique()),
                      train_count=len(train), validation_count=len(validation), max_steps=None,
                      label_origin='original frozen labels; no proposed AI changes applied',
                      prepared_data_sha256={filename: sha(DATA / filename) for filename in [f'{variant}_training_rows.csv', f'{variant}_validation_rows.csv', 'preparation_manifest.json']},
                      code_sha256=plan['code_sha256'],
                      library_versions={name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'datasets', 'accelerate']})
    write_json(output / 'provenance.json', provenance)
    train.to_csv(output / 'training_rows.csv', index=False, encoding='utf-8-sig')
    validation.to_csv(output / 'validation_rows.csv', index=False, encoding='utf-8-sig')
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, fix_mistral_regex=False)
    provenance['tokenizer_backend_sha256'] = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
    def dataset(rows):
        enc = encode_items(tok, rows.to_dict('records'), cfg)
        return Dataset.from_dict({**enc, 'labels': rows.label.map(L2I).tolist()})
    train_ds, validation_ds = dataset(train), dataset(validation)
    counts = np.maximum(np.bincount(train.label.map(L2I), minlength=5), 1)
    weights = len(train) / (5 * counts)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, local_files_only=True,
                  num_labels=5, id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32)
    total = math.ceil(len(train) / 16) * cfg['epochs']
    provenance.update(base_model_revision=getattr(model.config, '_commit_hash', None),
        effective_training_arguments=dict(batch=1, accumulation=16, eval_batch=8, total_updates=total,
            warmup_steps=math.ceil(total * .1), optimizer='adamw_torch', scheduler='linear',
            class_weights=weights.tolist(), bf16=bool(torch.cuda.is_bf16_supported()), parameter_dtype='float32'))
    write_json(output / 'provenance.json', provenance)
    args = TrainingArguments(output_dir=str(output / 'checkpoints'), num_train_epochs=cfg['epochs'],
        learning_rate=cfg['learning_rate'], per_device_train_batch_size=1, gradient_accumulation_steps=16,
        per_device_eval_batch_size=8, warmup_steps=math.ceil(total * .1), weight_decay=.01, max_grad_norm=1.,
        optim='adamw_torch', lr_scheduler_type='linear', bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True, eval_strategy='epoch',
        save_strategy='epoch', load_best_model_at_end=True, metric_for_best_model='macro_f1',
        greater_is_better=True, save_total_limit=2, report_to='none', dataloader_num_workers=0, seed=seed)
    def compute_metrics(prediction):
        if not np.isfinite(prediction.predictions).all():
            raise FloatingPointError('Nonfinite validation logits')
        return dict(macro_f1=f1_score(prediction.label_ids, prediction.predictions.argmax(1),
                                     labels=range(5), average='macro', zero_division=0))
    trainer = WeightedClassificationTrainer(model=model, args=args, train_dataset=train_ds,
          eval_dataset=validation_ds, processing_class=tok, data_collator=DataCollatorWithPadding(tok),
          class_weights=weights, label_smoothing=cfg['label_smoothing'], compute_metrics=compute_metrics)
    trainer.train()
    destination.mkdir(parents=True)
    trainer.save_model(str(destination))
    tok.save_pretrained(destination)
    trainer.state.save_to_json(str(destination / 'training_state.json'))
    write_json(destination / 'inference_config.json', dict(**cfg, labels=LABELS, ctx_version=3,
               other_bias=0., loss_normalization='microbatch_mean_trainer_accumulation'))
    write_json(destination / 'training_provenance.json', provenance)
    del trainer, model, train_ds, validation_ds, tok
    gc.collect()
    torch.cuda.empty_cache()
    manifest = inference_manifest(destination)
    probabilities, benchmark = predict_local(destination, validation)
    if inference_manifest(destination) != manifest:
        raise ValueError('Inference pipeline or numeric settings changed during evaluation')
    save_evaluation(output, 'validation', validation, probabilities, dict(evaluation_kind='validation_selection',
        config=cfg, benchmark=benchmark, train_sha256=fingerprint(train), validation_sha256=fingerprint(validation),
        model_weight_sha256=sha(destination / 'model.safetensors'), validation_inference_manifest=manifest))
    write_json(output / 'run_status.json', dict(state='complete'))


def summarize():
    rows, groups, per_class = [], {}, []
    for seed in SEEDS:
        for variant in VARIANTS:
            name = f'{variant}_seed_{seed}'
            folder = STUDY / 'experiments' / name
            if not (folder / 'run_status.json').is_file() or read_json(folder / 'run_status.json')['state'] != 'complete':
                continue
            result = read_json(folder / 'validation_metrics.json')
            if result['evaluation_kind'] != 'validation_selection':
                raise ValueError('Only full validation runs may be aggregated')
            row = dict(run=name, variant=variant, seed=seed, macro_f1=result['macro_f1'], accuracy=result['accuracy'])
            rows.append(row)
            groups.setdefault(variant, []).append(row)
            per_class.extend(dict(run=name, label=label, **values) for label, values in result['per_class'].items())
    if not rows:
        return
    pd.DataFrame(rows).to_csv(STUDY / 'completed_experiments.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(STUDY / 'per_class_comparison.csv', index=False, encoding='utf-8-sig')
    summaries = []
    for variant, items in groups.items():
        values = [item['macro_f1'] for item in items]
        summaries.append(dict(variant=variant, n=len(values), seeds=[item['seed'] for item in items],
            mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    paired = []
    for seed in SEEDS:
        by_variant = {row['variant']: row for row in rows if row['seed'] == seed}
        if set(by_variant) == set(VARIANTS):
            paired.append(dict(seed=seed, baseline_f1=by_variant['baseline']['macro_f1'],
                context_f1=by_variant['context']['macro_f1'],
                delta=by_variant['context']['macro_f1'] - by_variant['baseline']['macro_f1']))
    write_json(STUDY / 'summary.json', dict(state='complete' if len(rows) == 6 else 'pending',
               completed_runs=len(rows), expected_runs=6, groups=summaries, paired_seed_deltas=paired,
               test_or_final_used=False, labels_changed=False, release_v1_replaced=False))
    text = '# So sánh classifier trên train đã loại trùng toàn nguồn\n\n'
    text += 'Hai phương án được train lại trên cùng tập đã purge nguồn trùng validation. Không so với weights cũ như một baseline công bằng. Nhãn validation và thứ tự giữ nguyên; chỉ Section khác giữa hai nhánh. Kết quả hiện tại là validation phát triển, không phải final test.\n\n'
    text += table(['Run', 'Seed', 'Variant', 'Macro-F1', 'Accuracy'], [[r['run'], r['seed'], r['variant'], f"{r['macro_f1']:.6f}", f"{r['accuracy']:.6f}"] for r in rows])
    text += '\n\n' + table(['Variant', 'Số seed xong', 'Mean', 'SD mẫu'], [[s['variant'], s['n'], f"{s['mean']:.6f}", f"{s['sample_std']:.6f}" if s['sample_std'] is not None else 'chưa đủ'] for s in summaries])
    text += '\n\n' + table(['Seed', 'Baseline', 'Context v2.1', 'Chênh lệch'], [[p['seed'], f"{p['baseline_f1']:.6f}", f"{p['context_f1']:.6f}", f"{p['delta']:+.6f}"] for p in paired])
    text += '\n\nChỉ kết luận độ ổn định sau khi đủ ba seed của mỗi phương án; không bỏ lượt thấp. Chọn cấu hình theo mean validation của đủ ba seed, sau đó checkpoint tốt nhất trong cấu hình đó là ứng viên đề xuất; không tự thay production hoặc thay bản v1 đã đóng băng. Giới hạn: nhãn gốc chưa hoàn tất phân xử người, near-duplicate nguồn chưa được chứng nhận hết và validation đã từng sử dụng trong phát triển.\n'
    (STUDY / 'KET_QUA_DOI_CHUNG.md').write_text(text, encoding='utf-8')
    if len(rows) == 6:
        winner = max(summaries, key=lambda item: item['mean'])['variant']
        best = max(groups[winner], key=lambda item: item['macro_f1'])
        write_json(STUDY / 'selection_recommendation.json', dict(
            status='VALIDATION_RECOMMENDATION; not automatically deployed',
            rule='configuration mean over all three seeds, then best checkpoint within that configuration',
            recommended_run=best['run'], macro_f1=best['macro_f1'], variant=winner,
            model_dir=str(ROOT / 'models/controlled_study' / STUDY.name / best['run']),
            final_test_used=False, production_replaced=False, original_release_preserved=True))


def queue():
    plan, _ = prepare()
    lock = STUDY / 'queue.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = dict(state='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                 planned=[e['name'] for e in plan['experiments']], completed=[], current=None)
    try:
        for experiment in plan['experiments']:
            name = experiment['name']
            folder = STUDY / 'experiments' / name
            if (folder / 'run_status.json').exists() and read_json(folder / 'run_status.json')['state'] == 'complete':
                state['completed'].append(name)
                continue
            state['current'] = name
            write_json(STUDY / 'queue_status.json', state)
            with (STUDY / f'{name}.stdout.log').open('a', encoding='utf-8') as stdout, (STUDY / f'{name}.stderr.log').open('a', encoding='utf-8') as stderr:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', name],
                                        cwd=ROOT, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect logs before retry')
            state['completed'].append(name)
            summarize()
            write_json(STUDY / 'queue_status.json', state)
        state.update(state='complete', current=None)
    except BaseException as exc:
        state.update(state='stopped', error=str(exc))
        raise
    finally:
        write_json(STUDY / 'queue_status.json', state)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', choices=[f'{variant}_seed_{seed}' for seed in SEEDS for variant in VARIANTS])
    group.add_argument('--queue', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print('Context study prepared; original release unchanged; no training yet.')
    elif args.run is not None:
        run(args.run)
    else:
        queue()
