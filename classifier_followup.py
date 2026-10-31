"""Frozen follow-up: original labels/full context, seeds 123 and 2026.

Seed 42 is referenced from the completed first study, never duplicated as a run.
No final-test inference or production model replacement is performed.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from study_data import ROOT, OUT, read_json, write_json, sha
from study_evaluation import fingerprint
from study_experiments import split_data, run_experiment, aggregate

STUDY = OUT / 'studies/classifier_stability_v1'
REFERENCE = OUT / 'studies/user_labels_v1/experiments/baseline_original_labels/validation_metrics.json'
NAMES = ['seed_123', 'seed_2026']
FILES = ['study_inputs.py', 'study_experiments.py', 'study_data.py', 'study_evaluation.py',
         'classifier_training.py', 'fair_fold0.py', 'train_human_cv.py', 'classifier_followup.py']


def prepare():
    reference = read_json(REFERENCE)
    if reference['evaluation_kind'] != 'validation_selection':
        raise ValueError('Reference must be completed validation selection')
    cfg = reference['config']
    if cfg['use_label_overlay'] or cfg['seed'] != 42:
        raise ValueError('Expected original-label seed 42 reference')
    train, val = split_data()
    if fingerprint(train) != reference['train_sha256'] or fingerprint(val) != reference['validation_sha256']:
        raise ValueError('Follow-up data differ from reference')
    plan = dict(selection='validation macro_f1 only; compare stability before model freeze',
                test_policy='No final test access; no production replacement',
                baseline_training_sha256=fingerprint(train), validation_sha256=fingerprint(val),
                overlay_sha256=None,
                reference=dict(path=str(REFERENCE.relative_to(ROOT)), sha256=sha(REFERENCE),
                               model_weight_sha256=reference['model_weight_sha256']),
                experiments=[dict(name=name, group='seed_stability',
                                  config={**cfg, 'seed': seed}, status='not_run')
                             for name, seed in zip(NAMES, [123, 2026])],
                code_sha256={name: sha(ROOT / name) for name in FILES})
    STUDY.mkdir(parents=True, exist_ok=True)
    target = STUDY / 'experiment_plan.json'
    if target.exists():
        if read_json(target) != plan:
            raise ValueError('Frozen follow-up changed; create a new study version')
    else:
        snapshot = STUDY / 'source_snapshot'
        snapshot.mkdir()
        for name in FILES:
            shutil.copy2(ROOT / name, snapshot / name)
        write_json(target, plan)
    return plan


def summarize():
    plan = prepare()
    aggregate(STUDY)
    reference = read_json(REFERENCE)
    records = [('reference_seed_42', reference)]
    for name in NAMES:
        folder = STUDY / 'experiments' / name
        if (folder / 'run_status.json').exists() and read_json(folder / 'run_status.json')['state'] == 'complete':
            records.append((name, read_json(folder / 'validation_metrics.json')))
    fixed = {k: v for k, v in reference['config'].items() if k != 'seed'}
    for name, record in records:
        if ({k: v for k, v in record['config'].items() if k != 'seed'} != fixed
                or record['train_sha256'] != plan['baseline_training_sha256']
                or record['validation_sha256'] != plan['validation_sha256']
                or record['evaluation_kind'] != 'validation_selection'):
            raise ValueError('Incomparable run: ' + name)
    seeds = [r['config']['seed'] for _, r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    values = [r['macro_f1'] for _, r in records]
    write_json(STUDY / 'stability_with_reference.json', dict(
        state='complete' if len(records) == 3 else 'pending',
        config_except_seed=fixed, seeds=seeds, n=len(records),
        mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        runs=[dict(name=name, seed=r['config']['seed'], macro_f1=r['macro_f1'],
                   model_weight_sha256=r['model_weight_sha256']) for name, r in records],
        reference=plan['reference'], final_test_evaluated=False, production_replaced=False))


def execute(name):
    prepare()
    folder = STUDY / 'experiments' / name
    existed = folder.exists()
    try:
        run_experiment(name, study_dir=STUDY)
        write_json(folder / 'run_status.json', dict(state='complete'))
        summarize()
    except Exception as exc:
        if not existed and folder.exists():
            write_json(folder / 'failure.json', dict(state='failed', error=str(exc)))
        raise


def queue():
    prepare()
    lock = STUDY / 'queue.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = dict(pid=os.getpid(), state='running', planned=NAMES, completed=[], current=None,
                 started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for name in NAMES:
            folder = STUDY / 'experiments' / name
            status = folder / 'run_status.json'
            if not (status.exists() and read_json(status)['state'] == 'complete'):
                state['current'] = name
                write_json(STUDY / 'queue_status.json', state)
                with (STUDY / f'{name}.stdout.log').open('a', encoding='utf-8') as stdout, (STUDY / f'{name}.stderr.log').open('a', encoding='utf-8') as stderr:
                    result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', name],
                                            cwd=ROOT, stdout=stdout, stderr=stderr)
                if result.returncode:
                    raise RuntimeError(f'{name} exited {result.returncode}; inspect logs')
            state['completed'].append(name)
            summarize()
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
    group.add_argument('--run', choices=NAMES)
    group.add_argument('--queue', action='store_true')
    group.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        summarize()
    elif args.run:
        execute(args.run)
    elif args.queue:
        queue()
    else:
        summarize()
