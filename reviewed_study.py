"""Versioned training-only label revision and sequential controlled experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

from study_data import ROOT, OUT, sha, read_json, write_json
from study_evaluation import fingerprint
from study_experiments import create_plan, run_experiment, split_data, aggregate
from train_human_cv import LABELS

STUDY = OUT / 'studies' / 'user_labels_v1'
OVERLAY = STUDY / 'label_overlay.csv'


def freeze_code():
    path = STUDY / 'experiment_plan.json'
    plan = read_json(path)
    files = ['study_inputs.py', 'study_experiments.py', 'study_data.py', 'study_evaluation.py',
             'classifier_training.py', 'fair_fold0.py', 'train_human_cv.py', 'reviewed_study.py']
    hashes = {f: sha(ROOT / f) for f in files}
    if 'code_sha256' in plan:
        if plan['code_sha256'] != hashes:
            raise ValueError('Code changed after study freeze; use a new study version')
        return
    snapshot = STUDY / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for filename in files:
        shutil.copy2(ROOT / filename, snapshot / filename)
    plan['code_sha256'] = hashes
    write_json(path, plan)


def prepare():
    if OVERLAY.exists():
        create_plan(STUDY, OVERLAY)
        freeze_code()
        return
    source = ROOT / 'psai_ai_labels_200.csv'
    incoming = pd.read_csv(source, dtype=str, keep_default_na=False)
    key = pd.read_csv(OUT / 'annotation/coordinator_key.csv', dtype=str, keep_default_na=False)
    if len(incoming) != 200 or incoming.item_id.duplicated().any() or set(incoming.item_id) != set(key.item_id):
        raise ValueError('Must provide exactly the assigned 200 rows')
    incoming = incoming.set_index('item_id').loc[key.item_id].reset_index()
    if not incoming.annotation.isin(LABELS).all():
        raise ValueError('Invalid labels')
    for col in ('document_id', 'source_line', 'text', 'section', 'prev_text', 'next_text'):
        if not incoming[col].equals(key[col]):
            raise ValueError('Source/context changed: ' + col)
    before, val_before = split_data()
    STUDY.mkdir(parents=True, exist_ok=True)
    incoming['original_label'] = key.label
    incoming.to_csv(OVERLAY, index=False, encoding='utf-8-sig')
    after, val_after = split_data(overlay_path=OVERLAY)
    if fingerprint(val_before) != fingerprint(val_after):
        raise AssertionError('Validation changed during training-label revision')
    if not before.drop(columns='label').equals(after.drop(columns='label')):
        raise AssertionError('Training features or row order changed')
    incoming.loc[incoming.annotation.ne(incoming.original_label)].to_csv(STUDY / 'proposed_label_changes.csv', index=False, encoding='utf-8-sig')
    before.to_csv(STUDY / 'training_original_labels.csv', index=False, encoding='utf-8-sig')
    after.to_csv(STUDY / 'training_revised_labels.csv', index=False, encoding='utf-8-sig')
    write_json(STUDY / 'dataset_revision.json', dict(
        label_origin='user_supplied; independence_not_verified; not_expert_ground_truth',
        authorization='User instructed to proceed using supplied labels; no independent-rater claim',
        source_sha256=sha(source), overlay_sha256=sha(OVERLAY),
        original_dataset_sha256=sha(ROOT / 'data_segment_cls/segment_classification.csv'),
        assigned_rows=len(incoming), proposed_changes=int(incoming.annotation.ne(incoming.original_label).sum()),
        effective_training_changes=int(before.label.ne(after.label).sum()),
        training_rows=len(after), validation_rows=len(val_after), validation_unchanged=True,
        split_policy='Freeze original fold assignments BEFORE label revision; only folds 2/3/4 may change',
        training_features_unchanged=True, original_dataset_modified=False,
        cohen_kappa=None, final_test_evaluated=False))
    create_plan(STUDY, OVERLAY)
    freeze_code()


def execute(name, smoke_steps=None):
    prepare()
    run_name = name if smoke_steps is None else f'{name}_smoke_{smoke_steps}'
    folder = STUDY / 'experiments' / run_name
    existed = folder.exists()
    try:
        run_experiment(name, smoke_steps, STUDY, OVERLAY)
        write_json(folder / 'run_status.json', dict(state='complete', smoke=smoke_steps is not None))
        aggregate(STUDY)
    except Exception as exc:
        if not existed and folder.exists():
            write_json(folder / 'failure.json', dict(state='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


def queue(names):
    prepare()
    allowed = {e['name'] for e in read_json(STUDY / 'experiment_plan.json')['experiments']}
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError('Queue requires unique predeclared experiment names')
    lock = STUDY / 'queue.lock'
    # Exclusive create prevents accidentally starting two jobs on the 6GB GPU.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = dict(pid=os.getpid(), state='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 planned=names, completed=[], current=None)
    try:
        for name in names:
            status = STUDY / 'experiments' / name / 'run_status.json'
            if status.exists() and read_json(status).get('state') == 'complete':
                state['completed'].append(name)
                continue
            state['current'] = name
            write_json(STUDY / 'queue_status.json', state)
            with (STUDY / f'{name}.stdout.log').open('a', encoding='utf-8') as stdout, (STUDY / f'{name}.stderr.log').open('a', encoding='utf-8') as stderr:
                result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', name],
                                        cwd=ROOT, stdout=stdout, stderr=stderr)
            if result.returncode != 0:
                raise RuntimeError(f'{name} exited {result.returncode}; inspect logs before retrying')
            state['completed'].append(name)
            aggregate(STUDY)
        state['state'], state['current'] = 'complete', None
    except BaseException as exc:
        state['state'], state['error'] = 'stopped', str(exc)
        raise
    finally:
        write_json(STUDY / 'queue_status.json', state)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run')
    group.add_argument('--queue', nargs='+')
    group.add_argument('--summary', action='store_true')
    parser.add_argument('--smoke-steps', type=int)
    args = parser.parse_args()
    if args.smoke_steps is not None and (not args.run or args.smoke_steps < 1):
        parser.error('--smoke-steps requires --run and a positive count')
    if args.prepare:
        prepare()
        print(read_json(STUDY / 'dataset_revision.json'))
    elif args.run:
        execute(args.run, args.smoke_steps)
    elif args.queue:
        queue(args.queue)
    else:
        aggregate(STUDY)
        print(read_json(STUDY / 'queue_status.json') if (STUDY / 'queue_status.json').exists() else 'No queue started')


if __name__ == '__main__':
    main()
