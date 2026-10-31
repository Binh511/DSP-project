"""PSAI: audit overlapping sentences in the last fold-0 comparison, WITHOUT training/GPU.

Place in the project root next to train_human_cv.py and run:
    python audit_fold0_overlap.py

Checks current fold construction, exact normalized sentence overlap, and
old/new saved fold-0 predictions. Does not modify source labels or models.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from train_human_cv import ROOT, LABELS, read_original, text_key


def scores(data, title):
    if data.empty:
        print(f"{title}: no rows")
        return
    print(f"\n{title}: N={len(data)}")
    for col in ('old', 'new'):
        f1 = f1_score(data['label'], data[col], labels=LABELS,
                      average='macro', zero_division=0)
        acc = accuracy_score(data['label'], data[col])
        print(f"  {col.upper():3s} macro-F1={f1:.4f} accuracy={acc:.4f}")
    print('  Note: subset Macro-F1 can be sensitive to class coverage.')


def main():
    data_path = ROOT / 'data_segment_cls' / 'segment_classification.csv'
    preds_path = ROOT / 'runs' / 'human_experiment' / 'cv_fold0_targeted.csv'
    if not data_path.is_file():
        raise SystemExit(f'Missing dataset: {data_path}')
    if not preds_path.is_file():
        raise SystemExit(f'Missing predictions: {preds_path}\n'
                         'This file is created by python train_human_cv.py --mode cv --folds 0')

    df = read_original(data_path)
    test = df[df['fold'].eq(0)].reset_index(drop=True).copy()
    train = df[df['fold'].isin([2, 3, 4])].copy()
    val = df[df['fold'].eq(1)].copy()
    pred = pd.read_csv(preds_path, encoding='utf-8-sig', dtype=str,
                       keep_default_na=False)
    expected = ['document_id', 'label', 'old', 'new']
    if not set(expected).issubset(pred.columns):
        raise SystemExit(f'CSV does not contain {expected}')
    if len(pred) != len(test) or not pred['document_id'].eq(test['document_id']).all() \
       or not pred['label'].eq(test['label']).all():
        raise SystemExit('STOP: current dataset fold0 does not align with saved predictions. '
                         'Do not interpret these scores; labels/splits may have changed.')

    test = test.assign(old=pred['old'].to_numpy(), new=pred['new'].to_numpy())
    train_norms = train['text'].map(text_key)
    val_norms = val['text'].map(text_key)
    test['norm'] = test['text'].map(text_key)
    test['in_train'] = test['norm'].isin(set(train_norms))
    test['in_val'] = test['norm'].isin(set(val_norms))
    test['in_any_other_fold'] = test['in_train'] | test['in_val']

    print('=== CURRENT SPLIT, FOLD 0 ===')
    print(f'Test: {len(test)}, nominal train: {len(train)}, nominal val: {len(val)}')
    print('Test sentences duplicated in nominal train:', int(test['in_train'].sum()))
    print('Test sentences duplicated in nominal val:', int(test['in_val'].sum()))
    print('Test sentences duplicated in train OR val:',
          int(test['in_any_other_fold'].sum()))
    print('\nDuplicated test sentences by class:')
    print(pd.crosstab(test['label'], test['in_train'])
          .reindex(index=LABELS, columns=[False, True], fill_value=0).to_string())

    label_sets = train.assign(norm=train_norms).groupby('norm')['label'].agg(lambda x: set(x))
    dup = test[test['in_train']].copy()
    conflicts = sum(r.label not in label_sets[r.norm] for r in dup.itertuples())
    print(f'Overlap test rows whose gold label is absent from their train duplicates: '
          f'{conflicts}/{len(dup)}')

    scores(test, 'FULL FOLD0 (contains duplicated texts)')
    scores(test[test['in_train']], 'OVERLAPPING TEST SENTENCES only')
    scores(test[~test['in_train']], 'NON-OVERLAPPING TEST SENTENCES only')
    scores(test[~test['in_any_other_fold']], 'TEST SENTENCES unseen in ANY other current fold')

    out = ROOT / 'runs' / 'human_experiment' / 'fold0_overlap_audit.csv'
    test[['document_id', 'text', 'label', 'old', 'new', 'in_train', 'in_val',
          'in_any_other_fold']].to_csv(out, index=False, encoding='utf-8-sig')
    print('\nSaved per-sentence audit:', out)
    print('IMPORTANT: Checkpoint kfold_f0 was not proven to use THIS split. '
          'Filtering duplicate sentences alone does not validate its old score.')
    print('For a trustworthy gain, freeze splits and train a fresh baseline '
          'and modified model under identical conditions.')


if __name__ == '__main__':
    main()
