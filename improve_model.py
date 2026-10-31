"""Train with document validation and select the checkpoint by macro-F1.

python improve_model.py --audit
python improve_model.py --epochs 5
Test and human holdout are evaluated only after checkpoint selection.
"""
import argparse
import hashlib
import json

import pandas as pd

from fair_fold0 import partitions, checksum, ORIGINAL_PATH, HUMAN_PATH
from train_human_cv import (
    ROOT, COLS, L2I, text_key, lazy_libs, train_model,
    predict_saved_model, metrics, print_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    _, train, test, val, _, ht, hv, _ = partitions()
    # The existing split already purges train against fold0/fold1. Also purge
    # against human holdout, including context where held-out text can recur.
    held = set(pd.concat([test, val, hv]).text.map(text_key))
    train = pd.concat([train[COLS], ht[COLS]], ignore_index=True)
    forbidden = train.text.map(text_key).isin(held)
    for column in ('prev_text', 'next_text'):
        forbidden |= train[column].fillna('').map(text_key).isin(held)
    train = train.loc[~forbidden].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError('Empty training or validation partition')
    assert not set(train.document_id) & set(pd.concat([test, val, hv]).document_id)
    info = {
        'original_sha256': checksum(ORIGINAL_PATH),
        'human_sha256': checksum(HUMAN_PATH),
        'training_sha256': hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
        'train_rows': len(train), 'validation_rows': len(val),
        'test_rows': len(test), 'human_holdout_rows': len(hv),
        'removed_train_rows': int(forbidden.sum()),
        'selection_metric': 'validation macro_f1', 'seed': 42,
        'epochs': args.epochs, 'human_repetitions': 1,
    }
    print(json.dumps(info, indent=2))
    if args.audit:
        return
    model_dir = ROOT / 'models' / 'validation_selected_f0'
    if model_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {model_dir}')
    libs = lazy_libs()
    train_model(train, 42, model_dir, args.epochs, libs, validation=val)
    torch, _, AutoModel, AutoTokenizer, *_ = libs
    results = {}
    for name, rows in [('test_fold0', test), ('human_holdout', hv)]:
        probs = predict_saved_model(model_dir, rows, torch, AutoModel, AutoTokenizer)
        results[name] = metrics(rows.label.map(L2I).to_numpy(), probs)
        print_metrics(name, results[name])
    (model_dir / 'experiment.json').write_text(
        json.dumps({'configuration': info, 'results': results}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
