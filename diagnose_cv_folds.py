"""Quick NO-TRAIN audit of the saved PSAI kfold_f0 checkpoint.

Place next to train_human_cv.py and run:
    python diagnose_cv_folds.py

This script does not modify dataset, weights, or production inference.
"""
from pathlib import Path
import hashlib
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from train_human_cv import ROOT, LABELS, L2I, read_original, text_key, predict_saved_model


def main():
    path = ROOT / 'data_segment_cls' / 'segment_classification.csv'
    model_dir = ROOT / 'models' / 'kfold_f0'
    if not path.is_file():
        raise SystemExit(f'Missing data: {path}')
    if not (model_dir / 'inference_config.json').exists():
        raise SystemExit(f'Missing old model: {model_dir}')
    df = read_original(path)
    print('Original rows:', len(df), '| documents:', df.document_id.nunique(),
          '| groups:', df.group.nunique())
    print('Current 5-fold split (fold, sentences, documents):')
    print(df.groupby('fold').agg(sentences=('text', 'size'), documents=('document_id', 'nunique')).to_string())
    for i in range(5):
        ids = sorted(df.loc[df.fold == i, 'document_id'].unique())
        digest = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()[:16]
        print(f'  fold {i}: document-ID SHA256 prefix={digest}, first IDs={ids[:6]}')
    test = df[df.fold.eq(0)]
    train = df[df.fold.isin([2, 3, 4])]
    seen_texts = set(train.text.map(text_key))
    overlap = test.text.map(text_key).isin(seen_texts)
    print(f'\nSame normalized sentence in current fold0 test vs folds2,3,4 train: '
          f'{int(overlap.sum())}/{len(test)}')
    print('Old fold0 checkpoint last modified:',
          __import__('datetime').datetime.fromtimestamp((model_dir / 'inference_config.json').stat().st_mtime).isoformat(timespec='seconds'))
    print('\nLoading old kfold_f0 once; inference only, NO training...')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit('This diagnostic currently requires the existing CUDA setup')
    probs = predict_saved_model(model_dir, df, torch, AutoModelForSequenceClassification, AutoTokenizer)
    prediction = probs.argmax(1)
    truth = np.array([L2I[label] for label in df.label])
    print('\nF1 of THE SAME old kfold_f0 checkpoint on each CURRENT fold:')
    print('Original training design: test fold0, validation fold1, TRAIN folds2,3,4.')
    for f in range(5):
        indices = np.where(df.fold.to_numpy() == f)[0]
        score = f1_score(truth[indices], prediction[indices], average='macro', labels=list(range(5)), zero_division=0)
        accuracy = accuracy_score(truth[indices], prediction[indices])
        role = 'TEST' if f == 0 else 'VAL' if f == 1 else 'TRAIN (old design)'
        print(f'  fold {f}: {role:18s} N={len(indices):4d}  macro-F1={score:.4f}  accuracy={accuracy:.4f}')
    print('\nReported original kfold_step1.py fold0 result was ~0.710; current')
    print('train_human_cv.py reports old checkpoint fold0=0.8521.')
    print('A mismatch is a WARNING, not proof of leakage. Without the original')
    print('training-fold document IDs, we cannot certify checkpoint/fold alignment.')
    print('Do not run new training just to resolve this discrepancy.')


if __name__ == '__main__':
    main()
