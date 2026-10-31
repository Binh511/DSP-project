"""Quick diagnostic: evaluate the CURRENT deployed ensemble against 274 human labels.
Run in project root: python eval_human.py
This dataset is biased toward uncertain predictions; don't report as unbiased test.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

import ensemble_classifier  # existing PSAI 3-seed production ensemble

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'extra_human.csv'
LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']
THR = {'OBJECTIVE': .70, 'REQUIREMENT': .60, 'ACTIVITY': .70,
       'EXPECTED_RESULT': .70}


def main():
    d = pd.read_csv(DATA, encoding='utf-8-sig', dtype={'document_id': str})
    d = d[d['label'].isin(LABELS) & d['text'].notna()].copy().reset_index(drop=True)
    for col in ('section', 'prev_text', 'next_text'):
        d[col] = d[col].fillna('')
    d['pos'] = pd.to_numeric(d['pos'], errors='coerce').fillna(0.)
    inputs = [dict(text=r.text, section=r.section, prev_text=r.prev_text,
                   pos=float(r.pos), next_text=r.next_text) for r in d.itertuples()]
    preds = ensemble_classifier.classify(inputs)
    d['pred'] = [r['label'] for r in preds]
    d['score'] = [r['score'] for r in preds]
    d['after_threshold'] = [
        x if x == 'OTHER' or score >= THR[x] else 'OTHER'
        for x, score in zip(d.pred, d.score)
    ]
    print('Evaluation on manually labeled NEW proposal sentences: N=',len(d))
    print('IMPORTANT: samples selected for review; this is NOT a random test sample.')
    for name in ('pred', 'after_threshold'):
        print('\n===', name, '===')
        print('Accuracy:', round(accuracy_score(d.label, d[name]), 4))
        print('Macro-F1:', round(f1_score(d.label, d[name], labels=LABELS,
                                        average='macro', zero_division=0), 4))
        print(classification_report(d.label, d[name], labels=LABELS,
                                    digits=3, zero_division=0))
        print('True positive labels incorrectly assigned OTHER:',
              int(((d.label != 'OTHER') & (d[name] == 'OTHER')).sum()))
    disagreements = d[d.label != d.after_threshold].sort_values('score', ascending=False)
    target = ROOT / 'runs' / 'human_experiment' / 'eval_human_disagreements.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    disagreements.to_csv(target, index=False, encoding='utf-8-sig')
    print('Disagreements for MANUAL review:', len(disagreements), '=>', target)


if __name__ == '__main__':
    main()
