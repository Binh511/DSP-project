"""Comparable extraction outputs and a human reference-based scorer."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import extraction
from planning import generate_plan
from study_data import OUT, read_json, write_json, sha
from study_evaluation import predict_local, fingerprint
from train_human_cv import ROOT, LABELS


def prepare_pipeline():
    folder = OUT / 'pipeline_review'
    if folder.exists():
        raise FileExistsError('Pipeline packet exists; do not overwrite review assignments')
    manifest = read_json(ROOT / 'runs/fair_fold0/frozen_split.json')
    docs = manifest['test_document_ids'][:3]
    documents, inputs = {}, []
    for doc in docs:
        paths = sorted((ROOT / 'data_cleaned' / doc).glob('*.md')) + sorted((ROOT / 'data_cleaned' / doc).glob('*.txt'))
        if len(paths) != 1:
            raise ValueError('Ambiguous or missing cleaned source for ' + doc)
        text = paths[0].read_text(encoding='utf-8-sig')
        documents[doc] = dict(text=text, sha256=sha(paths[0]), items=list(extraction.sentences_with_context(text)))
        for i in documents[doc]['items']:
            inputs.append(dict(document_id=doc, text=i['text'], section=i['section'],
                               prev_text=i['prev'], next_text=i['next_text'], pos=i['pos'],
                               char_start=i['start'], char_end=i['end']))
    rows = pd.DataFrame(inputs)
    probs, benchmarks = [], []
    members = ['final_s11', 'final_s22', 'final_s33']
    for member in members:
        p, benchmark = predict_local(ROOT / 'models' / member, rows)
        probs.append(p)
        benchmarks.append(dict(member=member, **benchmark))
    averaged = np.mean(probs, axis=0)
    predictions = [{'label': LABELS[int(p.argmax())], 'score': round(float(p.max()), 3)} for p in averaged]
    folder.mkdir(parents=True)
    rows['gold_fields'], rows['annotation_reason'] = '', ''
    # Reference sheet does NOT reveal predictions to annotators.
    rows.to_csv(folder / 'reference_to_complete.csv', index=False, encoding='utf-8-sig')
    offset, outputs = 0, {'rules': [], 'classifier': []}
    for doc in docs:
        n = len(documents[doc]['items'])
        document_predictions = predictions[offset:offset + n]
        outputs['rules'].append(extraction.extract_document(doc, documents[doc]['text'], use_model=False))
        import ensemble_classifier
        with patch.object(ensemble_classifier, 'classify', return_value=document_predictions):
            outputs['classifier'].append(extraction.extract_document(doc, documents[doc]['text'], use_model=True))
        offset += n
    for name, records in outputs.items():
        write_json(folder / f'{name}_outputs.json', records)
        write_json(folder / f'{name}_plans.json', [generate_plan(r) for r in records])
    write_json(folder / 'protocol.json', dict(document_ids=docs, source_hashes={d: documents[d]['sha256'] for d in docs},
               reference_schema='Every sentence: semicolon-separated gold_fields or NONE; blank means incomplete',
               reference_sha256=sha(folder / 'reference_to_complete.csv'),
               reference_rows_sha256=fingerprint(rows.drop(columns=['gold_fields', 'annotation_reason'])),
               scope='Source sentence-field exact-span extraction; same documents; rules vs active ensemble plus same postprocessing',
               retrieval='Not benchmarked: no identical frozen retrieval output supplied. Do not label this retrieval-only vs classifier+retrieval.',
               limitations='Inspected development documents; not independent final test. Planning needs separate human quality scoring.',
               members=members, member_benchmarks=benchmarks))


def score_pipeline():
    folder = OUT / 'pipeline_review'
    reference = pd.read_csv(folder / 'reference_to_complete.csv', keep_default_na=False, float_precision='round_trip')
    protocol = read_json(folder / 'protocol.json')
    if fingerprint(reference.drop(columns=['gold_fields', 'annotation_reason'])) != protocol['reference_rows_sha256']:
        raise ValueError('Reference content/offsets changed; only annotations may be edited')
    if reference.gold_fields.str.strip().eq('').any():
        raise ValueError('Human reference incomplete; blank is not equivalent to NONE')
    fields = set(extraction.FIELD_CUES)
    gold = set()
    for r in reference.itertuples():
        selected = {s.strip() for s in r.gold_fields.split(';')}
        if selected == {'NONE'}:
            continue
        if not selected <= fields:
            raise ValueError('Invalid gold fields: ' + r.gold_fields)
        gold |= {(r.document_id, int(r.char_start), int(r.char_end), f) for f in selected}
    results = []
    for mode in ('rules', 'classifier'):
        predicted = set()
        for record in read_json(folder / f'{mode}_outputs.json'):
            for field, entries in record['fields'].items():
                for e in entries:
                    if e['verified']:
                        predicted.add((record['document_id'], int(e['char_start']), int(e['char_end']), field))
        per_field = {}
        for field in sorted(fields) + ['ALL']:
            a = {x for x in gold if field == 'ALL' or x[3] == field}
            b = {x for x in predicted if field == 'ALL' or x[3] == field}
            tp, fp, fn = len(a & b), len(b - a), len(a - b)
            precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
            per_field[field] = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                                    f1=2 * precision * recall / (precision + recall) if precision + recall else 0.)
        results.append(dict(mode=mode, scores=per_field))
    write_json(folder / 'extraction_metrics.json', dict(reference_sha256=sha(folder / 'reference_to_complete.csv'),
               scope='exact source sentence-field extraction on reviewed development documents', results=results))
    return results
