"""Dataset audit, model provenance registry and blinded annotation packets."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from train_human_cv import ROOT, LABELS, read_original, read_human, text_key

OUT = ROOT / 'reports' / 'research_review'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def registry():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'runs/fair_fold0/frozen_split.json'
    manifest = read_json(manifest_path)
    data = read_original(ROOT / 'data_segment_cls/segment_classification.csv')
    human, holdout, _ = read_human(ROOT / 'extra_human.csv', data)
    rows = []
    for folder in sorted((ROOT / 'models').iterdir()):
        if not (folder / 'inference_config.json').exists():
            continue
        cfg = read_json(folder / 'inference_config.json')
        weights = sorted(folder.glob('*.safetensors'))
        evidence, docs, selection, human_used = 'unknown', [], 'unknown', 'unknown'
        variant = folder / 'frozen_data_sha256.json'
        if variant.exists():
            v = read_json(variant)
            if v['manifest_sha256'] == sha(manifest_path):
                evidence = 'recorded_manifest_matches; no historical weight hash'
                docs = sorted(data.loc[data.fold.isin([2, 3, 4]), 'document_id'].unique())
                human_used = v['variant'] != 'baseline'
                if human_used:
                    docs = sorted(set(docs) | set(human.document_id))
                selection = 'fixed 5 epochs; validation not used'
        experiment = folder / 'experiment.json'
        if experiment.exists() and folder.name == 'validation_selected_f0':
            e = read_json(experiment)['configuration']
            if e['original_sha256'] == sha(ROOT / 'data_segment_cls/segment_classification.csv') and e['human_sha256'] == sha(ROOT / 'extra_human.csv'):
                evidence = 'recorded_dataset_hashes_match; no historical weight hash'
                docs = sorted(set(data.loc[data.fold.isin([2, 3, 4]), 'document_id']) | set(human.document_id))
                human_used, selection = True, 'validation macro-F1'
        row = dict(model=folder.name, weight_sha256={p.name: sha(p) for p in weights},
                   config_sha256=sha(folder / 'inference_config.json'), context_version=cfg.get('ctx_version', 'legacy'),
                   base_model=read_json(folder / 'config.json').get('_name_or_path') or cfg.get('base_model', 'unknown'),
                   training_provenance=evidence, known_train_document_ids=docs,
                   human_labels_used=human_used, checkpoint_selection=selection,
                   current_fold0_eligibility='exploratory_recorded_split' if docs else 'unknown_do_not_rank_as_independent',
                   independent_final_test='not_available',
                   source_candidates=['train_final.py'] if folder.name.startswith('final_s') else
                       ['kfold_step1.py'] if folder.name.startswith('kfold_f') else [],
                   note='Script names and folder names are hypotheses, not proof of training history.')
        rows.append(row)
    write_json(OUT / 'model_registry.json', rows)
    snapshot_digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
    write_json(OUT / 'registry_snapshots' / f'{snapshot_digest}.json', rows)
    flat = [{**r, 'known_train_document_ids': '|'.join(r['known_train_document_ids']),
             'weight_sha256': json.dumps(r['weight_sha256']), 'source_candidates': '|'.join(r['source_candidates'])} for r in rows]
    pd.DataFrame(flat).to_csv(OUT / 'model_registry.csv', index=False, encoding='utf-8-sig')
    # Align saved predictions to current split; this is a reproducibility audit,
    # not proof that a checkpoint did not train on evaluation documents.
    saved = pd.read_csv(ROOT / 'runs/human_experiment/cv_fold0_targeted.csv')
    test = data.loc[data.fold.eq(0)].reset_index(drop=True)
    aligned = len(saved) == len(test) and saved.document_id.equals(test.document_id) and saved.label.equals(test.label)
    historical = read_json(ROOT / 'runs/human_experiment/cv_summary_targeted.json')
    investigation = dict(historical_macro_f1=historical['test_fold_results'][0]['old']['macro_f1'],
                         saved_predictions_align_current_fold0=aligned,
                         saved_predictions_contain_text='text' in saved,
                         original_train_document_manifest_available=False,
                         historical_checkpoint_weight_hash_available=False,
                         conclusion='0.8521 remains a historical diagnostic, not a certified independent test score',
                         active_models=['final_s11', 'final_s22', 'final_s33'],
                         active_model_evidence='ensemble_classifier.MODEL_NAMES; extraction imports ensemble_classifier')
    current = OUT / 'historical_fold0_diagnostic/kfold_f0_predictions.csv'
    if current.exists():
        evaluated = pd.read_csv(current)
        if len(evaluated) == len(saved) and evaluated.document_id.equals(saved.document_id) and evaluated.true_label.equals(saved.label):
            investigation.update(current_macro_f1=read_json(OUT / 'historical_fold0_diagnostic/kfold_f0_metrics.json')['macro_f1'],
                current_predictions_match_saved_fraction=float((evaluated.predicted_label.to_numpy() == saved.old.to_numpy()).mean()),
                reproduction_does_not_certify_training_split=True)
    write_json(OUT / 'investigation_08521.json', investigation)
    error_analysis(data)
    return rows


def error_analysis(original=None):
    target = OUT / 'error_analysis.csv'
    if target.exists():
        prior = pd.read_csv(target, keep_default_na=False)
        if any(prior[c].astype(str).str.strip().ne('').any() for c in ('reviewed_error_type', 'reviewed_gold_label', 'review_reason') if c in prior):
            raise FileExistsError('Human-reviewed error analysis exists; do not overwrite it with a new audit')
    if original is None:
        original = read_original(ROOT / 'data_segment_cls/segment_classification.csv')
    test = original[original.fold.eq(0)].reset_index(drop=True)
    frames, transitions = [], []
    for filename, dataset in [('test_predictions.csv', 'fold0'), ('human_holdout_predictions.csv', 'holdout')]:
        saved = pd.read_csv(ROOT / 'runs/fair_fold0' / filename)
        if dataset == 'fold0':
            for col in ('document_id', 'text', 'label'):
                if not saved[col].equals(test[col]):
                    raise ValueError('Saved predictions no longer match current test: ' + col)
            for col in ('section', 'prev_text', 'next_text', 'source_line'):
                saved[col] = test[col]
        for col, model in [('baseline_prediction', 'fair_fold0_baseline'), ('human_prediction', 'fair_fold0_human_once')]:
            errors = saved[saved.label.ne(saved[col])].copy()
            errors['model'], errors['dataset'] = model, dataset
            errors = errors.rename(columns={'label': 'true_label', col: 'predicted_label'})
            errors['confidence'] = None  # never invent confidence from hard labels
            errors['transition'] = errors.true_label + ' -> ' + errors.predicted_label
            errors['candidate_flags'] = errors.text.map(lambda t: '|'.join(k for k, flag in {
                'long_text': len(t) > 600,
                'encoding_or_section_sign': any(c in t for c in ('�', '§')),
                'modal_cue': any(w in t.lower().split() for w in ('must', 'shall', 'should')),
            }.items() if flag))
            errors['reviewed_error_type'], errors['reviewed_gold_label'], errors['review_reason'] = '', '', ''
            cols = ['model', 'dataset', 'document_id', 'text', 'true_label', 'predicted_label', 'confidence', 'transition', 'candidate_flags', 'reviewed_error_type', 'reviewed_gold_label', 'review_reason']
            cols += [c for c in ('section', 'prev_text', 'next_text', 'source_line') if c in errors]
            frames.append(errors[cols])
            for (truth, pred), n in errors.groupby(['true_label', 'predicted_label']).size().items():
                transitions.append(dict(model=model, dataset=dataset, true_label=truth, predicted_label=pred, n=int(n)))
    pd.concat(frames).to_csv(OUT / 'error_analysis.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(transitions).sort_values('n', ascending=False).to_csv(OUT / 'error_transitions.csv', index=False, encoding='utf-8-sig')


def annotation_packet():
    folder = OUT / 'annotation'
    if folder.exists():
        raise FileExistsError('Annotation packet exists; refusing to change assigned items')
    original = read_original(ROOT / 'data_segment_cls/segment_classification.csv')
    # Review training data; do not relabel the already inspected evaluation data.
    training = original[original.fold.isin([2, 3, 4])]
    sample = pd.concat([g.sample(n=min(40, len(g)), random_state=2026) for _, g in training.groupby('label')])
    sample = sample.copy()
    sample['item_id'] = sample.apply(lambda r: hashlib.sha256(f'{r.document_id}:{r.source_line}:{r.text}'.encode()).hexdigest(), axis=1)
    cols = ['item_id', 'document_id', 'source_line', 'text', 'section', 'prev_text', 'next_text']
    folder.mkdir(parents=True)
    sample[cols + ['label']].to_csv(folder / 'coordinator_key.csv', index=False, encoding='utf-8-sig')
    for name, seed in [('rater_A', 17), ('rater_B', 29)]:
        sheet = sample[cols].sample(frac=1, random_state=seed).copy()
        sheet['annotation'], sheet['reason'] = '', ''
        sheet.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    write_json(folder / 'manifest.json', dict(n=len(sample), seed=2026, per_original_label=sample.label.value_counts().to_dict(),
               source_sha256=sha(ROOT / 'data_segment_cls/segment_classification.csv'),
               warning='Keep coordinator_key.csv away from raters. Labels must be assigned independently. Training-data review only.'))


def validate_assigned_annotations(a, b, assignment):
    expected = pd.read_csv(assignment, dtype=str, keep_default_na=False).set_index('item_id')
    names = []
    for path in (a, b):
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
        if sheet.item_id.duplicated().any() or set(sheet.item_id) != set(expected.index):
            raise ValueError('Annotations must cover exactly the assigned 200 item IDs')
        sheet = sheet.set_index('item_id').loc[expected.index]
        for col in ('document_id', 'source_line', 'text', 'section', 'prev_text', 'next_text'):
            if not sheet[col].equals(expected[col]):
                raise ValueError('Assigned source/context was modified: ' + col)
        if 'reviewer' not in sheet or sheet.reviewer.str.strip().eq('').any() or sheet.reviewer.str.strip().nunique() != 1:
            raise ValueError('Each file must record one named reviewer; use the annotation form')
        if 'independence_ack' not in sheet or not sheet.independence_ack.eq('yes').all():
            raise ValueError('Independent annotation acknowledgement is required')
        names.append(sheet.reviewer.iloc[0].strip().casefold())
    if names[0] == names[1]:
        raise ValueError('Two different reviewers must label independently')


def agreement(a, b, output):
    if Path(a).resolve() == Path(b).resolve():
        raise ValueError('Two distinct independently completed rater files are required')
    frames = [pd.read_csv(p, keep_default_na=False) for p in (a, b)]
    for frame in frames:
        if frame.item_id.duplicated().any() or not frame.annotation.isin(LABELS).all():
            raise ValueError('All items require exactly one valid independent annotation; blank/duplicate/unknown not allowed')
    if set(frames[0].item_id) != set(frames[1].item_id):
        raise ValueError('Raters must label identical item IDs')
    right = frames[1].set_index('item_id').loc[frames[0].item_id]
    if not frames[0].text.reset_index(drop=True).equals(right.text.reset_index(drop=True)):
        raise ValueError('Same ID has different text')
    a_labels, b_labels = frames[0].annotation, right.annotation.to_numpy()
    kappa = float(cohen_kappa_score(a_labels, b_labels, labels=LABELS))
    import math
    result = dict(n=len(frames[0]), agreement=float((a_labels.to_numpy() == b_labels).mean()),
                  cohen_kappa=kappa if math.isfinite(kappa) else None,
                  confusion=confusion_matrix(a_labels, b_labels, labels=LABELS).tolist(), labels=LABELS,
                  rater_a_sha256=sha(a), rater_b_sha256=sha(b))
    write_json(output, result)
    adjudication_path = Path(output).with_name('adjudication_to_complete.csv')
    if not adjudication_path.exists():
        review = frames[0].copy()
        review = review.rename(columns={'annotation': 'annotation_A', 'reason': 'reason_A'})
        review['annotation_B'] = b_labels
        review['agreement'] = a_labels.to_numpy() == b_labels
        review['adjudicated_label'], review['adjudication_reason'], review['reviewer'] = '', '', ''
        review.to_csv(adjudication_path, index=False, encoding='utf-8-sig')
    return result
