"""Prepare new-document labels and enforce selection before final evaluation."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from extraction import sentences_with_context
from study_data import OUT, read_json, write_json, sha
from study_evaluation import fingerprint, predict_local, save_evaluation
from train_human_cv import ROOT, LABELS
from final_test_integrity import (inference_manifest, validate_validation_binding,
                                 known_development_artifacts, verify_screening,
                                 verify_candidate_sources)


def screen_final_candidates():
    """Screen known development text only; cannot certify unknown old histories."""
    from train_human_cv import text_key
    folder = OUT / 'final_test_candidate'
    rows = pd.read_csv(folder / 'labels_to_complete.csv', keep_default_na=False, float_precision='round_trip')
    original = pd.read_csv(ROOT / 'data_segment_cls/segment_classification.csv').fillna('')
    human = pd.read_csv(ROOT / 'extra_human.csv').fillna('')
    used = pd.concat([original, human], ignore_index=True)
    known_texts = set(used.text.map(text_key))
    repeated = rows[rows.text.map(text_key).isin(known_texts)].copy()
    repeated.to_csv(folder / 'sentence_overlap_for_review.csv', index=False, encoding='utf-8-sig')
    def grams(text):
        words = re.findall(r'\w+', text.casefold())
        return set(zip(*(words[i:] for i in range(6))))
    reference = {}
    for doc, group in used.groupby('document_id'):
        reference[doc] = grams(' '.join(group.text.astype(str)))
    screening = []
    inventory = known_development_artifacts()
    for doc, group in rows.groupby('document_id'):
        candidate = grams(' '.join(group.text.astype(str)))
        best, similarity = '', 0.
        for other_doc, other in reference.items():
            union = len(candidate | other)
            score = len(candidate & other) / union if union else 0.
            if score > similarity:
                best, similarity = other_doc, score
        screening.append(dict(document_id=doc, closest_known_document=best,
                              word6gram_jaccard=similarity,
                              exact_sentence_overlaps=int(repeated.document_id.eq(doc).sum()),
                              development_exposures=inventory['documents'].get(doc, []),
                              disposition='MANUAL_REVIEW_REQUIRED', reviewed_by=''))
    write_json(folder / 'screening.json', dict(documents=screening,
               candidate_rows_sha256=fingerprint(rows.drop(columns='label')),
               development_artifact_hashes=inventory['hashes'], read_errors=inventory['read_errors'],
               reference_scope='Known original, human, pseudo labels, annotation and review/tuning artifacts; unknown histories unavailable',
               warning='Low overlap is not proof of independence. Boilerplate overlap must be reviewed, not auto-deleted.'))
    return screening


def prepare_final_candidates():
    folder = OUT / 'final_test_candidate'
    if folder.exists():
        raise FileExistsError('Final candidate already exists; never replace an assigned packet')
    packs = sorted((ROOT / 'runs/independent_test').glob('fresh_test_pack_*'))
    if not packs:
        raise FileNotFoundError('No new-document candidate pack found')
    files = sorted(packs[-1].glob('*.md'))
    known_docs = set(pd.read_csv(ROOT / 'data_segment_cls/segment_classification.csv').document_id) | set(pd.read_csv(ROOT / 'extra_human.csv').document_id)
    rows = []
    documents, document_paths = {}, {}
    for path in files:
        doc = path.stem
        if doc in known_docs:
            raise ValueError('Candidate document appears in known labeled development data: ' + doc)
        documents[doc] = sha(path)
        document_paths[doc] = str(path.relative_to(ROOT)).replace('\\', '/')
        for item in sentences_with_context(path.read_text(encoding='utf-8-sig')):
            if item['start'] < 0:
                raise ValueError('Candidate has invalid source offset')
            rows.append(dict(item_id=hashlib.sha256(f'{doc}:{item["start"]}:{item["text"]}'.encode()).hexdigest(),
                             document_id=doc, text=item['text'], section=item['section'],
                             prev_text=item['prev'], next_text=item['next_text'], pos=item['pos'],
                             char_start=item['start'], char_end=item['end'], label=''))
    if not rows:
        raise ValueError('No candidate rows')
    folder.mkdir(parents=True)
    sheet = pd.DataFrame(rows)
    sheet.to_csv(folder / 'labels_to_complete.csv', index=False, encoding='utf-8-sig')
    write_json(folder / 'review_gate.json', dict(document_hashes=documents, document_paths=document_paths,
               candidate_rows_sha256=fingerprint(sheet.drop(columns='label')),
               known_development_documents_disjoint=True,
               approved_for_final_test=False, independence_reviewed_by='', annotation_reviewed_by='',
               warning='These documents were previously screened, not certified independent. No labels inferred by AI. Human review and valid labels required.'))


def freeze_selection(model_dir, validation_metrics):
    model_dir = Path(model_dir).resolve()
    metric_file = Path(validation_metrics)
    result = read_json(metric_file)
    if result.get('evaluation_kind') != 'validation_selection':
        raise ValueError('Choose only from completed validation experiments, never diagnostic/test/smoke scores')
    if result.get('model_weight_sha256') != sha(model_dir / 'model.safetensors'):
        raise ValueError('Selected weights do not match recorded validation evaluation')
    target = OUT / 'final_selection.json'
    if target.exists():
        raise FileExistsError('Selection already frozen; cannot select again after seeing final test')
    if not (model_dir / 'training_provenance.json').exists():
        raise ValueError('Missing training provenance')
    if read_json(model_dir / 'training_provenance.json').get('max_steps') is not None or (model_dir / 'invalid_checkpoint.json').exists():
        raise ValueError('Smoke or invalid checkpoints cannot be final candidates')
    validate_validation_binding(result, read_json(model_dir / 'training_provenance.json'),
                                read_json(model_dir / 'inference_config.json'))
    manifest = inference_manifest(model_dir)
    if result.get('validation_inference_manifest') != manifest:
        raise ValueError('Validation must bind the complete inference manifest. Replay validation into a new result; preserve historical scores.')
    write_json(target, dict(model_dir=str(model_dir), model_sha256=sha(model_dir / 'model.safetensors'),
                           inference_config_sha256=sha(model_dir / 'inference_config.json'),
                           validation_metrics_sha256=sha(metric_file),
                           training_provenance_sha256=sha(model_dir / 'training_provenance.json'),
                           validation_metrics_path=str(metric_file.resolve()),
                           inference_manifest=manifest,
                           criterion='validation macro-F1; selected before opening final labels'))


def evaluate_final():
    folder = OUT / 'final_test_candidate'
    selected = read_json(OUT / 'final_selection.json')
    gate = read_json(folder / 'review_gate.json')
    if gate['approved_for_final_test'] is not True or not gate['independence_reviewed_by'].strip() or not gate['annotation_reviewed_by'].strip():
        raise ValueError('Human independence and annotation review are not complete')
    if not (folder / 'screening.json').exists():
        raise ValueError('Known-development overlap screening must be completed first')
    rows = pd.read_csv(folder / 'labels_to_complete.csv', keep_default_na=False, float_precision='round_trip')
    if rows.empty or not rows.label.isin(LABELS).all() or rows.item_id.duplicated().any():
        raise ValueError('Final labels must all be valid, with unique item IDs')
    if fingerprint(rows.drop(columns='label')) != gate['candidate_rows_sha256']:
        raise ValueError('Candidate content changed after packet preparation')
    verify_candidate_sources(gate, rows)
    verify_screening(read_json(folder / 'screening.json'), fingerprint(rows.drop(columns='label')),
                     known_development_artifacts(), set(rows.document_id))
    model_dir = Path(selected['model_dir'])
    if selected.get('inference_manifest') != inference_manifest(model_dir):
        raise ValueError('Frozen model/tokenizer/configuration or inference implementation changed')
    metric_file = Path(selected['validation_metrics_path'])
    if sha(metric_file) != selected['validation_metrics_sha256']:
        raise ValueError('Frozen validation evidence changed')
    for filename, key in [('model.safetensors', 'model_sha256'), ('inference_config.json', 'inference_config_sha256'), ('training_provenance.json', 'training_provenance_sha256')]:
        if sha(model_dir / filename) != selected[key]:
            raise ValueError('Frozen model or provenance changed')
    train_ids = set(read_json(model_dir / 'training_provenance.json')['train_document_ids'])
    val_ids = set(read_json(model_dir / 'training_provenance.json')['validation_document_ids'])
    if set(rows.document_id) & (train_ids | val_ids):
        raise ValueError('Final documents overlap training or validation')
    attempt = OUT / 'final_evaluation_attempt.json'
    # Reserve before inference; a failed run remains visible and cannot silently
    # reopen a test set. No final labels are used during model selection.
    identity = dict(selection=selected, labels_sha256=sha(folder / 'labels_to_complete.csv'),
                    review_gate_sha256=sha(folder / 'review_gate.json'), screening_sha256=sha(folder / 'screening.json'))
    # Atomic exclusive creation prevents two concurrent final evaluations.
    with attempt.open('x', encoding='utf-8') as stream:
        json.dump(dict(**identity, state='started'), stream, indent=2, allow_nan=False)
    try:
        probs, benchmark = predict_local(model_dir, rows)
        report = save_evaluation(OUT / 'final_evaluation', 'selected_model', rows, probs,
                                 dict(evaluation_kind='human_reviewed_final_test', benchmark=benchmark, selection=selected,
                                      review_gate=gate, dataset_sha256=fingerprint(rows)))
        write_json(attempt, dict(**identity, state='complete', macro_f1=report['macro_f1']))
    except Exception as exc:
        write_json(attempt, dict(**identity, state='failed_requires_review', error=str(exc)))
        raise
