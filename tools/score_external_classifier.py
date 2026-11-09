"""Score cached production probabilities against the preselected AI reference.

This never runs a classifier or certifies references as human annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction import MIN_SCORE_BY_LABEL, sentences_with_context
from psai_project.study_metrics import LABELS, classification_metrics


def aligned_labels(source, references, predictions):
    """Check every source position and probability row before computing labels."""
    import math
    source_hash = hashlib.sha256(source.encode('utf-8')).hexdigest()
    items = list(sentences_with_context(source))
    if predictions.get('mode') != 'ensemble' or predictions.get('labels') != list(LABELS):
        raise ValueError('Expected captured production-ensemble probabilities with canonical label order')
    rows = predictions.get('items', [])
    if len(items) != len(references) or len(items) != len(rows):
        raise ValueError('Source/reference/prediction item count mismatch')
    truth, raw, thresholded = [], [], []
    for item, reference, prediction in zip(items, references, rows):
        if (reference.get('source_sha256') != source_hash or reference.get('text') != item['text']
                or reference.get('char_start') != item['start'] or reference.get('char_end') != item['end']
                or reference.get('label') not in LABELS):
            raise ValueError('Reference does not match the frozen source segmentation')
        expected_input = {'text': item['text'], 'section': item['section'], 'prev_text': item['prev'],
                          'pos': item['pos'], 'next_text': item['next_text']}
        if prediction.get('input') != expected_input:
            raise ValueError('Captured classifier input differs from the reference source context')
        probabilities = prediction.get('probabilities', [])
        if (len(probabilities) != len(LABELS)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-5)):
            raise ValueError('Invalid cached probability distribution')
        label = LABELS[max(range(len(LABELS)), key=lambda index: probabilities[index])]
        if prediction.get('label') != label:
            raise ValueError('Captured label disagrees with probability argmax')
        # Match ensemble_classifier.classify's rounded score, then the unchanged
        # extraction thresholds. This is not calibrated correctness probability.
        score = round(float(max(probabilities)), 3)
        effective = label if label == 'OTHER' or score >= MIN_SCORE_BY_LABEL.get(label, .5) else 'OTHER'
        truth.append(reference['label'])
        raw.append(label)
        thresholded.append(effective)
    return truth, raw, thresholded


def score(study, reference):
    study, reference = Path(study), Path(reference)
    annotations = json.loads(reference.read_text('utf-8'))
    by_hash = {doc['source_sha256']: doc for doc in annotations['documents']}
    attempts = json.loads((study / 'attempts.json').read_text('utf-8'))
    per_document, truth, raw, thresholded = [], [], [], []
    for attempt in attempts['attempts']:
        if attempt['mode'] != 'full':
            continue
        # Prefer the source-relative study layout after an archive is moved.
        local_run = study / attempt['mode'] / attempt['document_id']
        candidates = [local_run, Path(attempt['output']),
                      *(local_run.parent / Path(path).name for path in attempt['diagnostic_directories']),
                      *map(Path, attempt['diagnostic_directories'])]
        run = next((path for path in candidates if (path / 'jobs/extract.response.json').is_file()), None)
        if run is None:
            per_document.append({'document_id': attempt['document_id'], 'status': 'not_scored_no_extract_response',
                                 'full_pipeline_completed': attempt['returncode'] == 0})
            continue
        extraction = json.loads((run / 'jobs/extract.response.json').read_text('utf-8'))
        if extraction.get('status') != 'completed':
            per_document.append({'document_id': attempt['document_id'], 'status': 'not_scored_extract_failed',
                                 'full_pipeline_completed': False})
            continue
        source_raw = (run / 'source.txt').read_bytes()
        source_hash = hashlib.sha256(source_raw).hexdigest()
        if extraction.get('source_sha256') != source_hash or source_hash not in by_hash:
            raise ValueError('Cached extraction has no matching frozen reference/source binding')
        actual = extraction['classifier_predictions']
        labels = aligned_labels(source_raw.decode('utf-8'), by_hash[source_hash]['sentence_references'], actual)
        for collection, values in zip((truth, raw, thresholded), labels):
            collection.extend(values)
        per_document.append({'document_id': attempt['document_id'], 'source_sha256': source_hash,
                             'status': 'scored_AI_reference', 'full_pipeline_completed': attempt['returncode'] == 0,
                             'raw': classification_metrics(labels[0], labels[1]),
                             'thresholded': classification_metrics(labels[0], labels[2])})
    return {'evaluation_kind': 'production_ensemble_external_AI_reference_diagnostic',
            'reference_file_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            'reference_reviewer': annotations['reviewer'], 'human_ground_truth': False, 'independent_final_test': False,
            'documents': per_document, 'scored_sentences': len(truth),
            'scored_sources': sum(doc['status'] == 'scored_AI_reference' for doc in per_document),
            'attempted_sources': len(per_document),
            'raw': classification_metrics(truth, raw) if truth else None,
            'thresholded': classification_metrics(truth, thresholded) if truth else None,
            'aggregation_scope': 'Only source-aligned successful classifier stages, including ones whose later planner failed; coverage disclosed separately',
            'note': 'These diagnostics belong to the deployed three-member ensemble, not classifier research v2 validation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = score(args.study, args.reference)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'scored_sources': result['scored_sources'], 'scored_sentences': result['scored_sentences'],
                      'raw_macro_f1': result['raw']['macro_f1_fixed_labels'] if result['raw'] else None}, indent=2))


if __name__ == '__main__':
    main()
