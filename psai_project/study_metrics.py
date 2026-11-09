"""Small transparent metrics for explicitly scoped, externally reviewed pilots."""
from __future__ import annotations

from collections import Counter

LABELS = ('OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER')


def classification_metrics(reference, prediction, labels=LABELS):
    reference, prediction = list(reference), list(prediction)
    if not reference or len(reference) != len(prediction):
        raise ValueError('Classification requires equally sized nonempty label lists')
    labels = tuple(labels)
    if len(set(labels)) != len(labels) or not set(reference + prediction) <= set(labels):
        raise ValueError('Unknown or duplicated classification label')
    counts = Counter(zip(reference, prediction))
    rows = {}
    for label in labels:
        tp = counts[label, label]
        support = sum(counts[label, other] for other in labels)
        predicted = sum(counts[other, label] for other in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        rows[label] = {'support': support, 'predicted': predicted, 'tp': tp,
                       'precision': precision, 'recall': recall, 'f1': f1}
    active = [row for row in rows.values() if row['support'] or row['predicted']]
    return {'n': len(reference), 'accuracy': sum(a == b for a, b in zip(reference, prediction)) / len(reference),
            'macro_f1_fixed_labels': sum(row['f1'] for row in rows.values()) / len(labels),
            'macro_f1_active_labels': sum(row['f1'] for row in active) / len(active),
            'per_label': rows, 'labels': list(labels),
            'confusion_matrix_rows_reference_columns_prediction': [[counts[a, b] for b in labels] for a in labels],
            'zero_division_policy': 'Per-class precision/recall/F1 use zero when undefined; both fixed and active-label macro-F1 are reported',
            'reference_quality_certified': False}


def agreement_metrics(first, second, labels=LABELS):
    first, second = list(first), list(second)
    if not first or len(first) != len(second) or not set(first + second) <= set(labels):
        raise ValueError('Agreement requires matched nonempty valid labels')
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    a, b = Counter(first), Counter(second)
    expected = sum(a[label] * b[label] for label in labels) / (n * n)
    return {'n': n, 'agreement': observed, 'disagreements': sum(a != b for a, b in zip(first, second)),
            'cohen_kappa': (observed - expected) / (1 - expected) if expected < 1 else None,
            'kappa_status': 'defined' if expected < 1 else 'undefined_constant_identical_labels',
            'scope': 'Agreement of the supplied raters; AI raters do not establish human annotation reliability'}
