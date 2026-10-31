"""CPU checks for document bootstrap and fixed-label macro F1."""
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from classifier_analysis import macro_f1_from_matrices, paired_bootstrap
from train_human_cv import LABELS


class AnalysisTests(TestCase):
    def test_macro_f1_matches_sklearn_with_absent_classes(self):
        gold = [LABELS[0], LABELS[0], LABELS[1], LABELS[2]]
        pred = [LABELS[0], LABELS[1], LABELS[0], LABELS[2]]
        cm = confusion_matrix(gold, pred, labels=LABELS)
        expected = f1_score(gold, pred, labels=LABELS, average='macro', zero_division=0)
        np.testing.assert_allclose(macro_f1_from_matrices(np.stack([cm, cm])), [expected, expected])

    def test_paired_identical_predictions_have_zero_delta(self):
        frame = pd.DataFrame(dict(document_id=['a', 'a', 'b', 'c'],
                                  true_label=[LABELS[0], LABELS[1], LABELS[2], LABELS[3]],
                                  predicted_label=[LABELS[1], LABELS[1], LABELS[2], LABELS[0]]))
        results = paired_bootstrap({name: frame.copy() for name in ['baseline', 'baseline_original_labels', 'A0_text']}, repeats=40)
        for result in results:
            self.assertEqual(result['observed_macro_f1_delta'], 0)
            self.assertEqual(result['document_bootstrap_95_ci'], [0, 0])
