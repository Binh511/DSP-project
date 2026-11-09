import unittest

from psai_project.study_metrics import agreement_metrics, classification_metrics


class StudyMetricTests(unittest.TestCase):
    def test_known_confusion_matrix_and_absent_classes_are_explicit(self):
        result = classification_metrics(['ACTIVITY', 'ACTIVITY', 'OTHER'], ['ACTIVITY', 'OTHER', 'OTHER'])
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertEqual(result['per_label']['ACTIVITY']['support'], 2)
        self.assertAlmostEqual(result['per_label']['ACTIVITY']['f1'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1_active_labels'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1_fixed_labels'], 4 / 15)
        self.assertFalse(result['reference_quality_certified'])

    def test_agreement_and_constant_label_edge_case(self):
        result = agreement_metrics(['ACTIVITY', 'OTHER'], ['ACTIVITY', 'ACTIVITY'])
        self.assertEqual(result['agreement'], .5)
        self.assertEqual(result['cohen_kappa'], 0)
        result = agreement_metrics(['OTHER'], ['OTHER'])
        self.assertIsNone(result['cohen_kappa'])

    def test_invalid_labels_or_alignment_are_rejected(self):
        for function in [classification_metrics, agreement_metrics]:
            for first, second in [([], []), (['OTHER'], []), (['INVALID'], ['OTHER'])]:
                with self.subTest(function=function.__name__, first=first), self.assertRaises(ValueError):
                    function(first, second)


if __name__ == '__main__':
    unittest.main()
