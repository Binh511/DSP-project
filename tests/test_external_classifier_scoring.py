from copy import deepcopy
import hashlib
import unittest

from extraction import sentences_with_context
from psai_project.study_metrics import LABELS
from tools.score_external_classifier import aligned_labels


class ExternalClassifierScoringTests(unittest.TestCase):
    def setUp(self):
        self.source = 'We will implement the inventory API.\n'
        item = list(sentences_with_context(self.source))[0]
        self.reference = [{'source_sha256': hashlib.sha256(self.source.encode()).hexdigest(), 'text': item['text'],
                           'char_start': item['start'], 'char_end': item['end'], 'label': 'ACTIVITY'}]
        self.prediction = {'mode': 'ensemble', 'labels': list(LABELS), 'items': [{
            'input': {'text': item['text'], 'section': item['section'], 'prev_text': item['prev'],
                      'pos': item['pos'], 'next_text': item['next_text']},
            'probabilities': [.10, .10, .6996, .05, .0504], 'label': 'ACTIVITY'}]}

    def test_score_rounding_reproduces_existing_threshold_policy(self):
        truth, raw, thresholded = aligned_labels(self.source, self.reference, self.prediction)
        self.assertEqual((truth, raw, thresholded), (['ACTIVITY'], ['ACTIVITY'], ['ACTIVITY']))
        self.prediction['items'][0]['probabilities'] = [.10, .10, .6994, .05, .0506]
        self.assertEqual(aligned_labels(self.source, self.reference, self.prediction)[2], ['OTHER'])

    def test_misaligned_context_and_forged_label_fail(self):
        for key in ['context', 'label', 'reference']:
            prediction, reference = deepcopy(self.prediction), deepcopy(self.reference)
            if key == 'context':
                prediction['items'][0]['input']['text'] = 'Different source.'
            elif key == 'label':
                prediction['items'][0]['label'] = 'OTHER'
            else:
                reference[0]['char_start'] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                aligned_labels(self.source, reference, prediction)


if __name__ == '__main__':
    unittest.main()
