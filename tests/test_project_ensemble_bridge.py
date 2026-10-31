"""Contract tests for the smoke verifier; these tests do not execute model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_project_ensemble_rag import validate_probabilities, verify_export


LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']
SOURCE = 'We will implement a source-cited project report.'


def example_payload():
    evidence = {'evidence_id': 'fixture:S00000', 'value': SOURCE, 'quote': SOURCE,
                'char_start': 0, 'char_end': len(SOURCE), 'verified': True,
                'evidence_method': 'model_label'}
    hit = {'evidence_id': 'fixture:C00000', 'quote': SOURCE,
           'char_start': 0, 'char_end': len(SOURCE), 'verified': True,
           'citation_source': 'source.txt', 'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest()}
    return {'metadata': {'classifier': 'ensemble', 'production_model_changed': False},
            'extraction': {'extraction_method': 'deberta_classifier_plus_cue_rules',
                           'fields': {'activities': [evidence]}},
            'retrieval': {'hits': [hit]},
            'plan': {'steps': [{'description': SOURCE, 'evidence_ids': ['fixture:S00000']}],
                     'related_context': [deepcopy(hit)]}}


class EnsembleBridgeVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        (self.folder / 'source.txt').write_text(SOURCE, encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def export(self, payload):
        (self.folder / 'analysis.json').write_text(json.dumps(payload), encoding='utf-8')
        return verify_export(self.folder)

    def test_probability_receipt_covers_all_five_classes(self):
        self.assertEqual(validate_probabilities([[.1, .2, .3, .15, .25]], LABELS, 1),
                         [[.1, .2, .3, .15, .25]])
        for rows, expected in [([[.5, .5]], 1), ([[float('nan'), 0, 0, 0, 1]], 1),
                               ([[.9, .2, 0, 0, 0]], 1), ([[1, 0, 0, 0, 0]], 2),
                               ([[-.1, .1, 1, 0, 0]], 1)]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_probabilities(rows, LABELS, expected)

    def test_verified_bridge_receipt_explicitly_is_not_a_quality_metric(self):
        checks = self.export(example_payload())
        self.assertEqual(checks['model_label_evidence_count'], 1)
        self.assertTrue(checks['retrieval_context_connected_to_plan'])
        self.assertFalse(checks['research_accuracy_measured'])
        self.assertFalse(checks['human_label_quality_verified'])

    def test_rules_export_is_not_accepted_as_real_ensemble_proof(self):
        payload = example_payload()
        payload['metadata']['classifier'] = 'rules'
        with self.assertRaisesRegex(ValueError, 'production ensemble'):
            self.export(payload)

    def test_forged_extraction_or_retrieval_span_is_rejected(self):
        for kind in ('extraction', 'retrieval'):
            payload = example_payload()
            target = (payload['extraction']['fields']['activities'][0] if kind == 'extraction'
                      else payload['retrieval']['hits'][0])
            target['quote'] = 'Invented quotation.'
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'citation'):
                self.export(payload)

    def test_modified_snapshot_hash_is_rejected(self):
        payload = example_payload()
        payload['retrieval']['hits'][0]['source_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.export(payload)

    def test_retrieval_must_reach_the_plan_and_cannot_invent_steps(self):
        payload = example_payload()
        payload['plan']['related_context'] = []
        with self.assertRaisesRegex(ValueError, 'connected'):
            self.export(payload)
        payload = example_payload()
        payload['plan']['steps'][0]['description'] = 'Deploy an unsupported extra system.'
        with self.assertRaisesRegex(ValueError, 'not grounded'):
            self.export(payload)

    def test_snapshot_cannot_escape_exported_directory(self):
        payload = example_payload()
        payload['retrieval']['hits'][0]['citation_source'] = '../outside.txt'
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.export(payload)


if __name__ == '__main__':
    unittest.main()
