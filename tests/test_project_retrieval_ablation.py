import hashlib
import unittest

from extraction import extract_document
from tools.check_retrieval_ablation import compare_catalogs
from psai_planner import PlannerValidationError


class RetrievalAblationTests(unittest.TestCase):
    def test_document_only_hits_do_not_change_actual_planner_input(self):
        text = 'We will implement the project dashboard.\n'
        record = extract_document('D', text, use_model=False)
        result, _ = compare_catalogs(record, text, {'hits': [{'role': 'document'}]},
                                   {'protocol_version': 'psai-local-evidence-catalog-v1'})
        self.assertTrue(result['catalog_identical'])
        self.assertTrue(result['prompt_identical'])

    def test_v2_verified_document_evidence_reaches_actual_prompt(self):
        text = 'We will implement the project dashboard.\nA reviewer discusses its appearance.\n'
        record = extract_document('D', text, use_model=False)
        quote = 'A reviewer discusses its appearance.'
        start = text.index(quote)
        retrieval = {'hits': [{'role': 'document', 'doc_id': 'D', 'evidence_id': 'D:R1',
            'verified': True, 'quote': quote, 'char_start': start, 'char_end': start + len(quote),
            'source_sha256': record['source_sha256']}]}
        result, _ = compare_catalogs(record, text, retrieval)
        self.assertFalse(result['catalog_identical'])
        self.assertFalse(result['prompt_identical'])
        self.assertEqual(result['tasks_with'], result['tasks_without'])
        retrieval['hits'][0]['quote'] = 'Invented instructions'
        with self.assertRaises(PlannerValidationError):
            compare_catalogs(record, text, retrieval)

    def test_external_resources_really_can_change_prompt_and_are_not_ignored(self):
        text = 'We will implement the project dashboard.\n'
        resource = 'Use an authenticated API for project metrics.'
        source_hash = hashlib.sha256(resource.encode()).hexdigest()
        record = extract_document('D', text, use_model=False)
        retrieval = {'hits': [{'role': 'resource', 'evidence_id': 'R:1', 'quote': resource,
            'verified': True, 'char_start': 0, 'char_end': len(resource), 'source_sha256': source_hash}],
            'source_texts': {source_hash: resource}}
        result, _ = compare_catalogs(record, text, retrieval)
        self.assertTrue(result['tasks_with'] == result['tasks_without'])
        self.assertFalse(result['catalog_identical'])
        self.assertFalse(result['prompt_identical'])


if __name__ == '__main__':
    unittest.main()
