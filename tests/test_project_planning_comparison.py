"""Planning comparison holds model inputs fixed and never invents quality scores."""
from copy import deepcopy
import unittest

from extraction import extract_document
from planning import generate_plan
from psai_project.comparison import compare_plans


class PlanningComparisonTests(unittest.TestCase):
    def setUp(self):
        self.text = 'We will implement a collector.\nWe will create a dashboard.\n'
        self.record = extract_document('case', self.text, use_model=False)
        self.record['related_context'] = []
        self.baseline = generate_plan(self.record, source_text=self.text)
        self.retrieval = {'mode': 'bm25', 'hits': []}

    def compare(self, selected):
        return compare_plans(self.record, self.text, self.retrieval, self.baseline, selected)

    def test_fixed_inputs_and_no_quality_claim(self):
        result = self.compare(self.baseline)
        self.assertFalse(result['selection_changed'])
        self.assertFalse(result['relative_order_changed'])
        self.assertEqual(result['baseline'], result['selected'])
        self.assertIsNone(result['quality_score'])
        self.assertFalse(result['independent_evaluation'])
        self.assertEqual(len(result['fixed_inputs']['extraction_sha256']), 64)

    def test_reordering_and_selection_have_separate_meaning(self):
        selected = deepcopy(self.baseline)
        selected['steps'].reverse()
        for index, step in enumerate(selected['steps'], 1):
            step['step'] = index
        result = self.compare(selected)
        self.assertFalse(result['selection_changed'])
        self.assertTrue(result['relative_order_changed'])
        selected['steps'] = selected['steps'][:1]
        result = self.compare(selected)
        self.assertTrue(result['selection_changed'])
        self.assertFalse(result['relative_order_changed'])

    def test_modified_baseline_and_unrelated_task_are_rejected(self):
        forged = deepcopy(self.baseline)
        forged['steps'][0]['description'] = 'Pay an invented budget'
        with self.assertRaises(ValueError):
            compare_plans(self.record, self.text, self.retrieval, forged, self.baseline)
        with self.assertRaises(ValueError):
            self.compare(forged)

    def test_source_preservation_is_not_counted_as_model_selection(self):
        selected = deepcopy(self.baseline)
        selected['planning_method'] = 'local_llm_evidence_catalog_v2'
        selected['steps'][0]['selection_origin'] = 'llm_selection'
        selected['steps'][1]['selection_origin'] = 'source_catalog_preserved'
        result = self.compare(selected)
        self.assertEqual(result['baseline']['model_selected_tasks'], 0)
        self.assertEqual(result['selected']['model_selected_tasks'], 1)
        self.assertEqual(result['selected']['source_preserved_tasks'], 1)


if __name__ == '__main__':
    unittest.main()
