"""Stage-source and report checks; these tests use no learned model."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from psai_project.worker import execute_job
from psai_project.report import export_project_report, _ocr_images_html, _validate_local_plan


TEXT = '# Work\nWe will implement the project dashboard.\nThe system must protect API credentials.\n'


class WorkerReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.txt'
        self.source.write_text(TEXT, encoding='utf-8', newline='')
        (self.root / 'demo.md').write_text(TEXT, encoding='utf-8', newline='')
        self.common = {'source_path': str(self.source), 'source_sha256': hashlib.sha256(TEXT.encode()).hexdigest(),
                       'input_path': str(self.root / 'demo.md'), 'document_id': 'demo', 'device': 'cpu'}

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_stage_or_changed_source_is_rejected(self):
        for update in [{'stage': 'unknown'}, {'stage': 'extract', 'source_sha256': 'changed'}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                execute_job({**self.common, **update})

    def test_document_identity_must_match_input(self):
        with self.assertRaises(ValueError):
            execute_job({**self.common, 'stage': 'extract', 'document_id': 'other'})

    def test_rules_retrieval_evidence_export_keeps_exact_source_bytes(self):
        extraction = execute_job({**self.common, 'stage': 'extract', 'classifier': 'rules'})
        retrieve = execute_job({**self.common, 'stage': 'retrieve', 'retrieval': 'bm25', 'queries': ['dashboard']})
        ex_path, ret_path = self.root / 'extract.json', self.root / 'retrieve.json'
        ex_path.write_text(json.dumps(extraction), encoding='utf-8')
        ret_path.write_text(json.dumps(retrieve), encoding='utf-8')
        plan = execute_job({**self.common, 'stage': 'plan', 'planner': 'evidence',
                            'record_path': str(ex_path), 'retrieval_path': str(ret_path)})
        result = self.root / 'result'
        result.mkdir()
        html = export_project_report(result, TEXT, {'method': 'local_text'}, extraction, retrieve, plan,
            {'classifier': 'rules', 'source_text_sha256': self.common['source_sha256']})
        self.assertTrue(html.is_file())
        self.assertEqual((result / 'source.txt').read_bytes(), TEXT.encode('utf-8'))
        data = json.loads((result / 'analysis.json').read_text('utf-8'))
        self.assertEqual(data['plan']['related_context'], data['retrieval']['hits'])
        self.assertFalse(extraction['classifier_predictions']['items'])

    def test_no_retrieval_still_has_verified_main_snapshot(self):
        result = execute_job({**self.common, 'stage': 'retrieve', 'retrieval': 'none'})
        self.assertEqual(result['retrieval']['hits'], [])
        self.assertEqual(result['source_texts'][self.common['source_sha256']], TEXT)

    def test_classifier_cpu_request_is_honored_before_loading_members(self):
        import numpy as np
        seen = []
        fake = SimpleNamespace(_device='cuda', _members=None, MODEL_NAMES=[],
            MODELS_DIR=self.root, LABELS=['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'])
        fake.classify_probs = lambda items, batch_size=32: np.asarray([[.8, .1, .05, .03, .02]])
        def extract(*args, **kwargs):
            seen.append(fake._device)
            fake.classify_probs([{'text': 'Source text'}])
            return {'extraction_method': 'test_fixture', 'fields': {}}
        with patch.dict('sys.modules', {'ensemble_classifier': fake}), \
                patch('extraction.extract_document', side_effect=extract):
            result = execute_job({**self.common, 'stage': 'extract', 'classifier': 'ensemble'})
        self.assertEqual(seen, ['cpu'])
        self.assertEqual(result['provenance']['device'], 'cpu')
        self.assertEqual(fake._device, 'cuda')

    def test_ocr_image_hash_is_verified_before_embedding(self):
        path = self.root / 'page.png'
        path.write_bytes(b'original-image-fixture')
        metadata = {'pages': [{'page': 1, 'image_path': 'page.png',
                               'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
        self.assertIn('data:image/png', _ocr_images_html(self.root, metadata))
        path.write_bytes(b'tampered-image')
        with self.assertRaisesRegex(ValueError, 'hash'):
            _ocr_images_html(self.root, metadata)

    def test_export_rechecks_nested_llm_claims_against_exact_response(self):
        from psai_planner.planner import build_catalog, build_prompt, validate_response, DEFAULT_CONFIG
        from extraction import extract_document
        record = extract_document('demo', TEXT, use_model=False)
        retrieval = {'hits': []}
        catalog = build_catalog(record, TEXT, retrieval, DEFAULT_CONFIG)
        raw = json.dumps({'tasks': [{'task_id': catalog['tasks'][0]['task_id'], 'parent_task_id': None,
                                    'checklist': [], 'required_resources': []}],
                          'dependency_candidates': [], 'resource_references': []})
        messages = build_prompt(catalog)
        digest = lambda value: hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                                separators=(',', ':')).encode()).hexdigest()
        plan = validate_response(raw, catalog)
        plan.update({'raw_response': raw, 'messages': messages, 'catalog': catalog,
                     'provenance': {'config': DEFAULT_CONFIG, 'model_loaded': True,
                                    'catalog_sha256': digest(catalog), 'prompt_sha256': digest(messages)}})
        _validate_local_plan(plan, record, TEXT, retrieval, {})
        tampered = deepcopy(plan)
        evidence = record['fields']['technologies'][0]['evidence_id']
        tampered['steps'][0]['required_resources'] = [{'description': 'Pay 5000 USD for invented work',
                                                     'evidence_ids': [evidence], 'status': 'source_excerpt'}]
        with self.assertRaisesRegex(ValueError, 'altered'):
            _validate_local_plan(tampered, record, TEXT, retrieval, {})
        tampered = deepcopy(plan)
        tampered['steps'][0]['parent_task_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'altered'):
            _validate_local_plan(tampered, record, TEXT, retrieval, {})


if __name__ == '__main__':
    unittest.main()
