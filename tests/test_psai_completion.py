"""Offline CLI/report integration tests; no model inference or network access."""
from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import psai


SOURCE = ('# Project\n\n## Objective\nThe goal is to improve API visibility.\n\n'
          '## Work\nWe will implement a Python API collector.\n'
          'We will document the deployment procedure.\n\n'
          '## Requirements\nThe collector must protect API credentials.\n')


class OfflineAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'proposal.md'
        self.source.write_text(SOURCE, encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def test_rules_bm25_exports_verified_evidence_html_and_structured_result_without_ml(self):
        with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'sentence_transformers': None}), \
                patch.object(socket, 'create_connection', side_effect=AssertionError('unexpected network request')):
            report = psai.analyze(self.source, self.root / 'result', 'rules', retrieval='bm25',
                                  queries=['Python API collector', 'API credentials'], top_k=2)
        folder = report.parent
        self.assertTrue((folder / 'report.html').is_file())
        analysis = json.loads((folder / 'analysis.json').read_text(encoding='utf-8'))
        result = json.loads((folder / 'retrieval.json').read_text(encoding='utf-8'))
        self.assertEqual(analysis['metadata']['classifier'], 'rules')
        self.assertFalse(analysis['metadata']['production_model_changed'])
        self.assertTrue(result['hits'])
        self.assertEqual(analysis['plan']['related_context'], result['hits'])
        for hit in result['hits']:
            snapshot = (folder / hit['citation_source']).read_text(encoding='utf-8')
            self.assertEqual(snapshot[hit['char_start']:hit['char_end']], hit['quote'])
            self.assertEqual(hashlib.sha256(snapshot.encode()).hexdigest(), hit['source_sha256'])
            self.assertEqual(hashlib.sha256((folder / hit['citation_source']).read_bytes()).hexdigest(),
                             hit['source_sha256'])
        self.assertIn('report.html', report.read_text(encoding='utf-8'))

    def test_retrieval_does_not_create_additional_plan_steps(self):
        psai.analyze(self.source, self.root / 'plain', 'rules')
        psai.analyze(self.source, self.root / 'retrieved', 'rules', retrieval='bm25', queries=['collector'])
        plain = json.loads((self.root / 'plain/plan.json').read_text(encoding='utf-8'))
        retrieved = json.loads((self.root / 'retrieved/plan.json').read_text(encoding='utf-8'))
        self.assertEqual(plain['steps'], retrieved['steps'])
        self.assertTrue(retrieved['related_context'])

    def test_existing_output_is_preserved_before_retrieval_runs(self):
        folder = self.root / 'existing'
        folder.mkdir()
        marker = folder / 'keep.txt'
        marker.write_text('existing report')
        with patch('psai_retrieval.retrieve_document', side_effect=AssertionError('unexpected search')):
            with self.assertRaises(FileExistsError):
                psai.analyze(self.source, folder, 'rules', retrieval='bm25')
        self.assertEqual(marker.read_text(), 'existing report')

    def test_forged_retrieval_quote_is_rejected_before_any_export(self):
        from psai_retrieval import retrieve_document
        payload = retrieve_document(SOURCE, 'proposal', queries=['collector'], mode='bm25')
        self.assertTrue(payload['hits'])
        payload['hits'][0]['quote'] = 'Invented unsupported quotation.'
        with patch('psai_retrieval.retrieve_document', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'differs from its source'):
                psai.analyze(self.source, self.root / 'forged', 'rules', retrieval='bm25')
        self.assertFalse((self.root / 'forged').exists())

    def test_local_resource_is_snapshotted_and_keeps_its_own_citation_offsets(self):
        resources = self.root / 'resources'
        resources.mkdir()
        resource = resources / 'guide.md'
        resource_text = '# Collector guide\nThe Python collector keeps API credentials in environment variables.\n'
        resource.write_bytes(resource_text.replace('\n', '\r\n').encode('utf-8'))
        report = psai.analyze(self.source, self.root / 'resource_result', 'rules', retrieval='bm25',
                              queries=['Python collector credentials environment variables'],
                              resource_dir=resources, top_k=4)
        payload = json.loads((report.parent / 'retrieval.json').read_text(encoding='utf-8'))
        hits = [hit for hit in payload['hits'] if hit['role'] == 'resource']
        self.assertTrue(hits)
        for hit in hits:
            self.assertTrue(hit['citation_source'].startswith('retrieval_sources/'))
            snapshot = (report.parent / hit['citation_source']).read_text(encoding='utf-8')
            self.assertEqual(snapshot, resource_text)
            self.assertEqual(snapshot[hit['char_start']:hit['char_end']], hit['quote'])
        resource.write_text('changed after export', encoding='utf-8')
        self.assertEqual((report.parent / hits[0]['citation_source']).read_text(encoding='utf-8'), resource_text)

    def test_resource_citation_cannot_escape_configured_directory(self):
        from psai_retrieval import retrieve_document
        payload = retrieve_document(SOURCE, 'proposal', queries=['collector'], mode='bm25')
        hit = payload['hits'][0]
        hit.update(role='resource', source_file=str(self.source))
        resources = self.root / 'resources'
        resources.mkdir()
        with patch('psai_retrieval.retrieve_document', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'outside'):
                psai.analyze(self.source, self.root / 'escaped', 'rules', retrieval='bm25', resource_dir=resources)
        self.assertFalse((self.root / 'escaped').exists())

    def test_real_retrieval_failure_propagates_without_silent_fallback(self):
        with patch('psai_retrieval.retrieve_document', side_effect=RuntimeError('genuine engine failure')):
            with self.assertRaisesRegex(RuntimeError, 'genuine engine failure'):
                psai.main(['analyze', '--input', str(self.source), '--output', str(self.root / 'failed'),
                           '--classifier', 'rules', '--retrieval', 'bm25'])
        self.assertFalse((self.root / 'failed').exists())

    def test_invalid_cli_combination_is_actionable_and_does_not_create_output(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            psai.main(['analyze', '--input', str(self.source), '--output', str(self.root / 'bad'),
                       '--classifier', 'rules', '--query', 'collector'])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('--retrieval', errors.getvalue())
        self.assertFalse((self.root / 'bad').exists())

    def test_doctor_reports_package_independent_paths_without_loading_models(self):
        with patch.dict(sys.modules, {'torch': None, 'transformers': None, 'sentence_transformers': None}):
            result = psai.doctor()
        self.assertTrue(result['capabilities']['markdown_text_rules']['available'])
        self.assertTrue(result['capabilities']['offline_bm25']['available'])
        self.assertFalse(result['capabilities']['production_ensemble']['model_load_verified'])


if __name__ == '__main__':
    unittest.main()
