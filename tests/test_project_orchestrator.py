"""Serial-stage lifecycle and atomic export checks, without loading ML models."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from psai_project import runtime


SOURCE = '# Project\n\n## Work\nWe will implement a Python collector for API metrics.\n'


class ProjectOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / 'authored.md'
        self.input.write_bytes(b'\xef\xbb\xbf' + SOURCE.replace('\n', '\r\n').encode())
        self.output = self.root / 'output'
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def fake_worker(self, name, request, stage):
        self.calls.append((name, request))
        text = Path(request['source_path']).read_text('utf-8')
        self.assertEqual(text, SOURCE)
        self.assertEqual(request['source_sha256'], hashlib.sha256(SOURCE.encode()).hexdigest())
        if name == 'extract':
            from extraction import extract_document
            value = {'status': 'completed', 'record': extract_document(request['document_id'], text, use_model=False),
                     'classifier_predictions': [], 'provenance': {'mode': 'rules'}}
        elif name == 'retrieve':
            from psai_retrieval import retrieve_document
            from psai import _verified_retrieval
            retrieval = ({'mode': 'none', 'queries': [], 'hits': [], 'provenance': {'method': 'disabled'}}
                         if request['retrieval'] == 'none' else
                         retrieve_document(text, request['document_id'], queries=request['queries'],
                                           mode='bm25', top_k=request['top_k']))
            retrieval, snapshots = _verified_retrieval(retrieval, text, self.input, None)
            value = {'status': 'completed', 'retrieval': retrieval, 'source_texts': snapshots, 'provenance': {}}
        else:
            from planning import generate_plan
            record = json.loads(Path(request['record_path']).read_text('utf-8'))['record']
            retrieval = json.loads(Path(request['retrieval_path']).read_text('utf-8'))['retrieval']
            record['related_context'] = retrieval['hits']
            plan = generate_plan(record, source_text=text)
            value = {'status': 'completed', 'plan': plan, 'source_plan': plan, 'provenance': {}}
        value.update(stage=name, source_sha256=request['source_sha256'])
        jobs = stage / 'jobs'
        jobs.mkdir(exist_ok=True)
        runtime._json(jobs / f'{name}.response.json', value)
        return value

    @staticmethod
    def fake_export(stage, text, source, extraction, retrieval, planned, metadata):
        report = stage / 'report.html'
        report.write_text('<html>Authored integration fixture</html>', encoding='utf-8')
        runtime._json(stage / 'analysis.json', {'extraction': extraction, 'retrieval': retrieval,
                                               'plan': planned, 'metadata': metadata})
        return report

    def run_fixture(self, **kwargs):
        with patch.object(runtime, '_run_worker', side_effect=self.fake_worker), \
                patch('psai_project.report.export_project_report', side_effect=self.fake_export):
            return runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25',
                                        planner='evidence', queries=['Python API collector'], **kwargs)

    def test_stages_are_serial_and_export_source_hash_survives_windows_newlines(self):
        report = self.run_fixture()
        self.assertEqual(report, self.output / 'report.html')
        self.assertEqual([name for name, _ in self.calls], ['extract', 'retrieve', 'plan'])
        self.assertEqual((self.output / 'source.txt').read_bytes(), SOURCE.encode('utf-8'))
        manifest = json.loads((self.output / 'run_manifest.json').read_text('utf-8'))
        self.assertTrue(manifest['source_snapshot_unchanged'])
        self.assertTrue(manifest['input_unchanged'])
        self.assertFalse(manifest['production_model_changed'])
        self.assertFalse(manifest['final_test_evaluated'])
        self.assertEqual(len(manifest['stages']), 3)

    def test_existing_output_is_preserved_before_any_worker_runs(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('original')
        with patch.object(runtime, '_run_worker') as worker, self.assertRaises(FileExistsError):
            runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25', planner='evidence')
        worker.assert_not_called()
        self.assertEqual(marker.read_text(), 'original')

    def test_failure_preserves_diagnostics_without_completed_output(self):
        def failing(name, request, stage):
            if name == 'retrieve':
                raise RuntimeError('genuine retrieval failure')
            return self.fake_worker(name, request, stage)
        with patch.object(runtime, '_run_worker', side_effect=failing), self.assertRaisesRegex(
                runtime.ProjectStageError, 'Diagnostics preserved'):
            runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25', planner='evidence')
        self.assertFalse(self.output.exists())
        failed = list(self.root.glob('output.failed_*'))
        self.assertEqual(len(failed), 1)
        failure = json.loads((failed[0] / 'failure.json').read_text('utf-8'))
        self.assertEqual(failure['phase'], 'retrieve')
        self.assertFalse(failure['report_completed'])
        self.assertTrue((failed[0] / 'jobs/extract.response.json').is_file())
        self.assertFalse((failed[0] / 'report.html').exists())

    def test_tampered_source_snapshot_is_rejected_before_export(self):
        def tampering(name, request, stage):
            result = self.fake_worker(name, request, stage)
            if name == 'plan':
                Path(request['source_path']).write_text('tampered source', encoding='utf-8')
            return result
        with patch.object(runtime, '_run_worker', side_effect=tampering), \
                patch('psai_project.report.export_project_report') as export, \
                self.assertRaisesRegex(runtime.ProjectStageError, 'Source changed'):
            runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25', planner='evidence')
        export.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_unused_model_options_fail_before_creating_staging(self):
        with self.assertRaisesRegex(ValueError, 'require dense or hybrid'):
            runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25',
                                planner='evidence', adapter_dir=self.root)
        self.assertEqual(list(self.root.glob('.psai-project-*')), [])
        self.assertFalse(self.output.exists())

    def test_worker_has_offline_environment_and_persists_logs_and_response(self):
        stage = self.root / 'stage'
        stage.mkdir()
        observed = {}
        def process(command, **kwargs):
            observed.update(kwargs)
            response = Path(command[command.index('--response') + 1])
            runtime._json(response, {'status': 'completed', 'stage': 'extract',
                                     'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(), 'record': {}})
            kwargs['stdout'].write('real subprocess output')
            return SimpleNamespace(returncode=0)
        with patch.object(runtime.subprocess, 'run', side_effect=process):
            result = runtime._run_worker('extract', {'classifier': 'rules',
                'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest()}, stage)
        self.assertEqual(observed['env']['HF_HUB_OFFLINE'], '1')
        self.assertEqual(observed['env']['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(result['worker_execution']['returncode'], 0)
        self.assertEqual((stage / 'jobs/extract.stdout.log').read_text(), 'real subprocess output')
        self.assertEqual(json.loads((stage / 'jobs/extract.request.json').read_text())['stage'], 'extract')

    def test_completed_worker_must_match_request_stage_and_source_hash(self):
        digest = hashlib.sha256(SOURCE.encode()).hexdigest()
        cases = [('missing-stage', {'status': 'completed', 'source_sha256': digest}, 'stage identity'),
                 ('wrong-stage', {'status': 'completed', 'stage': 'plan', 'source_sha256': digest}, 'stage identity'),
                 ('missing-hash', {'status': 'completed', 'stage': 'extract'}, 'source_sha256'),
                 ('wrong-hash', {'status': 'completed', 'stage': 'extract', 'source_sha256': '0' * 64}, 'source_sha256')]
        for name, payload, message in cases:
            stage = self.root / name
            stage.mkdir()
            def process(command, **kwargs):
                runtime._json(Path(command[-1]), payload)
                return SimpleNamespace(returncode=0)
            with self.subTest(case=name), patch.object(runtime.subprocess, 'run', side_effect=process), \
                    self.assertRaisesRegex(runtime.ProjectStageError, message):
                runtime._run_worker('extract', {'source_sha256': digest}, stage)
            # Rejecting a response must not erase the evidence needed to inspect it.
            self.assertEqual(json.loads((stage / 'jobs/extract.response.json').read_text()), payload)
            self.assertEqual(json.loads((stage / 'jobs/extract.request.json').read_text())['source_sha256'], digest)

    def test_missing_request_hash_cannot_match_a_missing_response_hash(self):
        stage = self.root / 'unbound'
        stage.mkdir()
        def process(command, **kwargs):
            runtime._json(Path(command[-1]), {'status': 'completed', 'stage': 'extract'})
            return SimpleNamespace(returncode=0)
        with patch.object(runtime.subprocess, 'run', side_effect=process), \
                self.assertRaisesRegex(runtime.ProjectStageError, 'request has no valid source_sha256'):
            runtime._run_worker('extract', {}, stage)

    def test_nonzero_worker_exit_is_not_accepted_with_forged_success_response(self):
        stage = self.root / 'stage'
        stage.mkdir()
        def failing(command, **kwargs):
            runtime._json(Path(command[-1]), {'status': 'completed', 'stage': 'extract',
                'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(), 'record': {}})
            return SimpleNamespace(returncode=9)
        with patch.object(runtime.subprocess, 'run', side_effect=failing), \
                self.assertRaisesRegex(runtime.ProjectStageError, 'exit 9'):
            runtime._run_worker('extract', {'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest()}, stage)

    def test_worker_timeout_has_actionable_stage_name(self):
        stage = self.root / 'stage'
        stage.mkdir()
        with patch.object(runtime.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['worker'], 1)), \
                self.assertRaisesRegex(runtime.ProjectStageError, 'plan exceeded'):
            runtime._run_worker('plan', {'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest()}, stage)

    def test_real_rules_bm25_evidence_workers_export_grounded_report_without_ml(self):
        report = runtime.run_project(self.input, self.output, classifier='rules', retrieval='bm25',
                                     planner='evidence', queries=['Python collector API metrics'], device='cpu')
        self.assertTrue(report.is_file())
        result = json.loads((self.output / 'analysis.json').read_text('utf-8'))
        self.assertEqual(result['metadata']['classifier'], 'rules')
        self.assertEqual(result['retrieval']['mode'], 'bm25')
        self.assertTrue(result['retrieval']['hits'])
        self.assertEqual(result['plan']['related_context'], result['retrieval']['hits'])
        for name in ('extract', 'retrieve', 'plan'):
            response = json.loads((self.output / 'jobs' / f'{name}.response.json').read_text('utf-8'))
            self.assertEqual(response['status'], 'completed')
            self.assertFalse(response['training_performed'])
            self.assertNotIn('cuda', response)
        for hit in result['retrieval']['hits']:
            snapshot = (self.output / hit['citation_source']).read_text('utf-8')
            self.assertEqual(snapshot[hit['char_start']:hit['char_end']], hit['quote'])

    def test_disabled_retrieval_still_exports_main_source_and_source_plan(self):
        report = runtime.run_project(self.input, self.output, classifier='rules', retrieval='none',
                                     planner='evidence', device='cpu')
        result = json.loads((report.parent / 'analysis.json').read_text('utf-8'))
        self.assertEqual(result['retrieval']['mode'], 'none')
        self.assertEqual(result['retrieval']['hits'], [])
        self.assertTrue(result['plan']['steps'])


if __name__ == '__main__':
    unittest.main()
