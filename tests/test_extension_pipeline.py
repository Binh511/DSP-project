"""Verify the backend-to-pipeline boundary without starting learned models."""
import base64
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from psai_extension.server import JobManager


class ExtensionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manager_and_job(self, mode):
        manager = JobManager(
            self.root / mode, python_executable='fixture-python', start_worker=False,
            resource_dir=self.root / 'resources', base_model_dir=self.root / 'base-model',
            planner_model_dir=self.root / 'planner-model')
        self.addCleanup(manager.close)
        job = manager.create_job({
            'filename': 'project.md', 'mode': mode, 'device': 'cpu', 'ocr': 'none',
            'content_base64': base64.b64encode(b'# Project\nBuild a dashboard.').decode('ascii'),
        })
        # Private source metadata is passed only to the trusted pipeline runner.
        return manager, dict(manager._jobs[job['id']])

    def fake_process(self, returncode=0):
        process = MagicMock()
        process.__enter__.return_value = process
        process.stdout = iter(['PSAI stage: extract\n', 'Tài liệu được xử lý\n',
                               'PSAI stage: retrieve\n', 'PSAI stage: plan\n'])
        process.wait.return_value = returncode
        return process

    def test_legacy_patch_seam_preserves_mode_options_offline_env_and_stage_logs(self):
        for mode, expected in (('fast', ('rules', 'bm25', 'evidence')),
                               ('full', ('ensemble', 'hybrid', 'local'))):
            with self.subTest(mode=mode):
                manager, job = self.manager_and_job(mode)
                process = self.fake_process()
                with patch.dict(os.environ, {'HF_HUB_OFFLINE': '0'}), \
                        patch('psai_extension.server.subprocess.Popen', return_value=process) as launch:
                    manager._run_pipeline(job, manager.data_dir / job['id'], manager)
                    self.assertEqual(os.environ['HF_HUB_OFFLINE'], '0')
                command = launch.call_args.args[0]
                option = lambda flag: command[command.index(flag) + 1]
                self.assertEqual(command[:4], ['fixture-python', '-u', '-m', 'psai_project'])
                self.assertEqual(tuple(option(flag) for flag in ('--classifier', '--retrieval', '--planner')),
                                 expected)
                self.assertEqual(option('--device'), 'cpu')
                self.assertEqual(option('--ocr'), 'none')
                self.assertEqual(option('--resource-dir'), str((self.root / 'resources').resolve()))
                if mode == 'full':
                    self.assertEqual(option('--base-model-dir'), str((self.root / 'base-model').resolve()))
                    self.assertEqual(option('--planner-model-dir'), str((self.root / 'planner-model').resolve()))
                else:
                    self.assertNotIn('--base-model-dir', command)
                    self.assertNotIn('--planner-model-dir', command)
                environment = launch.call_args.kwargs['env']
                for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
                    self.assertEqual(environment[name], '1')
                self.assertEqual(environment['PYTHONIOENCODING'], 'utf-8')
                state = manager.get_job(job['id'])
                self.assertEqual(state['stage'], 'plan')
                self.assertIn('Tài liệu được xử lý', state['log_tail'])
                self.assertIsNone(manager._process)

    def test_pipeline_exit_failure_keeps_logs_and_releases_owned_process(self):
        manager, job = self.manager_and_job('fast')
        with patch('psai_extension.server.subprocess.Popen', return_value=self.fake_process(returncode=23)):
            with self.assertRaisesRegex(RuntimeError, '23'):
                manager._run_pipeline(job, manager.data_dir / job['id'], manager)
        self.assertIsNone(manager._process)
        self.assertIn('Tài liệu được xử lý', manager.get_job(job['id'])['log_tail'])


if __name__ == '__main__':
    unittest.main()
