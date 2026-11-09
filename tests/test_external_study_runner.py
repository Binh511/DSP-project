import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_external_study as study


class StudyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.md'
        self.reference = self.root / 'reference.json'
        self.source.write_text('A fixed external source.', encoding='utf-8')
        self.reference.write_text('{}', encoding='utf-8')
        self.protocol = self.root / 'protocol.json'
        self.protocol.write_text(json.dumps({
            'source_files_sha256': {str(self.source): study.sha(self.source)},
            'reference_files_sha256': {str(self.reference): study.sha(self.reference)},
            'code_files_sha256': {}, 'production_model_files_sha256': {}, 'external_model_files_sha256': {},
            'modes': {'fast': {'classifier': 'rules', 'retrieval': 'bm25', 'planner': 'evidence'},
                      'full': {'classifier': 'ensemble', 'retrieval': 'hybrid', 'planner': 'local'}},
            'base_model_dir': '/frozen/base', 'planner_model_dir': '/frozen/planner',
            'device': 'cpu', 'ocr': 'none', 'top_k': 7}), encoding='utf-8')

    def test_launch_failure_is_recorded_and_next_requested_attempt_remains_visible(self):
        output = self.root / 'run'
        seen = []
        def execute(command, **kwargs):
            self.assertTrue((output / 'attempts.json').is_file())
            self.assertNotIn('timeout', kwargs)
            seen.append(command)
            if len(seen) == 1:
                raise OSError('Synthetic launch failure')
            return SimpleNamespace(returncode=2)
        with patch.object(study, 'ROOT', self.root), patch.object(study.subprocess, 'run', side_effect=execute):
            result = study.run(self.protocol, output)
        self.assertEqual(result['status'], 'completed_with_failures')
        self.assertEqual(len(result['attempts']), 2)
        self.assertIn('Synthetic launch failure', result['attempts'][0]['launch_error'])
        self.assertEqual(result['attempts'][1]['returncode'], 2)
        self.assertEqual(seen[1][seen[1].index('--top-k') + 1], '7')
        self.assertEqual(seen[1][seen[1].index('--base-model-dir') + 1], '/frozen/base')
        self.assertEqual(seen[1][seen[1].index('--planner-model-dir') + 1], '/frozen/planner')

    def test_modified_reference_prevents_any_attempt_or_output_creation(self):
        self.reference.write_text('{"changed":true}', encoding='utf-8')
        output = self.root / 'run'
        with patch.object(study, 'ROOT', self.root), patch.object(study.subprocess, 'run') as process:
            with self.assertRaisesRegex(ValueError, 'differs from the frozen protocol'):
                study.run(self.protocol, output)
        process.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
