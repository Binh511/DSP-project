"""Launch contracts shared by project demos and the extension backend."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from psai_project.execution import offline_environment, project_command, runtime_python


class ProjectExecutionTests(unittest.TestCase):
    def test_fast_and_full_commands_preserve_pipeline_choices_and_argument_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'proposal with spaces.md', root / 'new report'
            python = root / 'runtime with spaces' / 'python.exe'
            for mode, choices in (('fast', ('rules', 'bm25', 'evidence')),
                                  ('full', ('ensemble', 'hybrid', 'local'))):
                with self.subTest(mode=mode):
                    command = project_command(python, source, output, mode=mode,
                        device='cpu', ocr='none', path_options={'resource-dir': root / 'resources',
                                                               'base-model-dir': None},
                        queries=['objective requirement', 'literal --flag as query'])
                    self.assertEqual(command, [str(python), '-u', '-m', 'psai_project',
                        '--input', str(source.resolve()), '--output', str(output.resolve()),
                        '--device', 'cpu', '--ocr', 'none', '--classifier', choices[0],
                        '--retrieval', choices[1], '--planner', choices[2],
                        '--resource-dir', str((root / 'resources').resolve()),
                        '--query', 'objective requirement', '--query', 'literal --flag as query'])

    def test_unknown_mode_cannot_silently_select_full_models(self):
        with self.assertRaisesRegex(ValueError, 'fast or full'):
            project_command(sys.executable, Path('source.md'), Path('result'), mode='typo')

    def test_explicit_paths_in_fast_demo_are_not_silently_discarded(self):
        # The runtime must still see and reject unused model configuration.
        model = Path('explicit local planner')
        command = project_command(sys.executable, Path('source.md'), Path('result'), mode='fast',
                                  path_options={'planner-model-dir': model})
        self.assertEqual(command[-2:], ['--planner-model-dir', str(model.resolve())])

    def test_offline_environment_overrides_network_flags_without_mutating_caller(self):
        original = {'PATH': 'keep caller search path', 'HF_HUB_OFFLINE': '0',
                    'TRANSFORMERS_OFFLINE': '0', 'TOKENIZERS_PARALLELISM': 'true'}
        environment = offline_environment(original)
        self.assertEqual(original['HF_HUB_OFFLINE'], '0')
        self.assertEqual(environment['PATH'], original['PATH'])
        self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
        self.assertEqual(environment['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(environment['HF_HUB_DISABLE_TELEMETRY'], '1')
        self.assertEqual(environment['PYTHONIOENCODING'], 'utf-8')
        self.assertEqual(environment['TOKENIZERS_PARALLELISM'], 'true')
        worker = offline_environment(original, disable_tokenizer_parallelism=True)
        self.assertEqual(worker['TOKENIZERS_PARALLELISM'], 'false')
        self.assertEqual(original['TOKENIZERS_PARALLELISM'], 'true')

    def test_default_environment_keeps_process_environment_unchanged(self):
        with patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'PSAI_TEST_SETTING': 'keep'}):
            environment = offline_environment()
            self.assertEqual(environment['PSAI_TEST_SETTING'], 'keep')
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '0')

    def test_interpreter_selection_honors_explicit_override_and_existing_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(runtime_python(root), sys.executable)
            runtime = root / '.venv/psai_runtime/Scripts/python.exe'
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            self.assertEqual(runtime_python(root), str(runtime))
            self.assertEqual(runtime_python(root, 'explicit-python'), 'explicit-python')


if __name__ == '__main__':
    unittest.main()
