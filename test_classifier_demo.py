"""Lightweight demo correctness and isolation; never load a checkpoint."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import classifier_demo as demo


class DemoTests(unittest.TestCase):
    def results(self):
        return [dict(index=0, id='example', label='ACTIVITY', confidence=1.,
                     probabilities={label: float(label == 'ACTIVITY') for label in demo.LABELS})]

    def test_saved_results_validate_ids_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps(self.results()))
            self.assertEqual(demo.read_saved([dict(id='example')], path)[0]['label'], 'ACTIVITY')
            with self.assertRaises(ValueError):
                demo.read_saved([dict(id='different')], path)
            result = self.results()
            result[0]['probabilities']['ACTIVITY'] = float('nan')
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                demo.read_saved([dict(id='example')], path)

    def test_live_gpu_is_blocked_while_training(self):
        with patch.object(demo, 'training_active', return_value=True), patch('classifier_predict.predict') as predict:
            with self.assertRaises(ValueError):
                demo.live([dict(text='Implement it.')], Path('unused'), cpu=False)
            predict.assert_not_called()

    def test_live_cpu_masks_cuda_for_demo_only_and_restores_environment(self):
        keys = ('CUDA_VISIBLE_DEVICES', 'PYTORCH_NVML_BASED_CUDA_CHECK')
        before = {key: os.environ.get(key) for key in keys}
        def predict(records, model_dir, batch_size):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'], '1')
            self.assertEqual(batch_size, 1)
            return self.results()
        with patch.object(demo, 'training_active', return_value=True), patch('classifier_predict.predict', side_effect=predict):
            self.assertEqual(len(demo.live([dict(text='Implement it.')], Path('unused'), cpu=True)), 1)
        self.assertEqual({key: os.environ.get(key) for key in keys}, before)

    def test_cpu_worker_failure_is_reported_even_without_python_traceback(self):
        # Emulate a native library terminating Python. No model or CUDA loaded.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / 'exit_worker.py'
            child.write_text("import os\nprint('worker started', flush=True)\nos._exit(7)\n")
            args = SimpleNamespace(cpu_threads=2, input=root / 'input.json', model_dir=root / 'model')
            real_popen = demo.subprocess.Popen
            observed = {}
            def start(command, **kwargs):
                observed.update(kwargs['env'])
                return real_popen([demo.sys.executable, '-u', str(child)], **kwargs)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(demo.subprocess, 'Popen', side_effect=start), redirect_stdout(stdout), \
                    redirect_stderr(stderr):
                result = demo.run_cpu_worker(args, root / 'prediction.json')
            self.assertEqual(result, 1)
            self.assertIn('worker started', stdout.getvalue())
            self.assertIn('0x00000007', stderr.getvalue())
            self.assertEqual(observed['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(observed['PYTORCH_NVML_BASED_CUDA_CHECK'], '1')
            self.assertIn('worker started', (root / 'prediction.log').read_text())

    def test_main_prints_startup_before_an_input_error(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('classifier_predict.read_records', side_effect=ValueError('bad input')), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            result = demo.main([])
        self.assertEqual(result, 1)
        self.assertIn('Starting classifier demo', stdout.getvalue())
        self.assertIn('bad input', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
