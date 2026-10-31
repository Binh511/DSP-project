"""CPU inference API checks with synthetic tokenizer/model; no checkpoints."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import classifier_predict as serving


class Encoding(dict):
    def to(self, device):
        return self


class FakeModel(torch.nn.Module):
    def __init__(self, labels, bad=None):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.bad = bad
        self.config = SimpleNamespace(num_labels=5, id2label=dict(enumerate(labels)),
                                      label2id={label: i for i, label in enumerate(labels)})
        self.calls = []

    def forward(self, input_ids):
        self.calls.append((self.training, torch.is_inference_mode_enabled(), self.parameter.dtype))
        logits = torch.zeros((len(input_ids), 5), dtype=torch.float32)
        if self.bad == 'nan':
            logits[0, 0] = float('nan')
        elif self.bad == 'shape':
            logits = logits[:, :4]
        return SimpleNamespace(logits=logits)


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(labels=list(serving.LABELS), input_protocol='study-v1', context_mode='full',
                           max_len=256, text_max_chars=600, other_bias=2.)
        self.save_config()
        self.calls = []
        def tokenizer(texts, contexts=None, **kwargs):
            self.calls.append((texts, contexts, kwargs))
            return Encoding(input_ids=torch.zeros((len(texts), 1), dtype=torch.long))
        self.tokenizer = tokenizer

    def save_config(self):
        (self.root / 'inference_config.json').write_text(json.dumps(self.config))
        labels = self.config['labels']
        model = dict(id2label={str(i): label for i, label in enumerate(labels)},
                     label2id={label: i for i, label in enumerate(labels)})
        (self.root / 'config.json').write_text(json.dumps(model))

    def predict(self, records, *, bad=None, batch_size=8):
        model = FakeModel(self.config['labels'], bad).eval()
        with patch.object(serving, '_load_checkpoint', return_value=(self.tokenizer, model, torch, 'cpu')):
            result = serving.predict(records, self.root, batch_size)
        return result, model

    def test_empty_input_does_not_touch_checkpoint_or_load_torch(self):
        with patch.object(serving, '_load_checkpoint', side_effect=AssertionError('loaded')), \
             patch.object(serving, '_checkpoint_config', side_effect=AssertionError('read model')):
            self.assertEqual(serving.predict([], 'nonexistent-local-path'), [])
        code = "import sys; import classifier_predict; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(serving.__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_probabilities_bias_shared_encoding_fp32_and_optional_ids(self):
        rows = [dict(id='first', item_id='assigned-1', text='x' * 900, section='Scope',
                     prev_text='Earlier', next_text='Later', pos=.4), dict(text='We will build it.')]
        result, model = self.predict(rows, batch_size=1)
        self.assertEqual(len(self.calls[0][0][0]), 600)
        self.assertIn('Section: Scope | Pos: 4', self.calls[0][1][0])
        self.assertEqual(self.calls[0][2]['truncation'], 'longest_first')
        self.assertEqual(result[0]['id'], 'first')
        self.assertEqual(result[0]['item_id'], 'assigned-1')
        self.assertEqual(result[1]['index'], 1)
        self.assertEqual(result[0]['label'], 'OTHER')
        self.assertEqual(set(result[0]['probabilities']), set(serving.LABELS))
        self.assertAlmostEqual(sum(result[0]['probabilities'].values()), 1., places=6)
        self.assertTrue(all(call == (False, True, torch.float32) for call in model.calls))

    def test_reordered_saved_labels_are_reported_in_canonical_mapping(self):
        self.config['labels'] = list(reversed(serving.LABELS))
        self.save_config()
        result, _ = self.predict([dict(text='Build the project.')])
        self.assertEqual(result[0]['label'], 'OTHER')
        self.assertEqual(list(result[0]['probabilities']), list(serving.LABELS))

    def test_invalid_records_and_batch_sizes_fail_before_loading(self):
        invalid = [[dict(text='')], [dict(text='ok', pos=float('nan'))],
                   [dict(text='ok', pos=1.1)], [dict(text='ok', pos=True)],
                   [dict(text='ok', section=None)], [dict(text=3)], [dict(text='ok', id=[])], {}]
        with patch.object(serving, '_load_checkpoint', side_effect=AssertionError('loaded')):
            for rows in invalid:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    serving.predict(rows, self.root)
            for size in (0, -1, True, 1.5):
                with self.subTest(batch_size=size), self.assertRaises(ValueError):
                    serving.predict([], self.root, size)

    def test_invalid_checkpoint_protocol_and_label_mapping_are_rejected(self):
        marker = self.root / 'invalid_checkpoint.json'
        marker.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'marked invalid'):
            self.predict([dict(text='Build it.')])
        marker.unlink()
        self.config['input_protocol'] = 'legacy'
        self.save_config()
        with self.assertRaisesRegex(ValueError, 'study-v1'):
            self.predict([dict(text='Build it.')])
        self.config['input_protocol'] = 'study-v1'
        self.save_config()
        model_config = json.loads((self.root / 'config.json').read_text())
        model_config['id2label']['0'] = 'OTHER'
        (self.root / 'config.json').write_text(json.dumps(model_config))
        with self.assertRaisesRegex(ValueError, 'label mapping'):
            self.predict([dict(text='Build it.')])

    def test_nan_logits_and_wrong_shapes_are_never_exported(self):
        for bad in ('nan', 'shape'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'logits'):
                self.predict([dict(text='Build the project.')], bad=bad)

    def test_loader_is_local_fp32_and_evaluation_only(self):
        model = FakeModel(self.config['labels'])
        tokenizer_factory = SimpleNamespace(from_pretrained=MagicMock(return_value=self.tokenizer))
        model_factory = SimpleNamespace(from_pretrained=MagicMock(return_value=model))
        stub = SimpleNamespace(AutoTokenizer=tokenizer_factory, AutoModelForSequenceClassification=model_factory)
        with patch.dict('sys.modules', {'transformers': stub}), patch.object(torch.cuda, 'is_available', return_value=False):
            _, loaded, _, device = serving._load_checkpoint(self.root, self.config)
        self.assertEqual(device, 'cpu')
        self.assertFalse(loaded.training)
        for factory in (tokenizer_factory, model_factory):
            self.assertTrue(factory.from_pretrained.call_args.kwargs['local_files_only'])
            self.assertFalse(factory.from_pretrained.call_args.kwargs['trust_remote_code'])
        self.assertIs(model_factory.from_pretrained.call_args.kwargs['dtype'], torch.float32)

    def test_json_jsonl_empty_and_malformed_inputs_are_explicit(self):
        rows = [dict(text='Build the project.'), dict(text='Test the project.')]
        for filename, text in [('input.json', json.dumps(rows)), ('input.jsonl', '\n'.join(json.dumps(r) for r in rows))]:
            path = self.root / filename
            path.write_text(text)
            self.assertEqual(serving.read_records(path), rows)
        path.write_text('')
        self.assertEqual(serving.read_records(path), [])
        path = self.root / 'bad.json'
        path.write_text(json.dumps(rows[0]))
        with self.assertRaises(ValueError):
            serving.read_records(path)

    def test_atomic_writer_never_overwrites_or_leaves_partial_on_failure(self):
        output = self.root / 'result.json'
        serving.write_predictions(output, [])
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            serving.write_predictions(output, [dict(label='OTHER')])
        self.assertEqual(output.read_bytes(), before)
        broken = self.root / 'broken.json'
        with self.assertRaises(ValueError):
            serving.write_predictions(broken, [dict(confidence=float('nan'))])
        self.assertFalse(broken.exists())
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_cli_existing_output_refuses_before_loading(self):
        output = self.root / 'already.json'
        output.write_text('preserve')
        with patch.object(serving, 'predict', side_effect=AssertionError('loaded')), contextlib.redirect_stderr(io.StringIO()):
            code = serving.main(['--model-dir', str(self.root), '--input', 'missing.json', '--output', str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main(verbosity=2)
