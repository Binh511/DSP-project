"""CPU regression checks: no download, production checkpoints or long training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from classifier_training import WeightedClassificationTrainer


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
        self.config = SimpleNamespace()

    def forward(self, input_ids, labels=None, **kwargs):
        return SequenceClassifierOutput(logits=self.linear(input_ids.float()))


def update(batch, accumulation, legacy=False, count=8):
    torch.manual_seed(7)
    model = TinyClassifier()
    data = [{'input_ids': torch.tensor([i / 8, 1., -0.5]), 'labels': i % 5}
            for i in range(count)]
    before = model.linear.weight.detach().clone()
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, use_cpu=True, max_steps=1,
                                 per_device_train_batch_size=batch,
                                 gradient_accumulation_steps=accumulation,
                                 learning_rate=0.05, optim='sgd', max_grad_norm=0,
                                 weight_decay=0, save_strategy='no', report_to='none',
                                 disable_tqdm=True, seed=11)
        trainer = WeightedClassificationTrainer(model=model, args=args,
                     train_dataset=data, class_weights=[1., 2., 0.7, 1.5, 0.5])
        if legacy:
            # Reproduces Trainer introspection of the previous custom loss:
            # the model accepts **kwargs but the custom loss ignores item counts.
            trainer.model_accepts_loss_kwargs = True
        trainer.train()
    return before - model.linear.weight.detach()


class FoundationTests(unittest.TestCase):
    def test_accumulation_matches_full_batch(self):
        full = update(8, 1)
        accumulated = update(1, 8)
        torch.testing.assert_close(full, accumulated, atol=1e-6, rtol=1e-5)
        legacy = update(1, 8, legacy=True)
        torch.testing.assert_close(legacy, full * 8, atol=1e-6, rtol=1e-5)

    def test_partial_accumulation_matches_full_batch(self):
        torch.testing.assert_close(update(3, 1, count=3), update(1, 8, count=3),
                                   atol=1e-6, rtol=1e-5)

    def test_loss_keeps_input_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = WeightedClassificationTrainer(model=TinyClassifier(),
                args=TrainingArguments(output_dir=tmp, use_cpu=True, report_to='none'),
                class_weights=[1, 2, 1, 1, 1])
            data = {'input_ids': torch.zeros(1, 3), 'labels': torch.tensor([1])}
            loss = trainer.compute_loss(trainer.model, data)
            self.assertIn('labels', data)
            self.assertTrue(torch.isfinite(loss))

    def test_nonfinite_loss_stops_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = WeightedClassificationTrainer(model=TinyClassifier(),
                args=TrainingArguments(output_dir=tmp, use_cpu=True, report_to='none'),
                class_weights=[1, 1, 1, 1, 1])
            with patch.object(trainer.model, 'forward', return_value=SequenceClassifierOutput(logits=torch.full((1, 5), float('nan')))):
                with self.assertRaises(FloatingPointError):
                    trainer.compute_loss(trainer.model, {'input_ids': torch.zeros(1, 3), 'labels': torch.tensor([0])})

    def test_ensemble_applies_bias_and_training_text_limit(self):
        import ensemble_classifier as ens
        seen = []

        class Enc(dict):
            def to(self, device):
                return self

        def tokenizer(texts, contexts, **kwargs):
            seen.extend(texts)
            return Enc(input_ids=torch.zeros(len(texts), 1))

        def model(**kwargs):
            return SimpleNamespace(logits=torch.zeros(len(kwargs['input_ids']), 5))

        config = {'ctx_version': 3, 'max_len': 256, 'other_bias': 2.}
        with patch.object(ens, '_members', [(config, tokenizer, model)]):
            probs = ens.classify_probs([{'text': 'a' * 900}])
        self.assertEqual(len(seen[0]), 600)
        self.assertEqual(probs.argmax(1)[0], 4)
        np.testing.assert_allclose(probs.sum(1), 1.)

    def test_empty_inference_does_not_load_models(self):
        import ensemble_classifier as ens
        import sentence_classifier as single
        with patch.object(ens, '_load', side_effect=AssertionError('loaded')):
            self.assertEqual(ens.classify_probs([]).shape, (0, 5))
        with patch.object(single, '_load', side_effect=AssertionError('loaded')):
            self.assertEqual(single.classify([]), [])

    def test_sentence_offsets_preserve_literal_section_signs(self):
        from extraction import sentences_with_context
        text = '# Scope\nWe will support §4 and e.g. Python. We will support §4 and e.g. Python.'
        items = list(sentences_with_context(text))
        self.assertEqual(len(items), 2)
        self.assertLess(items[0]['start'], items[1]['start'])
        for item in items:
            self.assertEqual(text[item['start']:item['end']], item['text'])
            self.assertIn('§4', item['text'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
