"""Integrity tests for research controls. No checkpoint loading or full training."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from study_data import agreement, read_json, write_json
from study_data import validate_assigned_annotations
from study_evaluation import save_evaluation
from study_final_test import freeze_selection
from study_inputs import encode_items, context_for


class ResearchControlsTests(TestCase):
    def test_label_overlay_only_updates_training_folds(self):
        from study_experiments import apply_training_overlay
        with tempfile.TemporaryDirectory() as tmp:
            original = pd.DataFrame(dict(document_id=['D1', 'D2'], source_line=[1, 1], text=['Train', 'Test'], label=['OTHER', 'OTHER'], fold=[2, 0]))
            overlay = pd.DataFrame(dict(document_id=['D1'], source_line=[1], text=['Train'], annotation=['ACTIVITY']))
            path = Path(tmp) / 'overlay.csv'
            overlay.to_csv(path, index=False)
            updated = apply_training_overlay(original, original.iloc[:1].copy(), path)
            self.assertEqual(updated.label.iloc[0], 'ACTIVITY')
            self.assertEqual(original.label.iloc[0], 'OTHER')
            overlay.loc[0, ['document_id', 'text']] = ['D2', 'Test']
            overlay.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                apply_training_overlay(original, original.iloc[:1].copy(), path)

    def test_assigned_annotation_integrity_and_reviewer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = pd.DataFrame(dict(item_id=['x'], document_id=['D'], source_line=['1'], text=['Build it.'], section=['Scope'], prev_text=[''], next_text=['']))
            source.to_csv(folder / 'key.csv', index=False)
            a = source.assign(annotation='ACTIVITY', reviewer='Person A', independence_ack='yes')
            b = source.assign(annotation='ACTIVITY', reviewer='Person B', independence_ack='yes')
            a.to_csv(folder / 'a.csv', index=False)
            b.to_csv(folder / 'b.csv', index=False)
            validate_assigned_annotations(folder / 'a.csv', folder / 'b.csv', folder / 'key.csv')
            b.loc[0, 'reviewer'] = 'person a'
            b.to_csv(folder / 'b.csv', index=False)
            with self.assertRaises(ValueError):
                validate_assigned_annotations(folder / 'a.csv', folder / 'b.csv', folder / 'key.csv')
            b.loc[0, 'reviewer'] = 'Person B'
            b.loc[0, 'text'] = 'Changed'
            b.to_csv(folder / 'b.csv', index=False)
            with self.assertRaises(ValueError):
                validate_assigned_annotations(folder / 'a.csv', folder / 'b.csv', folder / 'key.csv')

    def test_text_ablation_passes_no_context_to_tokenizer(self):
        calls = []
        def tokenizer(*args, **kwargs):
            calls.append((args, kwargs))
            return {}
        encode_items(tokenizer, [{'text': 'x' * 900}], {'context_mode': 'text', 'max_len': 128})
        self.assertEqual(len(calls[0][0]), 1)
        self.assertEqual(len(calls[0][0][0][0]), 600)
        self.assertEqual(calls[0][1]['max_length'], 128)

    def test_full_context_matches_training_formatter(self):
        from train_human_cv import context
        row = dict(text='- We must implement an API in week 3.', section='Scope',
                   prev_text='Previous', next_text='Next', pos=0.4)
        self.assertEqual(context_for(row, {}), context(row['section'], row['prev_text'], row['pos'], row['text'], row['next_text']))
        with self.assertRaises(ValueError):
            context_for({**row, 'pos': float('nan')}, {})

    def test_section_ablation_does_not_include_neighbors(self):
        ctx = context_for(dict(text='test', section='Scope', prev_text='SECRET'), {'context_mode': 'section'})
        self.assertEqual(ctx, 'Section: Scope')

    def test_agreement_aligns_ids_not_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, out = [Path(tmp) / name for name in ('a.csv', 'b.csv', 'out.json')]
            pd.DataFrame(dict(item_id=['1', '2', '3'], text=['a', 'b', 'c'], annotation=['OTHER', 'ACTIVITY', 'OTHER'])).to_csv(a, index=False)
            pd.DataFrame(dict(item_id=['3', '1', '2'], text=['c', 'a', 'b'], annotation=['OTHER', 'OTHER', 'ACTIVITY'])).to_csv(b, index=False)
            result = agreement(a, b, out)
            self.assertEqual(result['agreement'], 1.)
            self.assertEqual(result['cohen_kappa'], 1.)

    def test_agreement_rejects_unfinished_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / 'a.csv'
            pd.DataFrame(dict(item_id=['a'], text=['test'], annotation=[''])).to_csv(a, index=False)
            with self.assertRaises(ValueError):
                agreement(a, a, Path(tmp) / 'out.json')

    def test_final_selection_rejects_diagnostic_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / 'metrics.json'
            write_json(metrics, dict(evaluation_kind='diagnostic', macro_f1=0.99))
            with self.assertRaises(ValueError):
                freeze_selection(Path(tmp), metrics)

    def test_nonfinite_probabilities_are_never_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = pd.DataFrame(dict(label=['OTHER']))
            with self.assertRaises(ValueError):
                save_evaluation(Path(tmp), 'invalid', rows, np.full((1, 5), np.nan), {})

    def test_prediction_writer_preserves_probabilities_and_zero_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = pd.DataFrame(dict(document_id=['a', 'b'], text=['a', 'b'], label=['OTHER', 'ACTIVITY']))
            p = np.array([[0., 0., 0., 0., 1.], [0., 0., 1., 0., 0.]])
            report = save_evaluation(Path(tmp), 'test', rows, p, dict(evaluation_kind='unit_test'))
            self.assertEqual(report['accuracy'], 1.)
            self.assertAlmostEqual(report['macro_f1'], 0.4)
            saved = pd.read_csv(Path(tmp) / 'test_predictions.csv')
            self.assertIn('prob_OTHER', saved)
            self.assertEqual(report['per_class']['REQUIREMENT']['support'], 0.)

    def test_final_evaluation_rejects_incomplete_review_gate(self):
        import study_final_test
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'final_selection.json', {})
            write_json(root / 'final_test_candidate/review_gate.json', dict(approved_for_final_test=False))
            with patch.object(study_final_test, 'OUT', root), self.assertRaises(ValueError):
                study_final_test.evaluate_final()

    def test_pipeline_scores_exact_spans_and_rejects_changed_inputs(self):
        import study_pipeline
        from study_evaluation import fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'pipeline_review'
            folder.mkdir()
            rows = pd.DataFrame(dict(document_id=['D1'], text=['Build it.'], section=['Scope'],
                                     prev_text=[''], next_text=[''], pos=[0.], char_start=[0], char_end=[9],
                                     gold_fields=['activities'], annotation_reason=['reviewed']))
            rows.to_csv(folder / 'reference_to_complete.csv', index=False)
            write_json(folder / 'protocol.json', dict(reference_rows_sha256=fingerprint(rows.drop(columns=['gold_fields', 'annotation_reason']))))
            output = [dict(document_id='D1', fields={'activities': [dict(char_start=0, char_end=9, verified=True)]})]
            write_json(folder / 'rules_outputs.json', output)
            write_json(folder / 'classifier_outputs.json', output)
            with patch.object(study_pipeline, 'OUT', root):
                result = study_pipeline.score_pipeline()
                self.assertEqual(result[0]['scores']['activities']['f1'], 1.)
                rows.loc[0, 'text'] = 'Different source'
                rows.to_csv(folder / 'reference_to_complete.csv', index=False)
                with self.assertRaises(ValueError):
                    study_pipeline.score_pipeline()
