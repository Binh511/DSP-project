"""End-to-end local report checks and synthetic final-gate regression checks."""
import json
from pathlib import Path
import tempfile
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from psai import analyze
from pdf_preprocessing import extract_pdf
from final_test_integrity import (inference_manifest, validate_validation_binding,
                                 known_development_artifacts, verify_screening, verify_candidate_sources)


class ProjectRuntimeTests(unittest.TestCase):
    def test_rules_pipeline_exports_exact_source_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input.md'
            source.write_text('# Work\nWe will implement a Python API.\nWe will document the API.\n', encoding='utf-8')
            report = analyze(source, root / 'result', 'rules')
            self.assertTrue(report.is_file())
            plan = json.loads((report.parent / 'plan.json').read_text(encoding='utf-8'))
            self.assertEqual(plan['grounding_validation'], 'source_hash_and_spans')
            self.assertTrue(plan['steps'])
            self.assertEqual((report.parent / 'source.txt').read_text(), source.read_text())
            before = report.read_bytes()
            with self.assertRaises(FileExistsError):
                analyze(source, root / 'result', 'rules')
            self.assertEqual(report.read_bytes(), before)

    def test_pdf_page_spans_and_empty_page_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'sample.pdf'
            source.write_bytes(b'%PDF-synthetic-test')
            reader = SimpleNamespace(is_encrypted=False, pages=[
                SimpleNamespace(extract_text=lambda: 'First page.'),
                SimpleNamespace(extract_text=lambda: ''),
                SimpleNamespace(extract_text=lambda: 'Third page.')])
            result = extract_pdf(source, reader_factory=lambda _: reader)
            for page in result['pages']:
                span = result['text'][page['char_start']:page['char_end']]
                self.assertEqual(bool(span.strip()), page['selectable_text_present'])
            self.assertFalse(result['pdf_visual_review_complete'])

    def test_report_accepts_normalized_table_values_but_cites_original_quote(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'table.md'
            source.write_text('# Plan\n| We will implement a Python API | Week 1 |\n', encoding='utf-8')
            report = analyze(source, root / 'result', 'rules')
            extraction = json.loads((report.parent / 'extraction.json').read_text(encoding='utf-8'))
            evidence = [item for values in extraction['fields'].values() for item in values]
            self.assertTrue(evidence)
            text = source.read_text()
            for item in evidence:
                self.assertEqual(text[item['char_start']:item['char_end']], item['quote'])

    def test_scanned_and_encrypted_pdf_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'sample.pdf'
            source.write_bytes(b'%PDF-synthetic-test')
            for reader in [SimpleNamespace(is_encrypted=True),
                           SimpleNamespace(is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: '')])]:
                with self.assertRaises(ValueError):
                    extract_pdf(source, reader_factory=lambda _, value=reader: value)

    def test_selection_binds_train_validation_and_inference_config(self):
        config = {'seed': 42, 'context_mode': 'full', 'input_protocol': 'study-v1',
                  'labels': ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'],
                  'other_bias': 0, 'ctx_version': 3}
        result = dict(evaluation_kind='validation_selection', train_sha256='train', validation_sha256='val',
                      config=config, macro_f1=.5, n=5)
        provenance = dict(train_sha256='train', validation_sha256='val', config=config)
        validate_validation_binding(result, provenance, config)
        with self.assertRaises(ValueError):
            validate_validation_binding(result, {**provenance, 'train_sha256': 'different'}, config)
        with self.assertRaises(ValueError):
            validate_validation_binding(result, provenance, {**config, 'context_mode': 'text'})

    def test_changed_bias_or_label_order_cannot_reuse_validation_score(self):
        config = {'input_protocol': 'study-v1'}
        result = dict(evaluation_kind='validation_selection', config=config, train_sha256='t',
                      validation_sha256='v', macro_f1=.5, n=1)
        provenance = dict(config=config, train_sha256='t', validation_sha256='v')
        inference = dict(**config, labels=['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'], other_bias=0, ctx_version=3)
        validate_validation_binding(result, provenance, inference)
        for changed in [{**inference, 'other_bias': 20}, {**inference, 'labels': list(reversed(inference['labels']))}]:
            with self.assertRaises(ValueError):
                validate_validation_binding(result, provenance, changed)

    def test_candidate_source_requires_matching_ids_and_exact_spans(self):
        from study_data import sha
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.md'
            source.write_text('Exact text.')
            gate = dict(document_paths={'WRONG': str(source)}, document_hashes={'WRONG': sha(source)})
            rows = pd.DataFrame(dict(document_id=['RIGHT'], text=['Exact text.'], char_start=[0], char_end=[11]))
            with self.assertRaises(ValueError):
                verify_candidate_sources(gate, rows)
            gate = dict(document_paths={'RIGHT': str(source)}, document_hashes={'RIGHT': sha(source)})
            verify_candidate_sources(gate, rows)
            rows.loc[0, 'char_end'] = 10
            with self.assertRaises(ValueError):
                verify_candidate_sources(gate, rows)

    def test_manifest_detects_tokenizer_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for filename in ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                             'inference_config.json', 'training_provenance.json']:
                (folder / filename).write_text('original')
            before = inference_manifest(folder)
            (folder / 'tokenizer.json').write_text('changed')
            self.assertNotEqual(inference_manifest(folder), before)

    def test_development_scan_detects_pseudo_exposure_and_unreadable_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pseudo_candidates.csv').write_text('document_id,text\nD1,known text\n')
            (root / 'label_queue.csv').write_text('document_id,text\nD2,"unclosed quote\n')
            inventory = known_development_artifacts(root)
            self.assertIn('D1', inventory['documents'])
            self.assertEqual(len(inventory['read_errors']), 1)
            screening = dict(candidate_rows_sha256='rows', development_artifact_hashes=inventory['hashes'],
                             documents=[dict(document_id='D1', disposition='APPROVED_INDEPENDENT', reviewed_by='Reviewer')])
            with self.assertRaises(ValueError):
                verify_screening(screening, 'rows', inventory, {'D1'})
            inventory['read_errors'] = []
            with self.assertRaises(ValueError):
                verify_screening(screening, 'rows', inventory, {'D1'})


class FinalGateIntegrationTests(unittest.TestCase):
    def prepare_fixture(self, root):
        from study_data import sha, write_json
        from study_evaluation import fingerprint
        import study_final_test as final
        folder, model = root / 'final_test_candidate', root / 'model'
        folder.mkdir()
        model.mkdir()
        source = root / 'source.md'
        source.write_text('Implement it.')
        for filename in ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']:
            (model / filename).write_text('synthetic unit-test fixture')
        config = dict(seed=42, context_mode='full', input_protocol='study-v1')
        inference = dict(**config, labels=['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'], other_bias=0, ctx_version=3)
        write_json(model / 'inference_config.json', inference)
        write_json(model / 'training_provenance.json', dict(max_steps=None, config=config,
                   train_sha256='train', validation_sha256='validation', train_document_ids=['TRAIN'],
                   validation_document_ids=['VAL']))
        metric = root / 'metrics.json'
        write_json(metric, dict(evaluation_kind='validation_selection', config=config, n=1,
                   macro_f1=.2, train_sha256='train', validation_sha256='validation',
                   validation_inference_manifest=inference_manifest(model),
                   model_weight_sha256=sha(model / 'model.safetensors')))
        rows = pd.DataFrame(dict(item_id=['synthetic'], document_id=['NEW'], text=['Implement it.'], label=['ACTIVITY'], char_start=[0], char_end=[13]))
        rows.to_csv(folder / 'labels_to_complete.csv', index=False)
        rows_hash = fingerprint(rows.drop(columns='label'))
        write_json(folder / 'review_gate.json', dict(approved_for_final_test=True,
                   independence_reviewed_by='Synthetic reviewer', annotation_reviewed_by='Synthetic reviewer',
                   document_paths={'NEW': str(source)}, document_hashes={'NEW': sha(source)},
                   candidate_rows_sha256=rows_hash))
        write_json(folder / 'screening.json', dict(candidate_rows_sha256=rows_hash,
                   development_artifact_hashes={}, documents=[dict(document_id='NEW', disposition='APPROVED_INDEPENDENT', reviewed_by='Synthetic reviewer')]))
        with patch.object(final, 'OUT', root):
            final.freeze_selection(model, metric)
        return model

    def test_tokenizer_change_blocks_before_any_final_inference(self):
        import study_final_test as final
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.prepare_fixture(root)
            (model / 'tokenizer.json').write_text('changed')
            with patch.object(final, 'OUT', root), patch.object(final, 'known_development_artifacts', return_value=dict(hashes={}, documents={}, read_errors=[])), patch.object(final, 'predict_local') as predict:
                with self.assertRaises(ValueError):
                    final.evaluate_final()
                predict.assert_not_called()
            self.assertFalse((root / 'final_evaluation_attempt.json').exists())

    def test_legacy_validation_without_full_manifest_cannot_freeze(self):
        import study_final_test as final
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.prepare_fixture(root)
            (root / 'final_selection.json').unlink()
            path = root / 'metrics.json'
            result = json.loads(path.read_text())
            del result['validation_inference_manifest']
            path.write_text(json.dumps(result))
            with patch.object(final, 'OUT', root), self.assertRaises(ValueError):
                final.freeze_selection(model, path)

    def test_validation_replay_writes_new_bound_result_with_mocked_model(self):
        import replay_validation as replay
        from study_evaluation import fingerprint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.prepare_fixture(root)
            experiment = root / 'experiment'
            experiment.mkdir()
            rows = pd.DataFrame(dict(document_id=['NEW'], text=['Implement it.'], label=['ACTIVITY']))
            rows.to_csv(experiment / 'validation_rows.csv', index=False)
            provenance = json.loads((model / 'training_provenance.json').read_text())
            provenance.update(validation_document_ids=['NEW'], validation_sha256=fingerprint(rows))
            (model / 'training_provenance.json').write_text(json.dumps(provenance))
            result = json.loads((root / 'metrics.json').read_text())
            result['validation_sha256'] = fingerprint(rows)
            (experiment / 'validation_metrics.json').write_text(json.dumps(result))
            with patch.object(replay, 'predict_local', return_value=(np.array([[0, 0, 1, 0, 0]]), {})):
                replay.replay(experiment, model, root / 'replay')
            saved = json.loads((root / 'replay/validation_metrics.json').read_text())
            self.assertIn('validation_inference_manifest', saved)
            self.assertEqual(saved['validation_sha256'], fingerprint(rows))

    def test_exclusive_attempt_preserves_identity_and_prevents_second_inference(self):
        import study_final_test as final
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare_fixture(root)
            with patch.object(final, 'OUT', root), patch.object(final, 'known_development_artifacts', return_value=dict(hashes={}, documents={}, read_errors=[])), patch.object(final, 'predict_local', return_value=(np.array([[0, 0, 1, 0, 0]]), {})) as predict, patch.object(final, 'save_evaluation', return_value=dict(macro_f1=.2)):
                final.evaluate_final()
                with self.assertRaises(FileExistsError):
                    final.evaluate_final()
                self.assertEqual(predict.call_count, 1)
            record = json.loads((root / 'final_evaluation_attempt.json').read_text())
            self.assertEqual(record['state'], 'complete')
            self.assertIn('selection', record)
            self.assertIn('labels_sha256', record)


class LegacyPDFBackendTests(unittest.TestCase):
    def test_incomplete_pdf_output_is_not_silently_skipped(self):
        import ocr
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / 'sample.pdf'
            pdf.write_bytes(b'pdf fixture')
            (root / 'out/sample').mkdir(parents=True)
            (root / 'out/sample/sample.md').write_text('incomplete')
            extractor = ocr.PDFBatchExtractor.__new__(ocr.PDFBatchExtractor)
            extractor.output_dir = root / 'out'
            with patch.object(ocr, 'SKIP_IF_EXISTS', True), self.assertRaises(FileExistsError):
                extractor._process_single_pdf(pdf)

    def test_converted_image_extension_matches_markdown_link(self):
        import ocr
        class Image:
            def save(self, path, format):
                Path(path).write_bytes(b'image fixture')
        output_module = ModuleType('marker.output')
        output_module.text_from_rendered = lambda _: ('![diagram](diagram.webp)', {}, {'diagram.webp': Image()})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / 'sample.pdf'
            pdf.write_bytes(b'pdf fixture')
            extractor = ocr.PDFBatchExtractor.__new__(ocr.PDFBatchExtractor)
            extractor.output_dir, extractor.use_vlm, extractor.client = root / 'out', False, None
            extractor.converter = lambda _: None
            with patch.dict('sys.modules', {'marker.output': output_module}), redirect_stdout(io.StringIO()):
                self.assertEqual(extractor._process_single_pdf(pdf), 'success')
            markdown = (root / 'out/sample/sample.md').read_text(encoding='utf-8')
            self.assertIn('images/diagram.png', markdown)
            self.assertTrue((root / 'out/sample/images/diagram.png').is_file())
            self.assertTrue(json.loads((root / 'out/sample/source_manifest.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
